"""Training of one experiment: python -m src.train --config configs/baseline.yaml --exp E1 [--set key=value ...]

Model selection uses validation Dice only. The test split is never touched here."""
import argparse
import json
import os
import time

import keras
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import losses as L
from .config import load_config, project_path
from .dataset import PatchSequence, build_arrays, drive_root, make_splits
from .metrics import evaluate_set
from .models import build_model
from .seed import print_environment, set_seed

RESULT_COLS = ["experiment_id", "model", "encoder", "pretrained", "preprocessing", "augmentation", "loss", "optimizer",
               "learning_rate", "weight_decay", "batch_size", "image_size", "threshold", "best_epoch", "val_loss",
               "val_dice", "val_iou", "test_dice", "test_iou", "test_precision", "test_recall", "test_specificity",
               "test_auc", "test_pr_auc", "test_mcc", "training_time"]


class ValCallback(keras.callbacks.Callback):
    """Computes full-image validation loss/Dice/IoU each epoch, tracks best epoch, saves best weights."""

    def __init__(self, Xv, Yv, loss_fn, ckpt, warmup=10):
        super().__init__()
        self.warmup = warmup  # epochs ignored for best-checkpoint choice (early epochs can be degenerate all-vessel/all-background)
        self.Xv, self.Yv, self.loss_fn, self.ckpt = Xv, Yv, loss_fn, ckpt
        self.best, self.best_epoch, self.rows = -1.0, 0, []

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.Xv, batch_size=2, verbose=0)
        gt, fov = self.Yv[..., 0] > 0.5, self.Yv[..., 1] > 0.5
        ps, gts, fs = list(p[..., 0]), list(gt), list(fov)
        # Validation Dice = best over a fixed threshold grid (validation data only, same for every experiment), so that
        # BCE-only models whose probabilities stay below 0.5 early on are not unfairly stopped. Dice@0.5 is logged too.
        sweep = {t: evaluate_set(ps, gts, fs, t, with_auc=False)[0] for t in (0.3, 0.4, 0.5, 0.6, 0.7)}
        bt = max(sweep, key=lambda t: sweep[t]["dice"])
        m = sweep[bt]
        logs["val_loss"] = float(self.loss_fn(self.Yv, p))
        logs["val_dice"], logs["val_iou"], logs["val_thr"], logs["val_dice_at_0.5"] = m["dice"], m["iou"], bt, sweep[0.5]["dice"]
        logs["lr"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        if epoch + 1 > self.warmup and m["dice"] > self.best:
            self.best, self.best_epoch = m["dice"], epoch + 1
            self.model.save_weights(self.ckpt)
        self.rows.append({"epoch": epoch + 1, **{k: float(v) for k, v in logs.items()}})
        print(f"  ep{epoch + 1:3d} loss {logs['loss']:.4f} dice {logs.get("dice_metric", 0):.4f} | val_loss {logs['val_loss']:.4f} "
              f"val_dice {m['dice']:.4f}@{bt} (0.5: {sweep[0.5]['dice']:.4f}) val_iou {m['iou']:.4f} lr {logs['lr']:.1e}", flush=True)


def update_results(row):
    path = project_path("experiments", "results.csv")
    df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=RESULT_COLS)
    df = df[df["experiment_id"] != row["experiment_id"]]
    df = pd.concat([df, pd.DataFrame([row], columns=RESULT_COLS)], ignore_index=True)
    df.sort_values("experiment_id").to_csv(path, index=False)


def plot_curves(hist, exp, out):
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for a, (cols, t) in zip(ax, [(["loss", "val_loss"], "Loss"), (["dice_metric", "val_dice"], "Dice (train = soft Dice on patches)"),
                                 (["val_iou"], "Val IoU"), (["lr"], "Learning rate")]):
        for c in cols:
            if c in hist:
                a.plot(hist["epoch"], hist[c], label=c)
        a.set_title(f"{exp}: {t}"); a.set_xlabel("epoch"); a.legend()
        if t == "Learning rate":
            a.set_yscale("log")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def run(cfg, exp):
    set_seed(cfg["seed"])
    print_environment()
    root = drive_root(cfg)
    ids = make_splits(cfg)
    Xt, Yt, _ = build_arrays(root, ids["train"], cfg["preprocessing"], pad=True)
    Xv, Yv, _ = build_arrays(root, ids["val"], cfg["preprocessing"], pad=True)
    aug = cfg["augmentation"]
    seq = PatchSequence(Xt, Yt, cfg["patch_size"], cfg["batch_size"], cfg["steps_per_epoch"], aug, cfg["seed"])
    model = build_model(cfg)
    loss_fn = L.make_loss(cfg["loss"], cfg.get("bce_weight", 0.5), cfg.get("dice_weight", 0.5))
    opt = keras.optimizers.AdamW(learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    model.compile(opt, loss=loss_fn, metrics=[L.dice_metric])
    ckpt = project_path("models", "checkpoints", f"{exp}.weights.h5")
    es = cfg["early_stopping"]
    val_cb = ValCallback(Xv, Yv, loss_fn, ckpt, es.get("warmup", 10))
    cbs = [val_cb,
           keras.callbacks.ReduceLROnPlateau("val_loss", mode="min", factor=cfg["lr_factor"], patience=cfg["lr_patience"], min_lr=1e-6),
           keras.callbacks.EarlyStopping("val_dice", mode="max", patience=es["patience"], start_from_epoch=es.get("warmup", 10))]
    n = model.count_params()
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"[{exp}] {model.name}: params {n:,} trainable {trainable:,} | train {ids['train']} val {ids['val']}", flush=True)
    t0 = time.time()
    model.fit(seq, epochs=cfg["epochs"], callbacks=cbs, verbose=0)
    tt = time.time() - t0
    model.load_weights(ckpt)  # restore best-by-val-Dice weights
    hist = pd.DataFrame(val_cb.rows)
    hist.to_csv(project_path("experiments", f"{exp}_history.csv"), index=False)
    plot_curves(hist, exp, project_path("results", "curves", f"{exp}_curves.png"))
    best = hist[hist.epoch == val_cb.best_epoch].iloc[0]
    meta = {"experiment_id": exp, "model": cfg["model"], "encoder": "efficientnetb0" if cfg["model"] == "effnet_unet" else "",
            "pretrained": cfg.get("pretrained", False), "preprocessing": cfg["preprocessing"],
            "augmentation": bool(aug.get("enabled")), "loss": cfg["loss"], "optimizer": "AdamW",
            "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"], "batch_size": cfg["batch_size"],
            "image_size": cfg["patch_size"], "threshold": float(best["val_thr"]), "best_epoch": val_cb.best_epoch, "val_loss": best["val_loss"],
            "val_dice": best["val_dice"], "val_iou": best["val_iou"], "training_time": round(tt, 1)}
    update_results(meta)
    json.dump({**meta, "params": n, "trainable_params": trainable, "epochs_run": len(hist), "config": cfg},
              open(project_path("experiments", f"{exp}_summary.json"), "w"), indent=1, default=str)
    print(f"[{exp}] DONE best epoch {val_cb.best_epoch} val_dice {best['val_dice']:.4f} time {tt / 60:.1f} min", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like loss=bce augmentation.enabled=false")
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        v = json.loads(v) if v[:1] in "0123456789[{" or v in ("true", "false", "null") else v
        d = ov
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = v
    run(load_config(a.config, ov), a.exp)


if __name__ == "__main__":
    main()
