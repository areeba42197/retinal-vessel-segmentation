"""Model selection (validation only), threshold tuning (validation only), ONE-TIME test evaluation, error analysis.

  python -m src.evaluate select   # picks final model + threshold from validation data, writes configs/final.yaml
  python -m src.evaluate test     # single held-out test evaluation (refuses to run twice unless --force)
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config, project_path
from .dataset import build_arrays, drive_root, make_splits
from .metrics import evaluate_set
from .models import build_model
from .train import RESULT_COLS

H, W = 584, 565
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LOCK = project_path("experiments", "TEST_EVALUATED.lock")


def load_exp(exp):
    s = json.load(open(project_path("experiments", f"{exp}_summary.json")))
    cfg = s["config"]
    model = build_model(cfg)
    model.load_weights(project_path("models", "checkpoints", f"{exp}.weights.h5"))
    return model, cfg, s


def predict(model, X, tta=False):
    p = model.predict(X, batch_size=2, verbose=0)[..., 0]
    if tta:
        ps = [p]
        for ax in (2, 1):  # h-flip, v-flip of the (N,H,W) arrays
            q = model.predict(np.flip(X, axis=ax), batch_size=2, verbose=0)[..., 0]
            ps.append(np.flip(q, axis=ax))
        p = np.mean(ps, axis=0)
    return [x[:H, :W] for x in p]


def unpack(Y):
    return [y[:H, :W, 0] > 0.5 for y in Y], [y[:H, :W, 1] > 0.5 for y in Y]


def select():
    df = pd.read_csv(project_path("experiments", "results.csv"))
    df = df[df.experiment_id.str.match(r"^E\d+$")].copy()
    top = df.val_dice.max()
    near = df[df.val_dice >= top - 0.002]  # near-tie -> prefer the simpler (fewer-parameter) model
    params = {e: json.load(open(project_path("experiments", f"{e}_summary.json")))["params"] for e in near.experiment_id}
    final = min(params, key=params.get)
    note = f"best val_dice {top:.4f}; within 0.002 tie-band: {sorted(params)}; chose fewest params -> {final}"
    print(note)
    model, cfg, s = load_exp(final)
    root = drive_root(cfg)
    ids = make_splits(cfg)
    Xv, Yv, _ = build_arrays(root, ids["val"], cfg["preprocessing"])
    gt, fov = unpack(Yv)
    probs = predict(model, Xv)
    rows = []
    for t in THRESHOLDS:
        m, _ = evaluate_set(probs, gt, fov, t, with_auc=False)
        rows.append({"threshold": t, "val_dice": m["dice"], "val_iou": m["iou"]})
    th = pd.DataFrame(rows)
    th.to_csv(project_path("results", "tables", "threshold_sweep_val.csv"), index=False)
    best_thr = float(th.loc[th.val_dice.idxmax(), "threshold"])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(th.threshold, th.val_dice, "o-", label="Dice"); ax.plot(th.threshold, th.val_iou, "s-", label="IoU")
    ax.axvline(best_thr, ls="--", c="gray"); ax.set_xlabel("threshold"); ax.set_title(f"{final}: validation threshold sweep")
    ax.legend(); plt.tight_layout(); plt.savefig(project_path("results", "curves", "fig7_threshold_vs_dice.png"), dpi=120); plt.close()
    # TTA comparison on validation only (informational)
    tm, _ = evaluate_set(predict(model, Xv, tta=True), gt, fov, best_thr, with_auc=False)
    nm, _ = evaluate_set(probs, gt, fov, best_thr, with_auc=False)
    print(th.round(4).to_string(index=False)); print(f"best threshold {best_thr}; val Dice normal {nm['dice']:.4f} vs TTA(flips) {tm['dice']:.4f}")
    cfg["threshold"] = best_thr
    cfg["final_experiment"] = final
    yaml.safe_dump(cfg, open(project_path("configs", "final.yaml"), "w"), sort_keys=False)
    json.dump({"final": final, "threshold": best_thr, "note": note, "val_dice_at_thr": nm["dice"], "val_dice_tta": tm["dice"]},
              open(project_path("experiments", "final_selection.json"), "w"), indent=1)
    df2 = pd.read_csv(project_path("experiments", "results.csv"))
    df2.loc[df2.experiment_id == final, "threshold"] = best_thr
    df2.to_csv(project_path("experiments", "results.csv"), index=False)


# -------------------------------------------------------------------------------- test + analysis
def error_map(pred, gt, fov):
    e = np.zeros(pred.shape + (3,), np.uint8)
    e[(pred & gt) & fov] = (255, 255, 255)      # TP white
    e[(pred & ~gt) & fov] = (255, 0, 0)         # FP red
    e[(~pred & gt) & fov] = (0, 120, 255)       # FN blue
    return e                                    # TN black


def thin_thick_recall(pred, gt, fov, thin_max=1.5):
    """Recall on GT skeleton pixels split by local vessel half-width (distance transform)."""
    sk = skeletonize(gt & fov)
    dt = distance_transform_edt(gt)
    thin, thick = sk & (dt <= thin_max), sk & (dt > thin_max)
    r = lambda m: float((pred & m).sum() / max(m.sum(), 1))
    return r(thin), r(thick), int(thin.sum()), int(thick.sum())


def test(force=False):
    if os.path.exists(LOCK) and not force:
        raise SystemExit("Test set already evaluated once (lock file present). Refusing to evaluate again.")
    sel = json.load(open(project_path("experiments", "final_selection.json")))
    exp, thr = sel["final"], sel["threshold"]
    model, cfg, s = load_exp(exp)
    root = drive_root(cfg)
    ids = make_splits(cfg)
    Xs, Ys, raw = build_arrays(root, ids["test"], cfg["preprocessing"])
    gt, fov = unpack(Ys)
    probs = predict(model, Xs)
    m, per = evaluate_set(probs, gt, fov, thr, with_auc=True, with_cldice=True)
    print(f"FINAL TEST ({exp}, thr {thr}):", json.dumps({k: round(v, 4) for k, v in m.items()}))
    open(LOCK, "w").write(f"test evaluated once for {exp} thr {thr}\n")
    json.dump({"experiment": exp, "threshold": thr, "test_ids": ids["test"], "metrics": m,
               "per_image_dice": dict(zip(ids["test"], per))}, open(project_path("experiments", "final_test_metrics.json"), "w"), indent=1)
    df = pd.read_csv(project_path("experiments", "results.csv"))
    for k_csv, k in [("test_dice", "dice"), ("test_iou", "iou"), ("test_precision", "precision"), ("test_recall", "recall"),
                     ("test_specificity", "specificity"), ("test_auc", "roc_auc"), ("test_pr_auc", "pr_auc"), ("test_mcc", "mcc")]:
        df.loc[df.experiment_id == exp, k_csv] = m[k]
    df.to_csv(project_path("experiments", "results.csv"), index=False)
    # ---- error analysis (post-hoc, no tuning) ----
    rows = []
    fig, ax = plt.subplots(len(ids["test"]), 5, figsize=(16, 3.4 * len(ids["test"])))
    for k, i in enumerate(ids["test"]):
        pred = probs[k] > thr
        tp, fp, fn, tn = (int(x) for x in np.array([(pred & gt[k] & fov[k]).sum(), (pred & ~gt[k] & fov[k]).sum(),
                                                    (~pred & gt[k] & fov[k]).sum(), (~pred & ~gt[k] & fov[k]).sum()]))
        rth, rtk, nth, ntk = thin_thick_recall(pred, gt[k], fov[k])
        rows.append({"id": i, "dice": per[k], "FP": fp, "FN": fn, "recall_thin_vessels": rth, "recall_thick_vessels": rtk,
                     "n_thin_skel_px": nth, "n_thick_skel_px": ntk, "green_mean": float(raw[k][..., 1][fov[k]].mean())})
        ov = raw[k].copy(); ov[pred & fov[k]] = (0, 255, 0)
        for a, im, t in zip(ax[k], [raw[k], gt[k], probs[k] > thr, ov, error_map(pred, gt[k], fov[k])],
                            [f"image {i}", "ground truth", f"prediction (Dice {per[k]:.3f})", "overlay", "error map"]):
            a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9); a.axis("off")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="white", label="TP", ec="k"), Patch(color="black", label="TN"), Patch(color="red", label="FP"),
                        Patch(color="#0078ff", label="FN")], loc="lower center", ncol=4)
    plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.savefig(project_path("results", "error_maps", "fig5_6_test_predictions_error_maps.png"), dpi=80); plt.close()
    pd.DataFrame(rows).round(4).to_csv(project_path("results", "tables", "test_error_analysis.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for k, i in enumerate(ids["test"]):
        np.save(project_path("results", "predictions", f"test_{i}_prob.npy"), probs[k].astype(np.float16))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "test"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    select() if a.cmd == "select" else test(a.force)
