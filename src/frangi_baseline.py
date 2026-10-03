"""Classical CV baseline (NOT part of the deep model): Frangi vesselness on the inverted green channel.

The single free parameter (binarisation threshold on vesselness) is chosen on TRAIN+VAL images only to maximise Dice;
the held-out test images are then evaluated once with that fixed threshold.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
from skimage.filters import frangi

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config, project_path
from .dataset import drive_root, load_sample, make_splits
from .metrics import evaluate_set


def vesselness(img, fov):
    g = img[..., 1].astype(np.float32) / 255.0
    v = frangi(g, sigmas=[1.5, 2, 3], black_ridges=True)  # vessels are dark in the green channel
    v = v * (np.asarray(fov, np.uint8) > 0)
    # suppress the FOV border artefact: erode FOV
    from scipy.ndimage import binary_erosion
    v = v * binary_erosion(fov, iterations=8)
    return v / (np.percentile(v[fov], 99.5) + 1e-9)


def main():
    cfg = load_config("configs/baseline.yaml")
    root = drive_root(cfg)
    ids = make_splits(cfg)
    data = {i: load_sample(root, i) for i in ids["train"] + ids["val"] + ids["test"]}
    V = {i: np.clip(vesselness(data[i][0], data[i][2]), 0, 1) for i in data}
    dev = ids["train"] + ids["val"]
    best_t, best_d = None, -1
    for t in np.linspace(0.02, 0.6, 30):
        m, _ = evaluate_set([V[i] for i in dev], [data[i][1] for i in dev], [data[i][2] for i in dev], t, with_auc=False)
        if m["dice"] > best_d:
            best_t, best_d = float(t), m["dice"]
    out = {"threshold_chosen_on_train_val": best_t, "train_val_dice": best_d}
    for name, group in [("val", ids["val"]), ("test", ids["test"])]:
        m, per = evaluate_set([V[i] for i in group], [data[i][1] for i in group], [data[i][2] for i in group], best_t, with_cldice=True)
        out[name] = m
    json.dump(out, open(project_path("experiments", "frangi_baseline.json"), "w"), indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()}) for k, v in out.items()}, indent=1))
    # figure: Frangi vs ground truth (+ deep prediction if the final test predictions exist)
    import os
    fig, ax = plt.subplots(len(ids["test"]), 4, figsize=(13, 3.3 * len(ids["test"])))
    for k, i in enumerate(ids["test"]):
        img, v, f = data[i]
        p = project_path("results", "predictions", f"test_{i}_prob.npy")
        deep = np.load(p).astype(np.float32) if os.path.exists(p) else np.zeros_like(V[i])
        thr = json.load(open(project_path("experiments", "final_selection.json")))["threshold"] if os.path.exists(p) else 0.5
        for a, im, t in zip(ax[k], [img, V[i] > best_t, deep > thr, v], [f"image {i}", "Frangi (classical)", "U-Net family (final model)", "ground truth"]):
            a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(project_path("results", "comparisons", "fig9_frangi_vs_deep.png"), dpi=80); plt.close()


if __name__ == "__main__":
    main()
