"""Creates per-image visual assets for the dashboard (results/gallery). Visualisation only: no metrics are computed or changed.
 - held-out test images: original / ground truth / prediction / probability heat-map / overlay / error map (using the locked threshold)
 - the 20 official DRIVE test images (no ground truth): qualitative overlay + mask only"""
import json
import os

import cv2
import numpy as np
from matplotlib import cm
from PIL import Image

from .config import load_config, project_path
from .dataset import build_arrays, drive_root, load_test_image, make_splits
from .evaluate import error_map
from .inference import load_final, segment

G = project_path("results", "gallery")
os.makedirs(os.path.join(G, "official_test"), exist_ok=True)
model, cfg = load_final()
thr = cfg["threshold"]
root = drive_root(cfg)
ids = make_splits(cfg)
H, W = 584, 565
Xs, Ys, raw = build_arrays(root, ids["test"], cfg["preprocessing"])
save = lambda a, n: Image.fromarray(np.asarray(a, np.uint8)).save(os.path.join(G, n))
for k, i in enumerate(ids["test"]):
    p = np.load(project_path("results", "predictions", f"test_{i}_prob.npy")).astype(np.float32)  # saved at test time
    gt, fov = Ys[k][:H, :W, 0] > 0.5, Ys[k][:H, :W, 1] > 0.5
    pred = (p > thr) & fov
    ov = raw[k].copy(); ov[pred] = (0, 255, 0)
    save(raw[k], f"test_{i}_original.png"); save(gt * 255, f"test_{i}_gt.png"); save(pred * 255, f"test_{i}_pred.png")
    save(cm.inferno(p * fov)[..., :3] * 255, f"test_{i}_prob.png"); save(cv2.addWeighted(raw[k], 0.5, ov, 0.5, 0), f"test_{i}_overlay.png")
    save(error_map(pred, gt, fov), f"test_{i}_error.png")
n = 0
for f in sorted(os.listdir(os.path.join(root, "test", "images"))):
    i = int(f.split("_")[0])
    img, fov = load_test_image(root, i)
    prob, mask, _ = segment(model, cfg, img, fov)
    ov = img.copy(); ov[mask] = (0, 255, 0)
    Image.fromarray(np.hstack([img, np.repeat((mask * 255).astype(np.uint8)[..., None], 3, 2), cv2.addWeighted(img, 0.5, ov, 0.5, 0)])
                    ).save(os.path.join(G, "official_test", f"{i:02d}.jpg"), quality=85)
    n += 1
json.dump({"test_ids": ids["test"], "official_test_n": n, "threshold": thr}, open(os.path.join(G, "index.json"), "w"))
print("gallery done:", len(ids["test"]), "held-out panels,", n, "official-test overlays")
