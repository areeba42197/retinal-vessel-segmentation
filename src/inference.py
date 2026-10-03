"""Inference on one retinal image:  python -m src.inference --image path.tif [--fov mask.gif] [--out results/inference]

Uses configs/final.yaml (model, preprocessing, validation-selected threshold) and the matching checkpoint.
Saves original.png, probability_map.png, binary_mask.png, overlay.png. Research/educational use only - not a diagnostic tool."""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

from .config import load_config, project_path
from .dataset import pad_to_multiple
from .models import build_model
from .preprocessing import preprocess


def estimate_fov(img):
    """Fallback FOV estimate when no mask is supplied: bright-enough pixels, largest component, filled."""
    g = cv2.GaussianBlur(img.max(axis=2), (0, 0), 3)
    m = (g > 20).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return cv2.erode(out, np.ones((9, 9), np.uint8)).astype(bool)


def load_final():
    cfg = load_config("configs/final.yaml")
    model = build_model(cfg)
    slim = project_path("models", "final_model.weights.h5")  # inference-only weights (committed to the repo)
    full = project_path("models", "checkpoints", f"{cfg['final_experiment']}.weights.h5")
    model.load_weights(slim if os.path.exists(slim) else full)
    return model, cfg


def segment(model, cfg, img_rgb, fov=None):
    fov = estimate_fov(img_rgb) if fov is None else fov
    h, w = img_rgb.shape[:2]
    x = pad_to_multiple(preprocess(img_rgb, fov, cfg["preprocessing"]))[None]
    prob = model.predict(x, verbose=0)[0, :h, :w, 0] * fov
    return prob, prob > cfg["threshold"], fov


def save_outputs(out, img, prob, mask):
    os.makedirs(out, exist_ok=True)
    ov = img.copy(); ov[mask] = (0, 255, 0)
    Image.fromarray(img).save(os.path.join(out, "original.png"))
    Image.fromarray((prob * 255).astype(np.uint8)).save(os.path.join(out, "probability_map.png"))
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out, "binary_mask.png"))
    Image.fromarray(cv2.addWeighted(img, 0.5, ov, 0.5, 0)).save(os.path.join(out, "overlay.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--fov", default=None)
    ap.add_argument("--out", default=project_path("results", "inference"))
    a = ap.parse_args()
    model, cfg = load_final()
    img = np.array(Image.open(a.image).convert("RGB"))
    fov = None if a.fov is None else np.array(Image.open(a.fov)) > 127
    prob, mask, _ = segment(model, cfg, img, fov)
    save_outputs(a.out, img, prob, mask)
    print("saved to", a.out, "| vessel fraction of image: %.3f" % mask.mean())
