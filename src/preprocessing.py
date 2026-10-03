"""Preprocessing. All modes return float32 HxWx3 in [0,1], zeroed outside the retinal field of view (FOV).

modes:
  basic : RGB / 255
  green : green channel (best vessel contrast), replicated to 3 channels
  clahe : CLAHE on the L channel of LAB, converted back to RGB
"""
import cv2
import numpy as np

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)


def preprocess(img_rgb, fov, mode="basic"):
    if mode == "basic":
        out = img_rgb.astype(np.float32) / 255.0
    elif mode == "green":
        g = img_rgb[..., 1].astype(np.float32) / 255.0
        out = np.repeat(g[..., None], 3, axis=2)
    elif mode == "clahe":
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        lab[..., 0] = clahe.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    else:
        raise ValueError(f"unknown preprocessing mode {mode}")
    return out * fov[..., None].astype(np.float32)
