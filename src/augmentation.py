"""Synchronised image/label augmentation via a single affine warp per patch.

`lab` is HxWx2: channel 0 = vessel mask, channel 1 = FOV mask. Labels are warped with
nearest-neighbour interpolation so they stay exactly {0,1}.
"""
import cv2
import numpy as np


def extract_patch(img, lab, center, size, aug, rng):
    """Return (patch_img, patch_lab) of `size`x`size` around `center`=(cx, cy) in full-image coords."""
    cx, cy = center
    theta, scale, fx, fy = 0.0, 1.0, 1.0, 1.0
    if aug and aug.get("enabled", False):
        if aug.get("horizontal_flip") and rng.random() < 0.5:
            fx = -1.0
        if aug.get("vertical_flip") and rng.random() < 0.5:
            fy = -1.0
        rot = aug.get("rotation", 0)
        theta = np.deg2rad(rng.uniform(-rot, rot)) if rot else 0.0
        zoom = aug.get("zoom", 0)
        scale = rng.uniform(1 - zoom, 1 + zoom) if zoom else 1.0
    c, s = np.cos(theta), np.sin(theta)
    # maps destination patch coords -> source image coords
    A = np.array([[c, -s], [s, c]], np.float32) @ np.diag([fx / scale, fy / scale]).astype(np.float32)
    half = np.array([size / 2.0, size / 2.0], np.float32)
    t = np.array([cx, cy], np.float32) - A @ half
    M = np.concatenate([A, t[:, None]], axis=1)
    flags = cv2.WARP_INVERSE_MAP
    p_img = cv2.warpAffine(img, M, (size, size), flags=flags | cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    p_lab = cv2.warpAffine(lab, M, (size, size), flags=flags | cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if aug and aug.get("enabled", False):
        b, ct, g = aug.get("brightness", 0), aug.get("contrast", 0), aug.get("gamma", 0)
        if ct:
            p_img = p_img * rng.uniform(1 - ct, 1 + ct)
        if b:
            p_img = p_img + rng.uniform(-b, b)
        p_img = np.clip(p_img, 0, 1)
        if g:
            p_img = p_img ** rng.uniform(1 - g, 1 + g)
        p_img = p_img * p_lab[..., 1:2]  # keep outside-FOV at zero
    return p_img.astype(np.float32), p_lab.astype(np.float32)
