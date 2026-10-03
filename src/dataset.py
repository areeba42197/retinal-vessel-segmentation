"""DRIVE loading, integrity report, leakage-free image-level splits, patch sampler.

NOTE: in the supplied copy of DRIVE the official *test* folder contains only images + FOV masks
(no vessel annotations), so quantitative test metrics are impossible there. We therefore split the
20 labelled training images (IDs 21-40) at IMAGE level into train / val / held-out test, once, with a
fixed seed, and store the IDs in splits/*.txt. Patches are only ever extracted *after* the split.
"""
import os
from collections import Counter

import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .augmentation import extract_patch
from .config import project_path
from .preprocessing import preprocess

PAD_MULTIPLE = 32


def drive_root(cfg=None):
    root = (cfg or {}).get("data_root", "data/DRIVE")
    return root if os.path.isabs(root) else project_path(root)


def labelled_ids(root):
    files = sorted(os.listdir(os.path.join(root, "training", "images")))
    return sorted(int(f.split("_")[0]) for f in files)


def _paths(root, i):
    d = os.path.join(root, "training")
    return (os.path.join(d, "images", f"{i}_training.tif"),
            os.path.join(d, "1st_manual", f"{i}_manual1.gif"),
            os.path.join(d, "mask", f"{i}_training_mask.gif"))


def load_sample(root, i):
    """Return (rgb uint8, vessel bool, fov bool) for labelled image id i."""
    pi, pm, pf = _paths(root, i)
    img = np.array(Image.open(pi).convert("RGB"))
    vessel = np.array(Image.open(pm)) > 127
    fov = np.array(Image.open(pf)) > 127
    return img, vessel, fov


def load_test_image(root, i):
    d = os.path.join(root, "test")
    img = np.array(Image.open(os.path.join(d, "images", f"{i:02d}_test.tif")).convert("RGB"))
    fov = np.array(Image.open(os.path.join(d, "mask", f"{i:02d}_test_mask.gif"))) > 127
    return img, fov


# ----------------------------------------------------------------------------------- report
def inspect_dataset(root, out_path=None):
    lines = ["DATASET REPORT", "--------------", "Dataset: DRIVE", f"Root: {root}", ""]
    problems = Counter()
    info = {}
    for split in ("training", "test"):
        sd = os.path.join(root, split)
        if not os.path.isdir(sd):
            continue
        sub = sorted(os.listdir(sd))
        lines.append(f"[{split}] subfolders: " + ", ".join(f"{s}({len(os.listdir(os.path.join(sd, s)))})" for s in sub))
        info[split] = sub
    lines.append("")
    ids = labelled_ids(root)
    shapes, dtypes, mvals, fvals, ch = set(), set(), set(), set(), set()
    for i in ids:
        for p in _paths(root, i):
            if not os.path.exists(p):
                problems["missing_files"] += 1
                lines.append(f"MISSING: {p}")
        try:
            pi, pm, pf = _paths(root, i)
            for p in (pi, pm, pf):
                Image.open(p).load()  # force full decode to catch truncated/corrupt data
            a, m, f = np.array(Image.open(pi)), np.array(Image.open(pm)), np.array(Image.open(pf))
            shapes.add((a.shape, m.shape, f.shape)); dtypes.add((str(a.dtype), str(m.dtype)))
            mvals |= set(np.unique(m).tolist()); fvals |= set(np.unique(f).tolist()); ch.add(a.shape[-1])
        except Exception as e:  # report, never hide
            problems["unreadable_files"] += 1
            lines.append(f"UNREADABLE id {i}: {e!r}")
    if len(ids) != len(set(ids)):
        problems["duplicate_ids"] += 1
    n_test = len(os.listdir(os.path.join(root, "test", "images"))) if os.path.isdir(os.path.join(root, "test")) else 0
    test_gt = os.path.isdir(os.path.join(root, "test", "1st_manual"))
    lines += [f"Labelled training images: {len(ids)} (IDs {ids[0]}-{ids[-1]})",
              f"Labelled vessel masks   : {len(ids)}",
              f"Official test images    : {n_test}",
              f"Official test vessel GT : {'present' if test_gt else 'NOT PRESENT (only FOV masks) -> no test metrics possible'}",
              f"(image, vessel-mask, FOV-mask) shapes: {sorted(shapes)}",
              f"Image channels: {sorted(ch)}  dtype (img, mask): {sorted(dtypes)}",
              f"Vessel mask values (raw): {sorted(mvals)}  -> binarised with >127",
              f"FOV mask values (raw)   : {sorted(fvals)}",
              f"Missing files: {problems['missing_files']}",
              f"Duplicate ids: {problems['duplicate_ids']}",
              f"Unreadable files: {problems['unreadable_files']}"]
    txt = "\n".join(lines)
    print(txt)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        open(out_path, "w").write(txt + "\n")
    return problems


# ----------------------------------------------------------------------------------- splits
def make_splits(cfg, force=False):
    """Create (or load) the fixed image-level split. IDs are saved to splits/{train,val,test}_ids.txt."""
    sd = project_path("splits")
    files = {k: os.path.join(sd, f"{k}_ids.txt") for k in ("train", "val", "test")}
    if all(os.path.exists(p) for p in files.values()) and not force:
        return {k: [int(x) for x in open(p).read().split()] for k, p in files.items()}
    ids = labelled_ids(drive_root(cfg))
    dev, test = train_test_split(ids, test_size=cfg.get("n_test", 4), random_state=cfg.get("split_seed", 42))
    train, val = train_test_split(dev, test_size=cfg.get("n_val", 4), random_state=cfg.get("split_seed", 42))
    out = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    os.makedirs(sd, exist_ok=True)
    for k, p in files.items():
        open(p, "w").write(" ".join(map(str, out[k])) + "\n")
    assert not (set(train) & set(val)) and not (set(dev) & set(test)), "LEAKAGE: overlapping splits"
    return out


# ----------------------------------------------------------------------------------- arrays
def pad_to_multiple(a, m=PAD_MULTIPLE):
    h, w = a.shape[:2]
    ph, pw = (-h) % m, (-w) % m
    pad = [(0, ph), (0, pw)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(a, pad)


def build_arrays(root, ids, mode, pad=True):
    """Full-image arrays: X (N,H,W,3) preprocessed, Y (N,H,W,2)=[vessel,fov]; also raw RGB list."""
    X, Y, raw = [], [], []
    for i in ids:
        img, v, f = load_sample(root, i)
        x = preprocess(img, f, mode)
        y = np.stack([v, f], -1).astype(np.float32)
        raw.append(img)
        X.append(pad_to_multiple(x) if pad else x)
        Y.append(pad_to_multiple(y) if pad else y)
    return np.stack(X), np.stack(Y), raw


class PatchSequence(keras.utils.PyDataset):
    """Random patches (centres sampled inside the FOV) from already-split training images."""

    def __init__(self, X, Y, patch, batch, steps, aug, seed):
        super().__init__(workers=0)
        self.X, self.Y, self.patch, self.batch, self.steps, self.aug, self.seed = X, Y, patch, batch, steps, aug, seed
        self.epoch = 0
        self.fov_pts = []
        for y in Y:
            ys, xs = np.nonzero(y[..., 1] > 0)
            self.fov_pts.append(np.stack([xs, ys], 1))

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, self.epoch, idx])
        xb, yb = [], []
        for _ in range(self.batch):
            k = rng.integers(len(self.X))
            c = self.fov_pts[k][rng.integers(len(self.fov_pts[k]))]
            px, py = extract_patch(self.X[k], self.Y[k], c, self.patch, self.aug, rng)
            xb.append(px); yb.append(py)
        return np.stack(xb), np.stack(yb)

    def on_epoch_end(self):
        self.epoch += 1
