"""Segmentation metrics, computed only on pixels inside the retinal FOV."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage.morphology import skeletonize


def confusion(pred, gt, fov):
    p, g = pred[fov], gt[fov]
    tp = int(np.sum(p & g)); fp = int(np.sum(p & ~g)); fn = int(np.sum(~p & g)); tn = int(np.sum(~p & ~g))
    return tp, fp, fn, tn


def from_confusion(tp, fp, fn, tn):
    d = lambda a, b: a / b if b else 0.0
    prec, rec, spec = d(tp, tp + fp), d(tp, tp + fn), d(tn, tn + fp)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"dice": d(2 * tp, 2 * tp + fp + fn), "iou": d(tp, tp + fp + fn), "precision": prec,
            "recall": rec, "specificity": spec, "f1": d(2 * prec * rec, prec + rec),
            "accuracy": d(tp + tn, tp + tn + fp + fn), "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0}


def cldice(pred, gt, fov):
    """Centreline Dice (topology-aware): harmonic mean of topology precision and sensitivity."""
    p, g = pred & fov, gt & fov
    sp, sg = skeletonize(p), skeletonize(g)
    tprec = (sp & g).sum() / max(sp.sum(), 1)
    tsens = (sg & p).sum() / max(sg.sum(), 1)
    return 2 * tprec * tsens / max(tprec + tsens, 1e-9)


def evaluate_set(probs, gts, fovs, thr=0.5, with_auc=True, with_cldice=False):
    """probs/gts/fovs: lists of HxW arrays (prob float, gt bool, fov bool).
    Returns pooled metrics (all FOV pixels together) + per-image Dice list."""
    tot = np.zeros(4, np.int64); per_img = []
    for p, g, f in zip(probs, gts, fovs):
        c = confusion(p > thr, g, f); tot += c
        per_img.append(from_confusion(*c)["dice"])
    out = from_confusion(*tot)
    out["dice_mean_per_image"] = float(np.mean(per_img))
    out["dice_std_per_image"] = float(np.std(per_img))
    if with_auc:
        y = np.concatenate([g[f] for g, f in zip(gts, fovs)])
        s = np.concatenate([p[f] for p, f in zip(probs, fovs)])
        out["roc_auc"] = float(roc_auc_score(y, s)); out["pr_auc"] = float(average_precision_score(y, s))
    if with_cldice:
        out["cldice"] = float(np.mean([cldice(p > thr, g, f) for p, g, f in zip(probs, gts, fovs)]))
    return out, per_img
