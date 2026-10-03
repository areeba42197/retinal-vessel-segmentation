"""FOV-masked losses/metrics. y_true has 2 channels: [vessel, fov]; y_pred is the sigmoid vessel map."""
import keras
from keras import ops

EPS = 1e-6


def _split(y_true, y_pred):
    return y_true[..., 0:1], y_true[..., 1:2], y_pred


def bce_loss(y_true, y_pred):
    t, w, p = _split(y_true, y_pred)
    p = ops.clip(p, EPS, 1 - EPS)
    l = -(t * ops.log(p) + (1 - t) * ops.log(1 - p)) * w
    return ops.sum(l) / (ops.sum(w) + EPS)


def dice_loss(y_true, y_pred):
    t, w, p = _split(y_true, y_pred)
    p = p * w
    inter = ops.sum(t * p, axis=(1, 2, 3))
    denom = ops.sum(t, axis=(1, 2, 3)) + ops.sum(p, axis=(1, 2, 3))
    return 1.0 - ops.mean((2 * inter + 1.0) / (denom + 1.0))


def dice_metric(y_true, y_pred):
    """Soft Dice (1 - dice_loss), logged during training."""
    return 1.0 - dice_loss(y_true, y_pred)


def make_loss(name="bce_dice", bce_w=0.5, dice_w=0.5):
    if name == "bce":
        return bce_loss
    if name == "dice":
        return dice_loss
    if name == "bce_dice":
        def f(y_true, y_pred):
            return bce_w * bce_loss(y_true, y_pred) + dice_w * dice_loss(y_true, y_pred)
        return f
    raise ValueError(name)
