"""Reproducibility helpers."""
import os
import random

import numpy as np


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        import keras
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def print_environment():
    import platform
    import tensorflow as tf
    import keras
    print("Python     :", platform.python_version())
    print("TensorFlow :", tf.__version__)
    print("Keras      :", keras.__version__)
    print("GPU        :", tf.config.list_physical_devices("GPU") or "none (CPU only)")
    print("CPU cores  :", os.cpu_count())
    try:
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]
        print("RAM        : %.1f GB" % (int(mem) / 1e6))
    except Exception:
        pass
