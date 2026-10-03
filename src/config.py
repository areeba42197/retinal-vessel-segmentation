"""Config loading: YAML file + optional dict overrides (nested dicts are merged)."""
import copy
import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path, overrides=None):
    """Load a YAML config. Relative paths in the file are relative to the project root."""
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return deep_update(cfg, overrides)


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)
