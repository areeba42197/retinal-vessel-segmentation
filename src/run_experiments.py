"""Runs the ablation sequentially (separate processes). E6's preprocessing is chosen from E3 vs E4 by VALIDATION Dice."""
import subprocess, sys, pandas as pd
from .config import project_path

def run(cfg, exp, *sets):
    cmd = [sys.executable, "-m", "src.train", "--config", cfg, "--exp", exp, "--set", *sets]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

U, E = "configs/baseline.yaml", "configs/efficientnet.yaml"
run(U, "E1", "loss=bce", "augmentation.enabled=false")
run(U, "E2", "loss=bce_dice", "augmentation.enabled=false")
run(U, "E3", "loss=bce_dice")
run(U, "E4", "loss=bce_dice", "preprocessing=clahe")
run(E, "E5", "loss=bce_dice")
df = pd.read_csv(project_path("experiments", "results.csv")).set_index("experiment_id")
pre = "clahe" if df.loc["E4", "val_dice"] > df.loc["E3", "val_dice"] else "basic"
print("E6 preprocessing chosen by val Dice (E3 vs E4):", pre, flush=True)
run(E, "E6", "loss=bce_dice", f"preprocessing={pre}")
print("ALL DONE", flush=True)
