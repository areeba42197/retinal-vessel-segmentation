"""Builds the ablation table (markdown + csv) and ablation figure from experiments/results.csv. Only measured values."""
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import project_path

NAMES = {"unet": "U-Net", "effnet_unet": "EfficientNet-B0 U-Net (random init)", "attention_unet": "Attention U-Net"}
PRE = {"basic": "Basic (RGB/255)", "clahe": "CLAHE", "green": "Green channel"}
LOSS = {"bce": "BCE", "bce_dice": "BCE+Dice", "dice": "Dice"}

df = pd.read_csv(project_path("experiments", "results.csv"))
df = df[df.experiment_id.str.match(r"^E\d+$")].sort_values("experiment_id", key=lambda s: s.str[1:].astype(int))
final = json.load(open(project_path("experiments", "final_selection.json")))["final"] if os.path.exists(project_path("experiments", "final_selection.json")) else None
rows = []
for _, r in df.iterrows():
    params = json.load(open(project_path("experiments", f"{r.experiment_id}_summary.json")))
    rows.append({"Experiment": r.experiment_id + (" (FINAL)" if r.experiment_id == final else ""), "Architecture": NAMES[r.model],
                 "Preprocessing": PRE[r.preprocessing], "Loss": LOSS[r.loss], "Augmentation": "Yes" if r.augmentation else "No",
                 "Params": f"{params['params']:,}", "Best epoch": int(r.best_epoch), "Val Dice": round(r.val_dice, 4), "Val IoU": round(r.val_iou, 4),
                 "Test Dice": round(r.test_dice, 4) if pd.notna(r.test_dice) else "not evaluated (held out)"})
t = pd.DataFrame(rows)
t.to_csv(project_path("results", "tables", "ablation_table.csv"), index=False)
open(project_path("results", "tables", "ablation_table.md"), "w").write(t.to_markdown(index=False) + "\n")
print(t.to_markdown(index=False))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.bar(df.experiment_id, df.val_dice, color=["#3b6fb6" if e != final else "#c0392b" for e in df.experiment_id])
for i, v in enumerate(df.val_dice):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylim(max(0, df.val_dice.min() - 0.1), min(1, df.val_dice.max() + 0.05)); ax.set_ylabel("validation Dice")
ax.set_title("Ablation (validation set, 4 images, single seed)"); plt.tight_layout()
plt.savefig(project_path("results", "comparisons", "fig8_ablation.png"), dpi=120)
