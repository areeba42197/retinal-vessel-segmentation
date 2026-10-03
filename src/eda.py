"""EDA: python -m src.eda  -> figures in results/eda, tables in results/tables."""
import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_config, project_path
from .dataset import drive_root, inspect_dataset, labelled_ids, load_sample, make_splits

cfg = load_config("configs/baseline.yaml")
root = drive_root(cfg)
inspect_dataset(root, project_path("results", "tables", "dataset_report.txt"))
ids = labelled_ids(root)
splits = make_splits(cfg)
data = {i: load_sample(root, i) for i in ids}
E = lambda n: project_path("results", "eda", n)

# Fig 1/2: samples + pair alignment overlay
sel = ids[::4][:5]
fig, ax = plt.subplots(3, len(sel), figsize=(3.2 * len(sel), 9))
for k, i in enumerate(sel):
    img, v, f = data[i]
    ov = img.copy(); ov[v] = [0, 255, 0]
    ax[0, k].imshow(img); ax[1, k].imshow(v, cmap="gray"); ax[2, k].imshow(ov)
    ax[0, k].set_title(f"image {i}")
for a, t in zip(ax[:, 0], ["RGB", "Vessel mask", "Overlay (alignment check)"]):
    a.set_ylabel(t)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(E("fig1_samples_pairs.png"), dpi=100); plt.close()

# mask statistics (inside FOV)
rows = []
for i in ids:
    img, v, f = data[i]
    nv, nb = int((v & f).sum()), int((~v & f).sum())
    g = img[..., 1][f].astype(float)
    rows.append({"id": i, "split": next(k for k, s in splits.items() if i in s), "vessel_px": nv, "background_px": nb,
                 "vessel_pct": 100 * nv / (nv + nb), "fov_px": int(f.sum()), "green_mean": g.mean(), "green_std": g.std(),
                 "green_p5": np.percentile(g, 5), "green_p95": np.percentile(g, 95),
                 "rgb_mean_r": img[..., 0][f].mean(), "rgb_mean_g": g.mean(), "rgb_mean_b": img[..., 2][f].mean(),
                 "contrast_p95_p5": np.percentile(g, 95) - np.percentile(g, 5)})
df = pd.DataFrame(rows)
df.round(3).to_csv(project_path("results", "tables", "mask_and_intensity_stats.csv"), index=False)
print(df[["id", "split", "vessel_pct", "green_mean", "contrast_p95_p5"]].round(2).to_string(index=False))
print("mean vessel %% of FOV pixels: %.2f +- %.2f" % (df.vessel_pct.mean(), df.vessel_pct.std()))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].bar(df.id.astype(str), df.vessel_pct, color="#3b6fb6"); ax[0].set_title("Vessel pixels (% of FOV) per image")
ax[0].set_xlabel("image id"); ax[0].set_ylabel("%")
tot_v, tot_b = df.vessel_px.sum(), df.background_px.sum()
ax[1].bar(["background", "vessel"], [tot_b, tot_v], color=["#aaa", "#3b6fb6"])
ax[1].set_title(f"Class imbalance: vessel = {100 * tot_v / (tot_v + tot_b):.1f}% of FOV pixels"); ax[1].set_ylabel("pixels")
plt.tight_layout(); plt.savefig(E("fig3_class_imbalance.png"), dpi=100); plt.close()

# intensity histograms + green channel
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for c, col in enumerate(["r", "g", "b"]):
    h = np.mean([np.histogram(data[i][0][..., c][data[i][2]], 64, (0, 255), density=True)[0] for i in ids], axis=0)
    ax[0].plot(np.linspace(0, 255, 64), h, col)
ax[0].set_title("Mean RGB histogram (inside FOV)")
ax[1].bar(df.id.astype(str), df.green_mean, yerr=df.green_std, color="#2e8b57"); ax[1].set_title("Green channel mean ± std per image (illumination)")
ax[2].bar(df.id.astype(str), df.contrast_p95_p5, color="#c0762d"); ax[2].set_title("Green contrast (p95 - p5) per image")
plt.tight_layout(); plt.savefig(E("intensity_analysis.png"), dpi=100); plt.close()

# difficult examples: lowest contrast, darkest, brightest, thinnest-vessel (lowest vessel %)
hard = {"lowest contrast": df.sort_values("contrast_p95_p5").id.iloc[0], "darkest": df.sort_values("green_mean").id.iloc[0],
        "brightest": df.sort_values("green_mean").id.iloc[-1], "fewest vessels": df.sort_values("vessel_pct").id.iloc[0]}
fig, ax = plt.subplots(3, 4, figsize=(14, 10))
clahe = cv2.createCLAHE(2.0, (8, 8))
for k, (name, i) in enumerate(hard.items()):
    img, v, f = data[i]
    ax[0, k].imshow(img); ax[0, k].set_title(f"{name} (id {i})")
    ax[1, k].imshow(img[..., 1], cmap="gray")
    ax[2, k].imshow(clahe.apply(img[..., 1]), cmap="gray")
for a, t in zip(ax[:, 0], ["RGB", "Green channel", "Green + CLAHE"]):
    a.set_ylabel(t)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(E("difficult_and_green_channel.png"), dpi=90); plt.close()
print("hard examples:", {k: int(v) for k, v in hard.items()})
print("EDA done.")
