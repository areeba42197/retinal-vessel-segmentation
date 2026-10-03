# Retinal Blood Vessel Segmentation Using Deep Learning

A reproducible U-Net-based pipeline for retinal vessel segmentation on DRIVE, with a controlled ablation study
(loss, augmentation, preprocessing, encoder), validation-only model/threshold selection, a single held-out test
evaluation, error analysis and a classical (Frangi) baseline.

> Academic/research project. It performs vessel segmentation only. It is **not** a diagnostic tool and has not been clinically validated.

## Important deviations from the original plan (read first)

1. **The supplied DRIVE "test" folder has no vessel ground truth** (only images + field-of-view masks), so it cannot be scored.
   Instead the 20 labelled images (IDs 21-40) were split **at image level, once, with a fixed seed**: 12 train / 4 val / 4 test
   (`splits/*.txt`). Test and validation sets are tiny (4 images each), so all numbers are noisy.
2. **No ImageNet weights**: the build environment could not download them, so the EfficientNet-B0 encoder is **randomly initialised**
   (`pretrained: false`). Two-stage fine-tuning was therefore not applicable. The EfficientNet results say nothing about pretrained EfficientNet.
3. **CPU-only budget** (2 cores): 128x128 patches, small U-Net (1.9 M params), 40 epochs x 30 steps, one seed per experiment. Best epochs for
   several runs were at or near the 40-epoch cap, i.e. models were probably **not fully converged**.
4. Not run: Attention U-Net (code exists in `src/models.py`, never trained), 5-fold cross-validation, CHASE_DB1 cross-dataset test (dataset not supplied),
   clDice as a loss, HD95. TTA was only compared on validation (not used for the reported test result). Notebooks were replaced by runnable scripts.

## Method

Preprocessing (basic RGB/255, green, CLAHE) -> synchronised augmentation (flips, +-15 deg rotation, +-10% zoom, brightness/contrast/gamma; labels warped with
nearest neighbour) -> U-Net or EfficientNet-B0 U-Net (sigmoid output) -> BCE / BCE+Dice loss **masked to the retinal FOV** -> full-image inference
(padded to a multiple of 32) -> threshold chosen on validation. All metrics are computed on pixels inside the FOV only.
Patches are drawn on the fly from training images only (no patch-level leakage). AdamW (lr 1e-3, wd 1e-4), ReduceLROnPlateau (on val loss),
early stopping + best checkpoint on validation Dice (first 10 epochs ignored because early epochs can be degenerate all-vessel/all-background).
"Val Dice" below = best over the thresholds {0.3,...,0.7} on the validation set (same rule for every experiment).

## Results (all measured; single seed; validation = 4 images)

| Experiment | Architecture | Preprocessing | Loss | Augmentation | Params | Val Dice | Val IoU | Test Dice |
|:--|:--|:--|:--|:--|--:|--:|--:|:--|
| E1 | U-Net | Basic | BCE | No | 1.95 M | 0.7935 | 0.6577 | held out |
| E2 | U-Net | Basic | BCE+Dice | No | 1.95 M | 0.8033 | 0.6712 | held out |
| **E3 (final)** | U-Net | Basic | BCE+Dice | Yes | 1.95 M | 0.8133 | 0.6853 | **0.8153** |
| E4 | U-Net | CLAHE | BCE+Dice | Yes | 1.95 M | 0.8138 | 0.6860 | held out |
| E5 | EffNet-B0 U-Net (random init) | Basic | BCE+Dice | Yes | 6.79 M | 0.7993 | 0.6657 | held out |
| E6 | EffNet-B0 U-Net (random init) | CLAHE | BCE+Dice | Yes | 6.79 M | 0.8062 | 0.6753 | held out |

**Selection rule:** highest validation Dice; E3 and E4 are within 0.002, so the simpler E3 (no CLAHE) was chosen. Threshold 0.45 was tuned on validation only
(`results/tables/threshold_sweep_val.csv`; Dice is flat, 0.804-0.814, over 0.30-0.70). The test set was evaluated **once** (a lock file prevents re-runs).

**Final held-out test (4 images, thr 0.45):** Dice 0.8153, IoU 0.6881, precision 0.8393, recall 0.7926, specificity 0.9771, accuracy 0.9529
(supplementary), MCC 0.7888, ROC-AUC 0.9749, PR-AUC 0.9000, clDice 0.8244. Per-image Dice 0.801-0.835.

**Classical baseline (Frangi on inverted green channel, threshold fitted on train+val):** val Dice 0.6926, test Dice 0.7213, test ROC-AUC 0.9054.

What the ablation supports (with the caveat of 4 val images and one seed): BCE->BCE+Dice (+0.010), augmentation (+0.010) and the U-Net over the randomly-initialised
EfficientNet-B0 U-Net were each consistent with the validation numbers; **CLAHE (+0.0005) is indistinguishable from no effect**. Differences of ~0.01 are
within plausible seed/split noise and should not be over-interpreted.

## Error analysis (test set, `results/error_maps/`, `results/tables/test_error_analysis.csv`)

Errors are dominated by **missed thin vessels**. Recall on skeleton pixels of thick vessels (half-width > 1.5 px) is 0.95-0.99, but only 0.49-0.70 on thin
vessels. Image 22 (Dice 0.801) and 36 (0.818) have the most false negatives (7.3k and 8.7k px). False positives are mostly short spurs along true vessels and
a few spots around the optic disc. Broken/discontinuous thin branches are the typical visual failure. Outside-FOV values in the saved prediction images are
meaningless (loss was FOV-masked) and are excluded from every metric.

## Overfitting / convergence

No overfitting was observed: validation loss tracks training loss (training uses augmentation, so training loss is not lower), and the validation-loss minimum
for E3 is at epoch 35 of 40. Best epochs of 35-40 suggest **under-training** rather than overfitting; longer training may help.

## Reproduce

```bash
pip install -r requirements.txt          # place DRIVE at data/DRIVE (training/{images,1st_manual,mask}, test/{images,mask})
python -m src.eda                        # dataset report + EDA figures
python -m src.run_experiments            # E1-E6 (~1.5 h on 2 CPU cores)
python -m src.evaluate select            # val-only model + threshold selection -> configs/final.yaml
python -m src.evaluate test              # ONE-TIME held-out test evaluation
python -m src.frangi_baseline && python -m src.make_tables
python -m src.inference --image path/to/fundus.tif [--fov mask.gif]   # -> results/inference/*.png
streamlit run app.py                     # full dashboard: overview, EDA, training curves, ablation, test results, error analysis,
                                         # Frangi baseline, image gallery, live demo (reads saved results; no dataset needed)
python -m src.make_gallery               # optional, needs the dataset: regenerates the gallery images used by the dashboard
```
Checkpoints are written to `models/checkpoints/` (E1-E6 `.weights.h5`; the final model is E3).

## Layout
`src/` code, `configs/` YAML, `splits/` fixed image IDs, `experiments/` results.csv + per-run histories/summaries + final test metrics,
`results/` figures and tables, `app.py` Streamlit demo, `REPORT.md` short research-style report.

## Limitations / future work
Tiny dataset and 4-image val/test splits; one seed; single dataset (domain shift untested); no pretrained encoder; CPU-limited training.
Future: pretrained encoders, longer training with multiple seeds / 5-fold CV, CHASE_DB1 external test, Attention U-Net, topology-aware losses (clDice), HD95.
