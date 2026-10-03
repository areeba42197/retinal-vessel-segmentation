# Retinal Blood Vessel Segmentation Using Deep Learning: A Reproducible U-Net Study on DRIVE

**Abstract.** We study automated retinal vessel segmentation on the DRIVE dataset with a leakage-free, image-level split of its 20 labelled images
(12 train / 4 validation / 4 held-out test), because the supplied official test folder has no vessel annotations. A controlled ablation over loss
(BCE vs BCE+Dice), augmentation, CLAHE preprocessing and encoder (U-Net vs randomly initialised EfficientNet-B0 U-Net) was run with a single seed. The model chosen
on validation data (U-Net, BCE+Dice, augmentation, threshold 0.45) reached **test Dice 0.8153, IoU 0.6881, sensitivity 0.7926, precision 0.8393,
ROC-AUC 0.9749** on 4 held-out images, versus Dice 0.7213 for a tuned Frangi filter. Errors concentrate on thin vessels. Results are from very small
evaluation sets and are not a claim of state-of-the-art performance or clinical suitability.

## 1-3. Introduction and problem
Vessel morphology in fundus images underlies many retinal analyses; manual annotation is slow. The task here is binary pixel-wise segmentation of vessels
(inside the circular field of view, FOV). Vessels are ~12.5% +- 1.9% of FOV pixels, so accuracy is uninformative and Dice is the primary metric. This project does not diagnose disease.

## 4-5. Dataset and EDA
DRIVE: 20 labelled training images (IDs 21-40, 584x565 RGB uint8), vessel masks (GIF, values {0,255}) and FOV masks; 20 test images with FOV masks only. Integrity check:
0 missing, 0 duplicate, 0 unreadable files (`results/tables/dataset_report.txt`). Image green-channel mean varies from 58 to 125 across images and contrast (p95-p5)
from 26 to 150, i.e. strong illumination/contrast variation (`results/eda/`). Hard cases: darkest id 30, lowest contrast id 26, fewest vessels id 31.

## 6-7. Preprocessing and augmentation
Basic (RGB/255), green channel and CLAHE (LAB L-channel, clip 2, 8x8 tiles) were implemented; basic and CLAHE were compared experimentally (green: implemented, not trained). Pixels outside the FOV are
zeroed and excluded from the loss and all metrics. Augmentation: flips, +-15 deg rotation, +-10% zoom, brightness/contrast/gamma +-0.10, applied identically to image and mask
(nearest-neighbour for labels; verified visually in `results/eda/augmentation_check.png`).

## 8-12. Methods
Splits are made on whole images before any patch extraction. Baseline: 4-level U-Net (filters 16-32-64-128, bottleneck 256, BatchNorm, 1.95 M params). Second model:
U-Net decoder on an EfficientNet-B0 encoder (6.79 M params, random init because ImageNet weights were unavailable). Losses: FOV-masked BCE; BCE+Dice (0.5/0.5).
AdamW (lr 1e-3, weight decay 1e-4), batch 16 patches of 128x128, 30 steps x up to 40 epochs, ReduceLROnPlateau on validation loss (factor 0.5, patience 5), early stopping (patience 10 after a 10-epoch warm-up) and
best-checkpoint selection on validation Dice. Inference is fully convolutional on the whole padded image.

## 13-15. Metrics, setup, ablation
Dice, IoU, precision, recall, specificity, F1, MCC, ROC-AUC, PR-AUC, clDice; pixel accuracy only as a supplementary number. Validation Dice is the best over thresholds {0.3,0.4,0.5,0.6,0.7} (identical rule for all runs).

| Exp | Setup | Val Dice | Val IoU | Best epoch |
|--|--|--:|--:|--:|
| E1 | U-Net, BCE, no aug | 0.7935 | 0.6577 | 25 |
| E2 | U-Net, BCE+Dice, no aug | 0.8033 | 0.6712 | 31 |
| E3 | U-Net, BCE+Dice, aug | 0.8133 | 0.6853 | 35 |
| E4 | E3 + CLAHE | 0.8138 | 0.6860 | 37 |
| E5 | EffNet-B0 U-Net (random init), BCE+Dice, aug | 0.7993 | 0.6657 | 40 |
| E6 | E5 + CLAHE (chosen by E3-vs-E4 rule: higher val Dice) | 0.8062 | 0.6753 | 40 |

E3 and E4 are tied within 0.002, so E3 (simpler) was selected. Threshold sweep on validation: Dice 0.8041-0.8137 over 0.30-0.70, maximum at 0.45 (locked).
Informational (validation only): flip-TTA gave val Dice 0.8171 vs 0.8137; it was not applied to the test result.

## 16. Results (held-out test, evaluated once)
Dice 0.8153, IoU 0.6881, precision 0.8393, recall 0.7926, specificity 0.9771, MCC 0.7888, ROC-AUC 0.9749, PR-AUC 0.9000, clDice 0.8244, accuracy 0.9529 (supplementary).
Per-image Dice: id 21 0.835, 22 0.801, 36 0.818, 38 0.809. Frangi baseline (threshold fitted on train+val): test Dice 0.7213, IoU 0.5641, ROC-AUC 0.9054.

## 17. Error analysis
Thick-vessel skeleton recall 0.95-0.99 vs thin-vessel 0.49-0.70 (thin = local half-width <= 1.5 px). Image 22 and 36 lose the most thin branches (FN 7.3k / 8.7k px); image 38 has the most FP
(6.0k px, mostly spurs and the optic-disc region). See `results/error_maps/fig5_6_test_predictions_error_maps.png` (white TP, red FP, blue FN, black TN).

## 18-21. Cross-dataset, limitations, conclusion, future work
Cross-dataset evaluation (CHASE_DB1) was **not run** (data not supplied). Limitations: 4-image val/test sets, single seed (differences of ~0.01 are within plausible noise), under-trained models (best epochs near the cap),
random-init EfficientNet, CPU-limited patch size, no cross-validation, not a medical device. Conclusion: a small U-Net trained with BCE+Dice and augmentation gives a solid, honest baseline (test Dice 0.815) that
clearly beats a classical Frangi filter; thin-vessel recall is the main weakness. Future work: pretrained encoders, longer multi-seed training / 5-fold CV, CHASE_DB1 evaluation, Attention U-Net, topology-aware losses, HD95.
