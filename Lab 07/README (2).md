# Lab 07 — Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

**Author:** Fahad Bin Shafi (FA23-BAI-033)
**Course:** Computer Vision — COMSATS University Islamabad
**Notebook:** `Lab07_CV_.ipynb` (Google Colab)

## Overview

This lab tests whether a **Genetic Algorithm (GA)** can improve a classic HOG + SVM pipeline for chest X-ray classification (**Normal vs. Pneumonia**) by selecting the most useful HOG features.

Two pipelines are compared on identical data splits and preprocessing:

| | Baseline | Proposed |
|---|---|---|
| Features | HOG (all 8100) | HOG → GA-selected subset |
| Classifier | RBF-SVM | RBF-SVM |

## Dataset

**PneumoniaMNIST** from Kaggle (`rijulshr/pneumoniamnist`), loaded automatically via `kagglehub`.

| Split | Images | Normal | Pneumonia |
|---|---|---|---|
| Train | 3882 | 388 | 3494 |
| Validation | 524 | 135 | 389 |
| Test | 624 | 234 | 390 |

Native resolution is 28×28; images are upsampled to 128×128. The dataset is imbalanced, so macro-F1 and balanced class weights are used throughout.

## Pipeline

1. **Preprocessing:** resize to 128×128, scale to [0, 1].
2. **HOG extraction:** 9 orientations, 8×8 pixels per cell, 2×2 cells per block, L2-Hys normalization → **8100 features** per image.
3. **Baseline:** RBF-SVM on all features. `C` and `gamma` are tuned on the validation set.
4. **GA feature selection:**
   - Chromosome: binary mask of length 8100 (1 = keep feature)
   - Fitness: `validation macro-F1 − λ · (n_selected / n_total)` with λ = 0.05
   - Selection: tournament (k = 3)
   - Crossover: uniform (p = 0.9)
   - Mutation: bit-flip
   - Elitism: 2 individuals
   - Initial population includes the full feature set, so the GA cannot start worse than the baseline
   - Fitness SVMs train on a stratified 2000-image subsample for speed
5. **Final model:** best GA setting is chosen on validation fitness, the SVM is re-tuned on the selected features, then trained on the full training set.
6. **Evaluation:** the test set is used once, after model selection.

The RBF kernel matrix is computed with PyTorch (GPU if available) and passed to `sklearn.svm.SVC` as a precomputed kernel. `gamma = gamma_mult / (n_features × variance)`, which keeps tuning fair across different feature counts.

## GA Settings Compared

| Setting | Population | Generations | Mutation prob. | Features kept | Val macro-F1 (%) | GA time (s) |
|---|---|---|---|---|---|---|
| GA-A | 20 | 15 | 0.002 | 2129 | 96.13 | 170.7 |
| **GA-B (selected)** | 30 | 15 | 0.002 | 2312 | 96.40 | 273.3 |
| GA-C | 20 | 15 | 0.010 | 2246 | 95.86 | 179.3 |

## Results

### Test set (HOG + SVM vs. GA-HOG + SVM)

| Metric | Baseline | GA-HOG + SVM | Change |
|---|---|---|---|
| Features | 8100 | 2312 | −71.5% |
| Macro-F1 (%) | 77.36 | 80.63 | +3.27 pp |
| Accuracy (%) | 81.57 | 83.81 | +2.24 pp |
| Balanced accuracy | — | — | +3.08 pp |
| MCC | — | — | +0.044 |

**Statistical check:** paired bootstrap (2000 resamples) gives a macro-F1 difference of **+3.27 pp, 95% CI [+1.44, +5.35]**. The interval excludes 0, so the improvement is statistically meaningful.

### Computational cost

| Stage (s) | HOG + SVM | GA-HOG + SVM |
|---|---|---|
| HOG extraction | 58.50 | 58.50 |
| GA feature selection | 0.00 | 273.25 |
| Hyperparameter tuning | 18.48 | 8.00 |
| Final SVM training | 4.85 | 1.75 |
| Test inference | 0.86 | 0.22 |
| **Total** | **82.69** | **341.72** |
| Support vectors | 947 | 889 |

## Key Takeaways

- The GA removed about **72% of HOG features** and still **improved** test macro-F1 and accuracy.
- The final model trains and predicts faster (~4× faster inference), but building it costs more: the GA search took ~273 s (450 SVM evaluations).
- Validation scores are much higher than test scores (about 96% vs. 80% macro-F1) for both pipelines, so the validation set is not fully representative of the test set.

## Limitations

- **Computational cost:** every GA fitness evaluation trains an SVM.
- **Validation overfitting:** the same validation set drives both the GA and model selection.
- **Single seed:** results depend on seed, population size, mutation rate, λ and generation count.
- **Low native resolution:** 28×28 images upsampled to 128×128, so HOG captures interpolated gradients. HOG typically lags CNNs on this task.
- The notebook ran on **CPU only** in the saved run.

## How to Run

1. Open `Lab07_CV_.ipynb` in Google Colab (a GPU runtime is optional but faster).
2. Run all cells top to bottom. The first cell installs `kagglehub`, `scikit-image`, `opencv-python-headless` and `seaborn`.
3. Kaggle data downloads automatically through `kagglehub`.

**Main dependencies:** numpy, pandas, matplotlib, seaborn, opencv, torch, scikit-image, scikit-learn, joblib, kagglehub.

## Outputs

Saved to `/content/lab07_outputs/` (and zipped for download in the last cell):

- `table1_config.csv`, `table2_main_comparison.csv`, `table3_computational_cost.csv`, `table_ga_settings.csv`
- `ga_history_selected.csv`, `selected_feature_mask.npy`
- `samples.png`, `ga_fitness.png`, `features_and_metrics.png`, `confusion_matrices.png`
