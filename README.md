# `ml_analysis` — Machine Learning Prediction Module

Publication-quality ML pipeline for pose-based ASD prediction and ADOS severity regression.

Built on top of `pose_analysis`, this module loads the raw **frame-level** pose metrics per subject,
computes rich per-subject summary statistics, and trains a battery of models under rigorous
nested cross-validation — suitable for reporting in a clinical AI publication.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Dependencies](#2-dependencies)
3. [Usage](#3-usage)
4. [Data and Feature Extraction](#4-data-and-feature-extraction)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Dimensionality Reduction](#6-dimensionality-reduction)
7. [Classification: ASD vs TD](#7-classification-asd-vs-td)
8. [Regression: ADOS-2 Total Score](#8-regression-ados-2-total-score)
9. [SHAP Explainability](#9-shap-explainability)
10. [Output File Reference](#10-output-file-reference)
11. [Module Reference](#11-module-reference)
12. [Design Decisions for Publishability](#12-design-decisions-for-publishability)

---

## 1. Overview

```
Input: 119 subjects × video pose records
         │
         ▼
  [features.py]     Frame-level CSVs → 1 278 features per subject
                    (58 metrics × 2 variants × 11 statistics + gender + age)
         │
         ▼
  [preprocessing]   1 280 → 717 features
                    (94 near-zero-variance removed, 469 correlated removed)
         │
         ├──► [dimreduce.py]      PCA + UMAP visualisations
         │
         ├──► [classification.py] ASD vs TD (79 vs 40)
         │                         6 models, nested 5×3 CV
         │
         ├──► [regression.py]    ADOS-2 total score prediction
         │                         6 models, nested 5×3 CV
         │
         └──► [explain.py]       SHAP global + local explanations
```

---

## 2. Dependencies

All packages are available in the shared venv:

```bash
module load GCCcore/13.3.0 Python/3.12.3 CUDA/12.8.0
source /home/shares/schaerm/schaer2/thibaut/humanlisbet/lisbet_venv/bin/activate
```

| Package | Version | Role |
|---------|---------|------|
| scikit-learn | 1.7.2 | Core ML, preprocessing, CV |
| pandas | 2.3.3 | Data loading and manipulation |
| numpy | 2.2.6 | Numerical operations |
| scipy | 1.16.2 | Skewness, kurtosis, Wilcoxon test |
| umap-learn | 0.5.9 | CPU UMAP embeddings |
| cuml | (GPU) | GPU-accelerated UMAP (via `--use-gpu`) |
| xgboost | — | XGBoost classifier / regressor |
| lightgbm | — | LightGBM classifier / regressor |
| shap | — | SHAP explainability |
| imbalanced-learn | — | SMOTE oversampling |
| joblib | 1.5.2 | Parallel feature loading + CV |

---

## 3. Usage

### Quick test (5 subjects, no GPU, skips slow steps)

```bash
cd humanLISBET-paper
python -m ml_analysis.run_ml \
    --csv          dataset/info/child_for_humanlisbet_paper_with_paths.csv \
    --pose-records dataset/pose_records \
    --output-dir   ml_analysis/results \
    --debug-n 5 \
    --skip-dimreduce \
    --skip-explain
```

### Full run (local, 4 CPUs)

```bash
python -m ml_analysis.run_ml \
    --csv          dataset/info/child_for_humanlisbet_paper_with_paths.csv \
    --pose-records dataset/pose_records \
    --output-dir   ml_analysis/results \
    --n-jobs 4
```

### Full run via SLURM (recommended — 16 CPUs, 64 GB, 1 GPU)

```bash
sbatch ml_analysis/run_ml.slurm
```

Logs are written to `ml_analysis/logs/run_ml_{SLURM_JOB_ID}.out`.

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | required | Path to clinical CSV |
| `--pose-records` | required | Path to `pose_records/` directory |
| `--output-dir` | `ml_analysis/results` | Root output directory |
| `--n-jobs` | `4` | Parallel workers (feature loading + outer CV) |
| `--use-gpu` | off | GPU support for XGB, LGBM, UMAP (requires CUDA) |
| `--corr-threshold` | `0.95` | Pearson \|r\| threshold for correlation filter |
| `--n-outer-folds` | `5` | Outer CV folds |
| `--n-inner-folds` | `3` | Inner CV folds (hyperparameter search) |
| `--n-iter` | `50` | `RandomizedSearchCV` iterations per inner search |
| `--umap-n-neighbors` | `15` | UMAP `n_neighbors` |
| `--umap-min-dist` | `0.1` | UMAP `min_dist` |
| `--skip-dimreduce` | — | Skip PCA + UMAP step |
| `--skip-classification` | — | Skip ASD vs TD step |
| `--skip-regression` | — | Skip ADOS regression step |
| `--skip-explain` | — | Skip SHAP step |
| `--debug-n` | — | Limit to first N subjects |
| `--random-state` | `42` | Global random seed |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## 4. Data and Feature Extraction

**Module:** `features.py`

### Why frame-level, not segment-level?

`pose_analysis` aggregates metrics at the segment level (weighted mean across segments).
This introduces a **segment-length bias**: summary statistics computed on a 30-second segment
are not directly comparable to those from a 5-minute segment, even after weighting.

`ml_analysis` avoids this by loading the two pre-concatenated **frame-level** CSV files
that `poseToRecord` already produces — one file per subject containing every valid frame
across all segments:

| File | Content |
|------|---------|
| `video_metrics_raw_2d.csv` | All frames, raw pixel-space metrics |
| `video_metrics_normalised.csv` | Same frames, trunk-height-normalised metrics |

Statistics are then computed **in a single pass on the full frame sequence**.
This is mathematically equivalent to computing on the original continuous signal.

### Statistics computed per metric × variant

For each of the 58 base metrics × 2 variants (raw / norm), 11 statistics are computed
on all valid (non-NaN) frames:

| Statistic | Formula | Interpretation |
|-----------|---------|----------------|
| `mean` | arithmetic mean | average level |
| `std` | standard deviation (ddof=1) | overall variability |
| `q25` | 25th percentile | lower quartile |
| `median` | 50th percentile | robust central tendency |
| `q75` | 75th percentile | upper quartile |
| `iqr` | q75 − q25 | robust spread |
| `min` | minimum | floor of signal |
| `max` | maximum | ceiling / peak |
| `cv` | std / mean | normalised variability (relative to mean) — **NaN for signed metrics** (see note) |
| `skewness` | Fisher skewness (bias=False) | asymmetry of distribution |
| `kurtosis` | excess kurtosis (bias=False) | tail heaviness vs Gaussian |

**Skewness and kurtosis** capture distributional shape that mean/std miss. For pose metrics
(which are often right-skewed due to occasional large values), these are particularly
informative about the proportion of time a child spends in "extreme" movement states.

**IQR-based outlier clipping for spread and higher-order statistics:** Normalised variants
divide every frame value by trunk height. When trunk height is momentarily mis-estimated
(a common MMPose artefact — e.g. person crouching or occluded), the division produces
spikes that are 10–100× the typical value, inflating `std`, `cv`, `skewness`, and
`kurtosis` for a single subject, driving PCA and tree splits with no biological meaning.

To prevent this, per-subject frame values are clipped to the Tukey IQR fences
`[Q1 − 3·IQR, Q3 + 3·IQR]` **before** computing spread and higher-order statistics.
This strategy is adaptive: for a metric with IQR=0.1, the upper fence sits at Q3 + 0.3,
which is far more aggressive toward artefacts than a fixed p99 (which could still be 50×
above Q3 if 60 artefact frames inflate the tail). At the same time, genuine behavioural
peaks (fast gestures = a handful of frames) sit well within the fence.

| Statistic | Computed on | Rationale |
|-----------|-------------|-----------|
| `mean` | original frames | Unbiased estimate of central tendency |
| `q10`, `q25`, `median`, `q75`, `q90`, `iqr` | original frames | Percentile-based, inherently robust |
| `std` | IQR-clipped frames | Removes inflation from artefact spikes |
| `cv` | IQR-clipped frames | cv = std_clipped / mean_clipped |
| `skewness` | IQR-clipped frames | 3rd-power moment; highly sensitive to extremes |
| `kurtosis` | IQR-clipped frames | 4th-power moment; even more sensitive |

**p10 / p90 replace min / max:** `min` of a non-negative metric (e.g. `acceleration_trunk`)
is ≈ 0 for almost every subject (there is always a near-constant-speed frame), giving a
near-zero cross-subject IQR — even a difference of 6×10⁻⁵ vs 10⁻⁶ becomes 30 IQRs after
robust scaling and dominates PC1. `max` is a single artefact frame.
`q10` ("typical low activity") and `q90` ("typical peak activity") are semantically
meaningful and robust to up to 10% artefact frames.

> **Before this fix:** 14 subjects had PC1 > 400 in the PCA embedding despite the earlier
> cv and kurtosis fixes; the worst (`acceleration_trunk__norm__min` = 6.2×10⁻⁵) was
> 30 IQRs from the median. After the fix the PCA embedding reflects genuine behavioural
> structure.

**CV and signed metrics:** `cv = std / mean` is only meaningful for non-negative signals
where the mean represents a genuine level. Several metrics are **signed** — they encode
direction and their mean is forced towards zero over a long recording as positive and
negative values cancel:

| Signed metric | Why the mean ≈ 0 |
|---------------|-----------------|
| `velocity_centroid_x/y` | Horizontal/vertical displacement: moves left AND right roughly equally |
| `velocity_trunk_x/y` | Same, restricted to trunk keypoints |
| `interpersonal_approach` | Rate of change of distance: approach and retreat cancel over a session |

For these metrics `cv = std / near_zero ≈ ∞`, which is meaningless and numerically
unstable. The implementation guards against this with a relative threshold:
`cv = NaN if |mean| ≤ 0.01 × std`. After imputation (median ≈ 0) and near-zero variance
filtering, these 18 cv features are automatically removed during preprocessing.

> **Why this matters for PCA:** Before this fix, subject `7892_Visite2_Recherche` had
> `clinician_velocity_centroid_x__cv ≈ 1,191,280` (mean ≈ 2×10⁻⁹, a floating-point
> near-zero). After `RobustScaler` this produced a value ~3,600× the IQR, causing PC1
> to point almost entirely at this one subject and explain 94.4% of variance — an artefact
> with no biological meaning.

**Special handling:**
- `segment_id` column — skipped entirely (not a metric)
- `kp_set_changed` (bool) — only `mean` computed (proportion of frames where the visible
  keypoint set changed); other statistics are set to NaN
- Metrics with fewer than 50 valid frames — all statistics set to NaN (insufficient data)
- **Signed metrics** — `cv` set to NaN when `|mean| ≤ 0.01 × std` (see above)
- **Spread / higher-order moments** — `std`, `cv`, `skewness`, `kurtosis` computed on IQR-clipped frames (Tukey fences k=3); see table above
- **`min` / `max` replaced by `q10` / `q90`** — robust to degenerate near-zero distributions and single-frame artefacts; see rationale above

### Feature count

- 58 base metrics × 2 variants × 11 statistics = **1 276 pose features**
- + `gender` (binary encoded: Male=1, Female=0) = **1 277**
- + `Ados_2_Age` (continuous, months) = **1 278 features**

Column naming convention (consistent with `pose_analysis`):

```
{base_metric}__{raw|norm}__{stat_type}
e.g.  child_speed_kp_left_wrist__raw__skewness
      interpersonal_distance_centroid__norm__cv
```

### Parallel loading

Subjects are loaded in parallel using `joblib.Parallel(n_jobs=N, backend='loky')`.
With 16 CPUs on 119 subjects (~67 MB per frame CSV), loading completes in ~3 minutes.

---

## 5. Preprocessing Pipeline

**Module:** `preprocessing.py`

All preprocessing steps are implemented as sklearn `BaseEstimator` / `TransformerMixin`
objects and assembled into a `Pipeline`. Critically, **the entire pipeline is fit only on
the training fold** inside the inner cross-validation loop — there is no data leakage from
test folds into feature selection.

### Pipeline steps (in order)

#### Step 1 — Missingness filter (`MissingnessFilter`)

Drop columns where more than **30%** of training subjects have NaN.
This removes metrics that are unreliable for a large fraction of the cohort
(e.g. a keypoint rarely detected by the pose estimator).

In practice at the dataset level: **0 columns dropped** (all metrics had <30% NaN).

#### Step 2 — Median imputation (`SimpleImputer(strategy='median')`)

Replace remaining NaN values with the column median computed on the training fold.
Median is preferred over mean for pose metrics because their distributions are
typically right-skewed (robust to the long right tails).

#### Step 3 — Near-zero variance filter (`NearZeroVarianceFilter`)

Drop features where more than **95%** of subjects share the same value.
These features carry essentially no discriminative information and can destabilise
some models (especially logistic regression and SVM).

At the dataset level: **94 features dropped**.
Typical examples: `min` statistics for metrics that are zero-bounded and near-zero
for almost all subjects (e.g. `child_speed_kp_nose__raw__min`).

#### Step 4 — Correlation filter (`CorrelationFilter`)

Drop one feature from each pair with |Pearson r| > **0.95** on the training fold.
The feature with **lower variance** is dropped (greedy algorithm, iterating from
highest to lowest variance so high-information features are always kept).

This is important because:
- Highly correlated features add no new information
- They inflate the feature count, increasing computation time
- Some models (logistic regression, linear SVM) can become numerically unstable

At the dataset level: **469 features dropped**, leaving **717 features**.

The `feature_selection_report.csv` records every dropped feature and the reason:

```csv
feature,reason,correlated_with
child_speed_centroid__raw__q75,high_correlation (|r|>0.95),child_speed_centroid__raw__max
...
```

#### Step 5 — Robust scaling (`RobustScaler`)

Scale each feature by subtracting the median and dividing by the IQR.
This is equivalent to `StandardScaler` but uses rank-based statistics instead of
mean/variance, making it robust to the outliers that remain in pose metric distributions.

### Why not StandardScaler?

Pose metrics (speeds, kinetic energies, distances) have **right-skewed distributions**
with occasional large values. StandardScaler uses mean and standard deviation which
are both influenced by these extremes. RobustScaler's use of median and IQR is largely
unaffected by outlier values.

---

## 6. Dimensionality Reduction

**Module:** `dimreduce.py`

Applied to the full dataset (preprocessed: imputed + RobustScaled) for visual exploration.
Not used as input to classification/regression — those use the raw feature matrix with
preprocessing inside CV folds.

### PCA

Standard linear dimensionality reduction. Fit with `sklearn.decomposition.PCA`.

**Figures:**

| Figure | Content |
|--------|---------|
| `pca_explained_variance.png` | Scree plot: individual + cumulative explained variance. Vertical red dashed line marks the PC where cumulative variance crosses 90%. |
| `pca_scatter_{diagnosis\|gender\|age\|ados}.png` | 2-D scatter of PC1 vs PC2, coloured by clinical variable. |
| `pca_biplot.png` | PC1 vs PC2 scatter (coloured by diagnosis) with top-10 feature loading vectors overlaid. Arrows point in the direction of highest positive loading; length indicates magnitude. |

In this dataset: **PC1 + PC2 together explain 97.7%** of the total variance.
This high fraction indicates that the feature space has strong low-dimensional structure
(likely one or two dominant axes of variation in movement behaviour).

### UMAP

Non-linear manifold learning. Preserves local neighbourhood structure and can reveal
cluster structure not visible in PCA.

Configuration: `n_neighbors=15`, `min_dist=0.1`, `n_components=2`, `random_state=42`.
With `--use-gpu`: uses `cuml.manifold.UMAP` (GPU-accelerated, much faster on large features).

**UMAP has no explained variance ratio** (it is non-linear). Two alternative quality metrics
are provided instead:

#### Trustworthiness curve

`sklearn.manifold.trustworthiness` measures how well local neighbourhoods in the original
high-dimensional space are preserved in the UMAP embedding.

- Score range: 0 (no preservation) to 1 (perfect preservation)
- A dashed line marks the 0.90 threshold (conventional "good" threshold)
- Computed for n_components = 1, 2, … , 8 to show how quality varies with embedding dimension

**How to read it:** If the curve rises steeply before n_components=2 and stays above 0.90,
the 2-D embedding faithfully represents the local structure of the data. A curve that only
reaches 0.90 at n_components=5 means 2-D is losing some neighbourhood information.

#### n_neighbors sensitivity grid

UMAP is sensitive to the `n_neighbors` parameter, which controls the balance between
local and global structure preservation:
- **Small n_neighbors (5)**: emphasises local clusters, may fragment global structure
- **Large n_neighbors (50)**: emphasises global topology, may merge nearby clusters

A 2×2 grid shows the embedding for `n_neighbors ∈ {5, 15, 30, 50}`, all coloured by
diagnosis. If the ASD/TD separation is consistent across n_neighbors values, the finding
is robust. If it only appears at one setting, it may be an artefact.

**Figures:**

| Figure | Content |
|--------|---------|
| `umap_scatter_{diagnosis\|gender\|age\|ados}.png` | 2-D UMAP coloured by clinical variable |
| `umap_trustworthiness.png` | Trustworthiness curve for n_components = 1..8 |
| `umap_n_neighbors_sensitivity.png` | 2×2 grid for n_neighbors ∈ {5, 15, 30, 50} |
| `combined_{diagnosis\|gender\|age\|ados}.png` | PCA and UMAP side-by-side for the same clinical variable |

---

## 7. Classification: ASD vs TD

**Module:** `classification.py`

**Target:** binary diagnosis (ASD = 1, TD = 0)
**Dataset:** 79 ASD, 40 TD (119 total; class ratio ≈ 2:1)

### Models

Six models span the complexity spectrum from fully interpretable to state-of-the-art:

| ID | Model | Rationale |
|----|-------|-----------|
| `dt` | `DecisionTreeClassifier(max_depth=4)` | Interpretable baseline; single decision tree |
| `lr` | `LogisticRegression(elasticnet)` | Linear baseline; feature weights interpretable |
| `svm` | `SVC(kernel='rbf', probability=True)` | Classical approach for small high-dimensional datasets |
| `rf` | `RandomForestClassifier(n_estimators=500)` | Ensemble, robust to noise |
| `xgb` | `XGBClassifier` | State-of-the-art gradient boosting |
| `lgbm` | `LGBMClassifier` | Fast gradient boosting (GPU-accelerated) |

All models use `class_weight='balanced'` (or equivalent `scale_pos_weight` for XGB)
to account for the ASD:TD imbalance.

### Cross-validation scheme

```
Outer: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  └─ Inner: StratifiedKFold(n_splits=3)
       └─ RandomizedSearchCV(n_iter=50, scoring='roc_auc', n_jobs=-1)
```

**Why nested CV?** A single train/test split or non-nested CV produces an optimistic
performance estimate because the test set is implicitly used during model selection.
Nested CV provides an **unbiased estimate of the generalisation performance** of the
model selection procedure:
- The **outer folds** measure true generalisation performance
- The **inner folds** are used for hyperparameter optimisation on the training data only
- The test fold in the outer loop is never seen during training or hyperparameter search

### Joint stratification

Standard `StratifiedKFold` only stratifies on one variable. With 119 subjects and a
2:1 ASD:TD imbalance, some folds could accidentally be all-TD or all-ASD.

To ensure each fold represents both diagnostic groups **and** the full range of ADOS
severity, a combined stratification label is created:

```python
ados_bins = pd.qcut(ADOS_2_TOTAL, q=3, labels=['low', 'med', 'high'])
strat_label = diagnosis + '_' + ados_bins
# e.g. "ASD_high", "ASD_med", "ASD_low", "TD_low", "TD_med", "TD_noados"
```

This creates up to 6 strata. Subjects missing ADOS get label `{diagnosis}_noados`.
The outer `StratifiedKFold` is applied to these combined labels.

**Why stratify by ADOS?** Models predicting diagnosis may otherwise be trained on
folds where ASD subjects happen to have systematically higher (or lower) ADOS severity,
leading to folds that are easier or harder than the true distribution.

### SMOTE (imbalanced data handling)

Inside each **inner training fold** only, SMOTE (Synthetic Minority Oversampling Technique)
generates synthetic ASD examples to balance the classes:

1. For each minority-class (ASD) sample, find its k=5 nearest neighbours in feature space
2. Create a synthetic sample by linear interpolation between the sample and one of its neighbours
3. Repeat until the class ratio is 1:1

SMOTE is applied **after** preprocessing inside the `imbalanced-learn Pipeline`, ensuring:
- Synthetic samples are only created from training data
- The test fold always contains real subjects only
- No information from the test fold contaminates the oversampling

### Evaluation metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| AUC-ROC | Area under the ROC curve | Primary metric; threshold-independent |
| Balanced accuracy | (Sensitivity + Specificity) / 2 | Equal weight to both classes |
| F1-macro | Unweighted mean of per-class F1 | Penalises poor performance on either class |
| F1-weighted | Class-frequency-weighted F1 | Reflects overall classification quality |
| Sensitivity (TPR) | TP / (TP + FN) | Rate of correctly identified ASD children |
| Specificity (TNR) | TN / (TN + FP) | Rate of correctly identified TD children |

**AUC-ROC is the primary metric** because it is insensitive to the class imbalance
and measures discriminative ability across all decision thresholds.

### Statistical model comparison

Pairwise Wilcoxon signed-rank tests are applied to the outer-fold AUC-ROC vectors
(one value per fold per model pair). Bonferroni correction is applied for the number
of pairs (15 pairs for 6 models).

The Wilcoxon test is used instead of a paired t-test because:
- The 5-fold AUC values are not normally distributed
- The test is paired (same outer fold splits across models)
- It is non-parametric (robust to distributional assumptions)

**Note:** With only 5 outer fold values, statistical power is limited. The p-values
should be interpreted with caution; large effect sizes (large AUC difference) are
more informative than p-values in this setting.

### Figures

| Figure | Content |
|--------|---------|
| `roc_curves.png` | All 6 models overlaid. Solid line = mean TPR across outer folds (interpolated to common FPR grid); shaded band = ±1 std. |
| `confusion_matrices.png` | Aggregated across all 5 outer folds for each model. True labels on y-axis, predicted on x-axis. Rows: TD (0), ASD (1). |
| `learning_curves.png` | Train vs validation AUC as a function of training set size for RF and XGB. Diagnoses overfitting (train >> val) or underfitting (both low). |

---

## 8. Regression: ADOS-2 Total Score

**Module:** `regression.py`

**Target:** `ADOS_2_TOTAL` (continuous, range typically 0–30 for Module 2/3)
**Dataset:** subjects with valid ADOS scores only (NaN subjects are excluded)

### Models

| ID | Model | Notes |
|----|-------|-------|
| `dt` | `DecisionTreeRegressor` | Interpretable baseline |
| `ridge` | `ElasticNet` | Linear with L1+L2 regularisation |
| `svr` | `SVR(kernel='rbf')` | Kernel-based regression |
| `rf` | `RandomForestRegressor(n_estimators=500)` | Ensemble |
| `xgb` | `XGBRegressor` | Gradient boosting |
| `lgbm` | `LGBMRegressor` | Fast gradient boosting |

### Cross-validation scheme

Same nested structure as classification, but stratification uses ADOS quintiles
(5 bins of equal frequency) to ensure each fold covers the full ADOS range:

```python
ados_bins = pd.qcut(ADOS_2_TOTAL, q=5, labels=False)   # 0, 1, 2, 3, 4
StratifiedKFold(n_splits=5) on ados_bins
```

This prevents folds where the training set only has high-ADOS subjects (which would
produce an optimistic RMSE on similarly high-ADOS test subjects).

### Evaluation metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| RMSE | √(mean squared error) | Primary; in ADOS score units |
| MAE | mean absolute error | More interpretable (mean error in points) |
| R² | coefficient of determination | Proportion of variance explained |
| Spearman ρ | rank correlation | Monotonic, robust to outliers |
| Pearson r | linear correlation | Magnitude of linear relationship |

**RMSE is the primary metric** for model comparison (inner CV scoring function:
`neg_root_mean_squared_error`).

**Spearman ρ is the most clinically relevant metric** because ADOS severity is
an ordinal construct and the relationship with pose metrics may be monotonic
but non-linear.

### Figures

| Figure | Content |
|--------|---------|
| `predicted_vs_actual_{model}.png` | Out-of-fold predicted vs true ADOS. Ideal fit line (y=x, dashed) + regression line (solid). RMSE, R², ρ, r annotated in title. |
| `residuals_{model}.png` | Left: histogram of residuals (predicted − true). Right: residuals vs predicted values. A good model should show residuals centred at 0 with no systematic pattern. |

---

## 9. SHAP Explainability

**Module:** `explain.py`

SHAP (SHapley Additive exPlanations) provides a theoretically grounded framework
for attributing model predictions to individual features, based on Shapley values
from cooperative game theory.

### What SHAP measures

For a given prediction, the SHAP value of feature *f* measures **how much feature *f*
changed the prediction compared to the expected prediction over all subjects**.

- Positive SHAP value → feature *f* pushed the prediction higher (towards ASD / higher ADOS)
- Negative SHAP value → feature *f* pushed the prediction lower (towards TD / lower ADOS)
- SHAP value ≈ 0 → feature *f* had little effect on this prediction

### Model selection for explainability

The best-performing model (by mean outer-fold AUC-ROC for classification, or lowest
mean outer-fold RMSE for regression) is retrained on **the full dataset** using the
most common best hyperparameters found during nested CV. SHAP is then computed on this
full-data model.

### Explainer selection

| Model type | Explainer | Notes |
|-----------|-----------|-------|
| RF, XGB, LGBM, DT | `shap.TreeExplainer` | Exact computation using tree structure; fast (polynomial in tree depth) |
| LR, ElasticNet | `shap.LinearExplainer` | Exact for linear models; uses feature covariance |
| SVM, SVR | `shap.KernelExplainer` | Model-agnostic approximation; slow (uses 50 k-means background samples) |

`TreeExplainer` is preferred when available because it computes exact SHAP values in
O(T × 2^D) time (T trees, D max depth) rather than the O(2^p) Shapley formula.

### Global explanations

**Beeswarm plot** (`shap_beeswarm_{task}_{model}.png`):
- One row per feature (top 20 by mean |SHAP|), sorted by importance (top = most important)
- Each dot = one subject; x-position = SHAP value for that feature and subject
- Colour = feature value for that subject (red = high, blue = low)
- **Reading the plot:** A feature where red dots cluster on the right (positive SHAP)
  means high values of that feature increase the ASD prediction. A feature where the
  dots are spread symmetrically around 0 contributes noise.

**Bar chart** (`shap_bar_{task}_{model}.png`):
- Mean |SHAP| for top 20 features — a clean, publication-ready importance ranking.

**Dependence plots** (`shap_dependence_*_{task}_{model}.png`):
- One plot per top-5 feature: SHAP value (y) vs feature value (x)
- Colour = the feature with the highest interaction (highest |corr| between its values
  and the SHAP values of the plotted feature)
- **Reading the plot:** The slope of the point cloud shows whether the relationship is
  linear (monotonic diagonal) or non-linear (e.g. threshold effect, U-shape).

### Local explanations (classification only)

**Waterfall plots** (`shap_waterfall_{TP|TN|FP|FN}_classification_{model}.png`):
- One plot for each type of outcome: True Positive, True Negative, False Positive, False Negative
- The representative subject is the one closest to the median predicted probability in that group
- Bars show the cumulative SHAP contributions, from smallest to largest |SHAP|
- **Reading the plot:** The bar chart shows which features were most responsible for
  this subject's prediction. For a False Positive (TD predicted as ASD), this reveals
  which pose features "tricked" the model.

---

## 10. Output File Reference

All results are saved under `--output-dir` (default: `ml_analysis/results/`).

```
ml_analysis/results/
├── feature_matrix.csv                 119 subjects × 1284 columns
│                                      (clinical + 1278 features + gender + age)
│
├── preprocessing/
│   └── feature_selection_report.csv  Every dropped feature + reason
│
├── dimreduce/
│   ├── pca_explained_variance.png
│   ├── pca_scatter_{diagnosis|gender|age|ados}.png
│   ├── pca_biplot.png
│   ├── umap_scatter_{diagnosis|gender|age|ados}.png
│   ├── umap_trustworthiness.png
│   ├── umap_n_neighbors_sensitivity.png
│   └── combined_{diagnosis|gender|age|ados}.png
│
├── classification/
│   ├── cv_results_all_models.csv      Per-fold per-model scores (5 folds × 6 models)
│   ├── model_comparison.csv           Mean ± std + pairwise Wilcoxon p (Bonferroni)
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── learning_curves.png
│
├── regression/
│   ├── cv_results_all_models.csv
│   ├── model_comparison.csv
│   ├── predicted_vs_actual_{dt|ridge|svr|rf|xgb|lgbm}.png
│   └── residuals_{dt|ridge|svr|rf|xgb|lgbm}.png
│
└── explain/
    ├── shap_beeswarm_{classification|regression}_{model}.png
    ├── shap_bar_{classification|regression}_{model}.png
    ├── shap_dependence_{rank}_{feature}_{task}_{model}.png
    └── shap_waterfall_{TP|TN|FP|FN}_classification_{model}.png
```

### `feature_matrix.csv`

One row per subject. Key columns:

| Column | Type | Description |
|--------|------|-------------|
| `uuid` | str | Session identifier |
| `diagnosis` | ASD / TD | Target for classification |
| `ADOS_2_TOTAL` | float | Target for regression |
| `Ados_2_Age` | float | Age in months |
| `gender` | Male / Female | |
| `{metric}__{raw\|norm}__{stat}` | float | 1278 pose features |

### `cv_results_all_models.csv`

One row per (fold × model) combination. Key columns for classification:

| Column | Description |
|--------|-------------|
| `fold` | Outer fold index (0–4) |
| `model` | `dt`, `lr`, `svm`, `rf`, `xgb`, `lgbm` |
| `auc_roc` | AUC-ROC for this fold |
| `balanced_acc` | Balanced accuracy |
| `f1_macro` | Macro-averaged F1 |
| `sensitivity` | True positive rate (ASD) |
| `specificity` | True negative rate (TD) |
| `best_params` | Best hyperparameters found by inner CV |

For regression, `auc_roc` / `sensitivity` / `specificity` are replaced by
`rmse`, `mae`, `r2`, `spearman_r`, `pearson_r`.

### `model_comparison.csv`

One row per model. Key columns (classification example):

| Column | Description |
|--------|-------------|
| `model` | Model ID |
| `auc_roc_mean` | Mean AUC-ROC across outer folds |
| `auc_roc_std` | Std across outer folds |
| `p_vs_{other_model}_bonf` | Bonferroni-corrected Wilcoxon p-value vs each other model |

### `feature_selection_report.csv`

| Column | Description |
|--------|-------------|
| `feature` | Dropped feature name |
| `reason` | `high_missingness (frac=X.XX)`, `near_zero_variance`, `high_correlation (\|r\|>0.95)` |
| `correlated_with` | (correlation drops only) The feature that was kept instead |

---

## 11. Module Reference

| Module | Key functions |
|--------|--------------|
| `features.py` | `load_feature_matrix(csv_path, pose_records_dir, n_jobs, debug_n)` → `pd.DataFrame` |
| | `get_feature_columns(df, variant, stat_type)` → `list[str]` |
| `preprocessing.py` | `build_preprocessing_pipeline(corr_threshold, near_zero_frac, max_missing_frac)` → `sklearn.Pipeline` |
| | `generate_feature_selection_report(X, output_dir, ...)` → `pd.DataFrame` |
| | `prepare_feature_matrix(df, feature_cols, include_gender, include_age)` → `(pd.DataFrame, list[str])` |
| `dimreduce.py` | `run_dimensionality_reduction(X_scaled, df_meta, feature_names, output_dir, ...)` → `dict` |
| `classification.py` | `run_classification(X, y, df_meta, feature_names, output_dir, ...)` → `pd.DataFrame` |
| | `build_strat_label(df)` → combined diagnosis × ADOS stratification labels |
| `regression.py` | `run_regression(X, y, df_meta, feature_names, output_dir, ...)` → `pd.DataFrame` |
| `explain.py` | `run_explainability(X, y_clf, y_reg, df_meta, feature_names, df_cv_clf, df_cv_reg, output_dir, ...)` |

---

## 12. Design Decisions for Publishability

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Frame-level features** | Load `video_metrics_raw_2d.csv` directly | Avoids segment-length bias from aggregating segment-level summaries |
| **11 statistics** | mean, std, quartiles, IQR, min, max, CV, skewness, kurtosis | Captures distributional shape (skewness/kurtosis) that segment-level means miss |
| **Nested CV** | 5 outer × 3 inner folds | Outer folds = unbiased performance; inner folds = hyperparameter selection. Required for honest generalisation estimate. |
| **Feature selection inside CV** | All preprocessing steps fit on training fold only | Prevents test-set leakage into feature selection (a common methodological error) |
| **Joint CV stratification** | Combined diagnosis + ADOS tertile label | Ensures each fold has both diagnostic groups and comparable ADOS distributions |
| **SMOTE** | Inside inner training fold only | Balances classes without exposing synthetic samples to the test fold |
| **Class imbalance** | `class_weight='balanced'` + SMOTE | Dual approach: loss weighting + resampling |
| **Evaluation** | AUC-ROC ± std across outer folds | Threshold-independent, informative for imbalanced datasets |
| **Model comparison** | Wilcoxon signed-rank (paired, non-parametric) | Accounts for fold correlation; does not assume normal distribution of AUC values |
| **Bonferroni correction** | Applied to all pairwise model comparisons | Conservative correction for 15 pairs (6 models) |
| **SHAP explainer** | TreeExplainer for tree models | Exact computation; not an approximation |
| **Reproducibility** | `random_state=42` everywhere + `feature_matrix.csv` saved | Any downstream analysis can start from the saved feature matrix |
| **Robust scaling** | `RobustScaler` (IQR-based) | More appropriate than `StandardScaler` for right-skewed pose metric distributions |
