# ML Analysis — Pipeline Documentation

This document describes each of the four processing pipelines in `ml_analysis` independently, including what data they receive, how preprocessing is applied, and what they produce.

---

## Shared preprocessing building block

All four pipelines reuse the same five-step preprocessing sequence, built by `preprocessing.build_preprocessing_pipeline()`.

| Step | Class | Action |
|------|-------|--------|
| 1 | `MissingnessFilter(max_missing_frac=0.30)` | Drops columns where > 30 % of values are NaN. Stores `keep_mask_`. |
| 2 | `SimpleImputer(strategy='median')` | Fills remaining NaN with the column median. |
| 3 | `NearZeroVarianceFilter(frac=0.95)` | Drops columns that are identical in ≥ 95 % of rows. Stores `keep_mask_`. |
| 4 | `CorrelationFilter(threshold=0.95)` | For every correlated pair (\|Pearson r\| ≥ threshold), drops the member with lower variance. Stores `keep_mask_`. |
| 5 | `RobustScaler()` | Centres each feature on its median and scales by its IQR (robust to outliers). |

Feature names after filtering are recovered via `preprocessing.get_pipeline_feature_names(pipeline, input_names)`, which propagates the real metric names through each `keep_mask_` in order.

**Critical rule**: Every pipeline fits the preprocessing steps **only on the training partition** (or the full dataset for visualisation). Test partitions are only transformed, never used to fit any step. This prevents data leakage.

---

## Pipeline 1 — Dimensionality Reduction (PCA + UMAP)

**Purpose:** Visual exploration of the full dataset. No cross-validation; results are for human interpretation only and must not be used to report predictive performance.

### Input
- `X_raw` — full feature matrix (all subjects × ~1 278 raw features, before any preprocessing)
- `df_meta` — clinical metadata (uuid, diagnosis, gender, age, ADOS)
- `feature_names` — list of column names corresponding to `X_raw`

### Preprocessing
The full five-step preprocessing pipeline is fit **once on the entire dataset** (no train/test split):

```
MissingnessFilter → SimpleImputer → NearZeroVarianceFilter → CorrelationFilter → RobustScaler
```

`X_scaled` (shape: n_subjects × n_kept_features) is the input to both PCA and UMAP.
`feature_names_filtered` are the surviving feature names, used for biplot labels.

### PCA
- `sklearn.decomposition.PCA(n_components=min(n_subjects, n_features))`
- Fit on `X_scaled` (full dataset).
- Outputs:
  - Explained variance curve (cumulative), 90 % threshold marked
  - 2D scatter: PC1 vs PC2, coloured by diagnosis / gender / age / ADOS
  - Biplot: PC1 vs PC2 with top-10 loading vectors overlaid
- **Saved data** (`dimreduce/`):
  - `pca_explained_variance.csv` — per-component explained variance ratio and cumulative
  - `pca_embedding.csv` — per-subject PC1..PC20 coordinates + clinical prefix
  - `pca_loadings_top50.csv` — top-50 features ranked by loading magnitude in PC1–PC2 plane

### UMAP
- `umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)`
- Fit on `X_scaled` (full dataset).
- Two quality diagnostics (substitutes for explained-variance, which is undefined for non-linear methods):
  - **Trustworthiness curve**: `sklearn.manifold.trustworthiness` evaluated for n_components 1..8 — measures how faithfully local high-dimensional neighbourhoods are preserved in the embedding. Dashed line at 0.90.
  - **n_neighbors sensitivity grid**: UMAP re-run with n_neighbors ∈ {5, 15, 30, 50}; 2 × 2 panel shows structural stability across parameter choices.
- **Saved data** (`dimreduce/`):
  - `umap_embedding.csv` — per-subject UMAP1, UMAP2 coordinates + clinical prefix

### No leakage risk
Because there is no train/test split and no performance metric is reported, fitting preprocessing on the full dataset is appropriate here.

---

## Pipeline 2 — Classification (ASD vs TD, Nested CV)

**Purpose:** Estimate out-of-sample AUC-ROC and other classification metrics for predicting diagnosis. All reported numbers come from held-out test folds to give an honest generalisation estimate.

### Input
- `X_raw` — full feature matrix (all subjects)
- `y_clf` — binary labels (ASD=1, TD=0)
- `df_meta` — for stratification label construction and linking predictions to UUIDs

### Stratification
To keep every fold balanced on both diagnosis and ADOS severity simultaneously:

```python
ados_tertile = pd.qcut(ADOS_2_TOTAL, q=3, labels=['low','med','high'])
strat_label = diagnosis + '_' + ados_tertile   # e.g. "ASD_high", "TD_low"
# Subjects with missing ADOS → "ASD_noados" / "TD_noados"
```

### Cross-validation scheme
```
Outer: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  ← on strat_label
  └─ Inner: StratifiedKFold(n_splits=3)                             ← on strat_label
       └─ RandomizedSearchCV(n_iter=50, scoring='roc_auc', n_jobs=-1)
```

### Per-fold pipeline (fit only on outer training fold)
```
MissingnessFilter → SimpleImputer → NearZeroVarianceFilter → CorrelationFilter
  → RobustScaler → SMOTE → Model
```

Implemented as an `imbalanced_learn.pipeline.Pipeline` so that SMOTE oversampling is applied only to the training fold and never to validation or test data.

Models searched:

| ID | Model |
|----|-------|
| `dt` | `DecisionTreeClassifier(max_depth=4)` |
| `lr` | `LogisticRegression(penalty='elasticnet', solver='saga')` |
| `svm` | `SVC(kernel='rbf', probability=True)` |
| `rf` | `RandomForestClassifier(n_estimators=500, class_weight='balanced')` |
| `xgb` | `XGBClassifier(tree_method='hist'/'cuda')` |
| `lgbm` | `LGBMClassifier(device='cpu'/'gpu')` |

### Per-fold metrics
AUC-ROC (primary), balanced accuracy, F1-macro, F1-weighted, sensitivity (TPR), specificity (TNR).

### Statistical model comparison
Wilcoxon signed-rank test on fold AUC-ROC vectors for all model pairs; Bonferroni correction applied.

### Outputs (`classification/`)
- `cv_results_all_models.csv` — per-fold, per-model scalar metrics
- `model_comparison.csv` — mean ± std + pairwise Wilcoxon p-values
- `predictions_per_subject.csv` — out-of-fold predicted probability and label per subject × model (UUIDs linked via `test_indices`)
- `roc_curve_data.csv` — FPR/TPR/threshold per fold × model (for custom ROC plots)
- `roc_curves.png`, `confusion_matrices.png`, `learning_curves.png`

---

## Pipeline 3 — Regression (ADOS-2 Total Score, Nested CV)

**Purpose:** Estimate out-of-sample RMSE, MAE, R², Spearman ρ, and Pearson r for predicting the continuous ADOS-2 total score. Only subjects with a valid (non-NaN) ADOS score are included.

### Input
- `X_reg` — feature matrix restricted to subjects with valid ADOS (subset of `X_raw`)
- `y_reg` — ADOS-2 total scores for those subjects
- `df_meta` (subset) — for stratification and UUID linking

### Stratification
ADOS-quintile stratification ensures each fold covers the full score range:

```python
ados_bins = pd.qcut(ADOS_2_TOTAL, q=5, labels=False)   # quintiles 0–4
# StratifiedKFold on quintile labels
```

### Cross-validation scheme
```
Outer: StratifiedKFold(n_splits=5) on ados_bins
  └─ Inner: StratifiedKFold(n_splits=3) on ados_bins
       └─ RandomizedSearchCV(n_iter=50, scoring='neg_root_mean_squared_error')
```

### Per-fold pipeline (fit only on outer training fold)
```
MissingnessFilter → SimpleImputer → NearZeroVarianceFilter → CorrelationFilter
  → RobustScaler → Model
```

Implemented as a standard `sklearn.pipeline.Pipeline` (no SMOTE — continuous target).

Models searched:

| ID | Model |
|----|-------|
| `dt` | `DecisionTreeRegressor(max_depth=4)` |
| `ridge` | `Ridge` (alpha grid) |
| `elasticnet` | `ElasticNet` (alpha, l1_ratio grid) |
| `svr` | `SVR(kernel='rbf')` |
| `rf` | `RandomForestRegressor(n_estimators=500)` |
| `xgb` | `XGBRegressor(tree_method='hist'/'cuda')` |
| `lgbm` | `LGBMRegressor(device='cpu'/'gpu')` |

### Per-fold metrics
RMSE (primary), MAE, R², Spearman ρ, Pearson r.

### Outputs (`regression/`)
- `cv_results_all_models.csv` — per-fold, per-model scalar metrics
- `model_comparison.csv` — mean ± std + Wilcoxon pairwise tests
- `predictions_per_subject.csv` — out-of-fold predicted ADOS, true ADOS, residual, diagnosis per subject × model
- `predicted_vs_actual_{model}.png`, `residuals_{model}.png`

---

## Pipeline 4 — SHAP Explainability

**Purpose:** Identify which pose features drive predictions in the best-performing model. The model is retrained on the **full dataset** (no held-out test set) to maximise the signal available for explanation. Performance metrics must not be read from this fit; they come from Pipelines 2 and 3.

### Input
- `X_raw` — full feature matrix (for classification) or ADOS-valid subset (for regression)
- `y_clf` / `y_reg` — target labels
- `df_cv_clf` / `df_cv_reg` — CV results DataFrames from Pipelines 2 and 3 (used to identify best model)
- `feature_names` — original raw feature column names

### Best model selection
Best model = model with highest mean AUC-ROC across outer folds (classification) or lowest mean RMSE (regression). Its hyperparameters are fixed to those most frequently chosen across outer folds.

### Retraining (`_retrain_best`)
```
MissingnessFilter → SimpleImputer → NearZeroVarianceFilter → CorrelationFilter
  → RobustScaler → Model
```

Fit on the full dataset. `get_pipeline_feature_names(preproc, feature_names)` traces `keep_mask_` through the filtering steps to recover the real metric names of the surviving features — these names appear on all SHAP plots and CSV outputs.

### SHAP computation
| Model type | SHAP explainer |
|-----------|----------------|
| RF, XGB, LGBM, DT | `shap.TreeExplainer` (exact, model-native) |
| LR, Ridge, ElasticNet | `shap.LinearExplainer` |
| SVM, SVR | `shap.KernelExplainer(background=KMeans(50 centroids))` |

### Outputs (`explain/`)
- `shap_values_{task}_{model}.csv` — SHAP value matrix (subjects × features) + clinical prefix columns
- `shap_feature_values_{task}_{model}.csv` — preprocessed feature values that produced the SHAP values (same shape)
- `shap_importance_{task}_{model}.csv` — features ranked by mean \|SHAP\| (global importance)
- `shap_beeswarm_{task}_{model}.png` — beeswarm (top 20 by mean \|SHAP\|)
- `shap_bar_{task}_{model}.png` — bar chart of mean \|SHAP\|
- `shap_dependence_{feature}_{task}_{model}.png` — top-5 dependence plots
- `shap_waterfall_{subject}_{task}_{model}.png` — local waterfall for 4 representative subjects (TP/TN/FP/FN)

---

## Summary comparison

| Property | Dimreduce | Classification | Regression | Explainability |
|----------|-----------|---------------|------------|----------------|
| Purpose | Visualisation | Performance estimation | Performance estimation | Feature importance |
| Subjects | All | All | ADOS-valid only | All / ADOS-valid |
| CV | None | Nested 5-outer × 3-inner | Nested 5-outer × 3-inner | None (full retrain) |
| Preprocessing fit on | Full dataset | Each outer train fold | Each outer train fold | Full dataset |
| SMOTE | No | Yes (inside train fold) | No | No (if RF/XGB) |
| Stratification | N/A | diagnosis + ADOS tertile | ADOS quintile | N/A |
| Primary metric | N/A | AUC-ROC | RMSE | Mean \|SHAP\| |
| Leakage risk | None (no metric) | Controlled (nested CV) | Controlled (nested CV) | None (explanation only) |
