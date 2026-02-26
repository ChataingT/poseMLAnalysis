"""
Preprocessing pipeline for the ML pipeline.

Steps applied inside sklearn Pipeline (all fit on train fold only, no leakage):
  1. Drop high-missingness features (>30% NaN)
  2. Median imputation (SimpleImputer)
  3. Near-zero variance filter (VarianceThreshold, drops if >95% constant)
  4. Correlation filter (custom: |Pearson r| > threshold → drop lower-variance partner)
  5. Robust scaling (RobustScaler)

Gender encoding (Male=1, Female=0) is applied once up-front, before any pipeline.

Outputs (call generate_feature_selection_report separately, outside CV):
  - feature_selection_report.csv: each dropped feature + reason
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

MAX_MISSING_FRAC = 0.30   # drop columns with > 30% NaN
CORR_THRESHOLD = 0.95     # |Pearson r| above which one feature is dropped
VAR_THRESHOLD = 0.0       # VarianceThreshold default (drops constant features)
NEAR_ZERO_FRAC = 0.95     # drop if >95% of values are the same (near-zero variance)


# ─────────────────────────────────────────────────────────────
# Gender encoding
# ─────────────────────────────────────────────────────────────

def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode gender column in-place: Male → 1, Female → 0.
    Other values → NaN (handled by downstream imputer).
    Returns a copy of the DataFrame.
    """
    df = df.copy()
    if "gender" in df.columns:
        mapping = {"Male": 1.0, "Female": 0.0, "male": 1.0, "female": 0.0}
        df["gender"] = df["gender"].map(mapping)
    return df


# ─────────────────────────────────────────────────────────────
# Custom sklearn transformers
# ─────────────────────────────────────────────────────────────

class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Drop columns with more than max_missing_frac NaN on the training set."""

    def __init__(self, max_missing_frac: float = MAX_MISSING_FRAC):
        self.max_missing_frac = max_missing_frac

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        missing_frac = np.mean(np.isnan(X), axis=0)
        self.keep_mask_ = missing_frac <= self.max_missing_frac
        self.n_dropped_ = int((~self.keep_mask_).sum())
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.keep_mask_))])[self.keep_mask_]
        return np.asarray(input_features)[self.keep_mask_]


class NearZeroVarianceFilter(BaseEstimator, TransformerMixin):
    """
    Drop features where more than `frac` of samples share the same value.
    This is more robust than VarianceThreshold for non-Gaussian features.
    """

    def __init__(self, frac: float = NEAR_ZERO_FRAC):
        self.frac = frac

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        keep = []
        for j in range(X.shape[1]):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                keep.append(False)
                continue
            # Most common value fraction
            vals, counts = np.unique(valid, return_counts=True)
            max_frac = counts.max() / len(valid)
            keep.append(max_frac <= self.frac)
        self.keep_mask_ = np.array(keep, dtype=bool)
        self.n_dropped_ = int((~self.keep_mask_).sum())
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.keep_mask_))])[self.keep_mask_]
        return np.asarray(input_features)[self.keep_mask_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Drop one feature from each highly correlated pair (|Pearson r| > threshold).

    Greedy algorithm (order by descending variance on train set):
      for each feature f (high→low variance):
          if f is already marked to drop: skip
          for each remaining feature g (lower variance than f):
              if |corr(f, g)| > threshold: mark g to drop

    The kept feature is always the one with higher variance.
    `dropped_pairs_` records (kept_feature, dropped_feature) for the report.
    """

    def __init__(self, threshold: float = CORR_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_feat = X.shape[1]

        # Compute pairwise correlations (ignore NaN)
        # Use pandas for NaN-aware Pearson
        df_tmp = pd.DataFrame(X)
        corr_mat = df_tmp.corr(method="pearson").values  # shape (n_feat, n_feat)

        # Sort features by descending variance
        variances = np.nanvar(X, axis=0)
        order = np.argsort(-variances)  # high variance first

        to_drop = set()
        dropped_pairs = []  # (kept_idx, dropped_idx)

        for i_pos, i in enumerate(order):
            if i in to_drop:
                continue
            for j in order[i_pos + 1:]:
                if j in to_drop:
                    continue
                r = corr_mat[i, j]
                if np.isnan(r):
                    continue
                if abs(r) > self.threshold:
                    to_drop.add(j)
                    dropped_pairs.append((int(i), int(j)))

        self.keep_mask_ = np.array(
            [i not in to_drop for i in range(n_feat)], dtype=bool
        )
        self.dropped_pairs_ = dropped_pairs  # (kept_col_idx, dropped_col_idx) in transformed space
        self.n_dropped_ = len(to_drop)
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.keep_mask_))])[self.keep_mask_]
        return np.asarray(input_features)[self.keep_mask_]


# ─────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    corr_threshold: float = CORR_THRESHOLD,
    near_zero_frac: float = NEAR_ZERO_FRAC,
    max_missing_frac: float = MAX_MISSING_FRAC,
) -> Pipeline:
    """
    Build and return the feature preprocessing Pipeline.

    Steps (all fit on train fold only inside nested CV):
      1. missingness_filter   → drop cols with >max_missing_frac NaN
      2. imputer              → median imputation (after dropping high-NaN cols)
      3. near_zero_var        → drop near-constant features
      4. corr_filter          → drop one from each correlated pair
      5. scaler               → RobustScaler (IQR-based, outlier-robust)
    """
    return Pipeline([
        ("missingness_filter", MissingnessFilter(max_missing_frac=max_missing_frac)),
        ("imputer", SimpleImputer(strategy="median")),
        ("near_zero_var", NearZeroVarianceFilter(frac=near_zero_frac)),
        ("corr_filter", CorrelationFilter(threshold=corr_threshold)),
        ("scaler", RobustScaler()),
    ])


# ─────────────────────────────────────────────────────────────
# Feature selection report (run on full dataset, outside CV)
# ─────────────────────────────────────────────────────────────

def generate_feature_selection_report(
    X: pd.DataFrame,
    output_dir: Path,
    corr_threshold: float = CORR_THRESHOLD,
    near_zero_frac: float = NEAR_ZERO_FRAC,
    max_missing_frac: float = MAX_MISSING_FRAC,
) -> pd.DataFrame:
    """
    Fit the preprocessing pipeline on the full feature matrix X and produce
    a CSV report listing every dropped feature and the reason it was dropped.

    This is run once for reporting purposes — the actual CV uses the Pipeline
    fitted independently on each train fold.

    Args:
        X: DataFrame of feature columns only (no clinical columns).
        output_dir: Directory where feature_selection_report.csv is saved.
        corr_threshold, near_zero_frac, max_missing_frac: filter parameters.

    Returns:
        DataFrame with columns: feature, reason, correlated_with
    """
    feature_names = np.array(X.columns.tolist())
    X_arr = X.values.astype(float)
    n_orig = X_arr.shape[1]

    rows = []

    # Step 1: Missingness filter
    miss_filt = MissingnessFilter(max_missing_frac=max_missing_frac)
    miss_filt.fit(X_arr)
    dropped_miss = feature_names[~miss_filt.keep_mask_]
    for feat in dropped_miss:
        miss_frac = np.mean(np.isnan(X_arr[:, list(feature_names).index(feat)]))
        rows.append({"feature": feat, "reason": f"high_missingness (frac={miss_frac:.2f})", "correlated_with": ""})
    X_arr2 = miss_filt.transform(X_arr)
    names2 = feature_names[miss_filt.keep_mask_]

    # Step 2: Imputation (no features dropped)
    imputer = SimpleImputer(strategy="median")
    X_arr3 = imputer.fit_transform(X_arr2)

    # Step 3: Near-zero variance filter
    nzv = NearZeroVarianceFilter(frac=near_zero_frac)
    nzv.fit(X_arr3)
    dropped_nzv = names2[~nzv.keep_mask_]
    for feat in dropped_nzv:
        rows.append({"feature": feat, "reason": "near_zero_variance", "correlated_with": ""})
    X_arr4 = nzv.transform(X_arr3)
    names4 = names2[nzv.keep_mask_]

    # Step 4: Correlation filter
    cf = CorrelationFilter(threshold=corr_threshold)
    cf.fit(X_arr4)
    dropped_corr_mask = ~cf.keep_mask_
    # Build a map: dropped_idx → kept_idx (in names4 space)
    kept_by_dropped = {}
    for kept_idx, dropped_idx in cf.dropped_pairs_:
        kept_by_dropped[dropped_idx] = kept_idx
    for j, feat in enumerate(names4):
        if dropped_corr_mask[j]:
            kept_feat = names4[kept_by_dropped.get(j, 0)]
            rows.append({
                "feature": feat,
                "reason": f"high_correlation (|r|>{corr_threshold})",
                "correlated_with": kept_feat,
            })

    report = pd.DataFrame(rows, columns=["feature", "reason", "correlated_with"])

    n_kept = n_orig - len(report)
    logger.info(
        f"Feature selection (full dataset): {n_orig} features → {n_kept} kept "
        f"({len(dropped_miss)} missingness, {len(dropped_nzv)} near-zero-var, "
        f"{(dropped_corr_mask).sum()} correlated)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "feature_selection_report.csv"
    report.to_csv(out_path, index=False)
    logger.info(f"Feature selection report saved: {out_path}")

    return report


# ─────────────────────────────────────────────────────────────
# Feature name tracing
# ─────────────────────────────────────────────────────────────

def get_pipeline_feature_names(pipeline: Pipeline, input_names: list[str]) -> list[str]:
    """
    Trace feature names through a fitted preprocessing pipeline.

    Works by applying each step's keep_mask_ in order.
    The scaler and imputer steps do not change the number of features, so they are skipped.

    Args:
        pipeline: A fitted Pipeline returned by build_preprocessing_pipeline().
        input_names: Feature names before any pipeline step (same length as X.shape[1]).

    Returns:
        List of feature names that survive all filtering steps.
    """
    names = np.asarray(input_names)
    for step_name, step in pipeline.steps:
        if hasattr(step, "keep_mask_"):
            names = names[step.keep_mask_]
    return list(names)


# ─────────────────────────────────────────────────────────────
# Feature matrix preparation helpers
# ─────────────────────────────────────────────────────────────

def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    include_gender: bool = True,
    include_age: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract and optionally augment the feature matrix.

    Args:
        df: Full DataFrame (clinical + features).
        feature_cols: List of pose feature column names.
        include_gender: Include gender (encoded) as a feature.
        include_age: Include Ados_2_Age as a feature.

    Returns:
        (X, used_feature_cols): Feature DataFrame and final column list.
    """
    cols = list(feature_cols)

    if include_gender and "gender" in df.columns:
        df = encode_gender(df)
        if "gender" not in cols:
            cols = ["gender"] + cols

    if include_age and "Ados_2_Age" in df.columns:
        if "Ados_2_Age" not in cols:
            cols = ["Ados_2_Age"] + cols

    X = df[cols].copy()
    return X, cols
