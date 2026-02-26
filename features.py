"""
Feature extraction for the ML pipeline.

Loads the pre-concatenated frame-level CSV files (video_metrics_raw_2d.csv and
video_metrics_normalised.csv) for each subject and computes 11 summary statistics
per metric × variant directly from the full frame sequence, avoiding the
segment-length bias that would arise from aggregating segment-level summaries.

Statistics computed per metric column (on valid / non-NaN frames):
    mean, std, q25, median, q75, iqr, min, max, cv, skewness, kurtosis

Column naming convention (consistent with pose_analysis):
    {base_metric}__{raw|norm}__{stat_type}
    e.g.  child_speed_kp_left_wrist__raw__skewness

Special handling:
  - segment_id column  → skipped (not a metric)
  - kp_set_changed (bool) → only mean computed (proportion of frames where the
    visible keypoint set differed from the previous frame); other stats are NaN
  - Metrics with fewer than MIN_VALID_FRAMES valid frames → all stats set to NaN
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as scipy_stats
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Minimum number of valid (non-NaN) frames required to compute stats
MIN_VALID_FRAMES = 50

# stat_types computed for float metrics
FLOAT_STAT_TYPES = [
    "mean", "std", "q10", "q25", "median", "q75", "q90", "iqr",
    "cv", "skewness", "kurtosis",
]

# stat_types computed for boolean metrics (kp_set_changed)
BOOL_STAT_TYPES = ["mean"]  # proportion of True frames

# Columns to skip entirely
SKIP_COLS = {"segment_id"}


# ─────────────────────────────────────────────────────────────
# Per-array statistics
# ─────────────────────────────────────────────────────────────

def _stats_float(arr: np.ndarray) -> dict[str, float]:
    """Compute all float statistics for one metric column."""
    valid = arr[~np.isnan(arr)]
    if len(valid) < MIN_VALID_FRAMES:
        return {s: np.nan for s in FLOAT_STAT_TYPES}

    q10, q25, median, q75, q90 = np.percentile(valid, [10, 25, 50, 75, 90])
    mean = float(np.mean(valid))
    iqr = float(q75 - q25)

    # IQR-based Tukey fence clipping (k=3) before computing spread and
    # higher-order statistics.
    #
    # Context: the normalised variant divides every frame value by trunk height.
    # When trunk height is momentarily mis-estimated (a common MMPose artefact)
    # the division produces spikes that are 10–100× the typical value for that
    # subject.  A fixed percentile clip (e.g. p1/p99) does not adapt to the
    # metric's scale and can still include burst artefacts lasting several
    # seconds (>1% of frames).
    #
    # IQR-based clipping is adaptive and well-motivated:
    #   lower fence = Q1 − 3·IQR  (equivalent to ~±4σ for Gaussian data)
    #   upper fence = Q3 + 3·IQR
    # Frames outside these fences are replaced by the fence value.  This
    # removes tracking artefacts while preserving genuine movement peaks
    # (fast gestures = only a few frames, well inside the fence).
    #
    # Statistics computed on original (unclipped) frames:
    #   mean, q10, q25, median, q75, q90, iqr
    #
    # Statistics computed on IQR-clipped frames:
    #   std, cv, skewness, kurtosis  — all sensitive to extreme frame values
    #
    # p10 / p90 replace min / max:
    #   min / max of a non-negative metric (e.g. acceleration) is essentially
    #   0 / a single artefact frame, giving near-zero IQR and degenerate
    #   robust-scaled values.  p10 = "typical low" and p90 = "typical high"
    #   behaviour over the session are semantically meaningful and robust.
    if iqr > 0:
        lo = q25 - 3.0 * iqr
        hi = q75 + 3.0 * iqr
    else:
        # Metric is constant (e.g. angle locked at 0); std/cv/moments are 0.
        lo, hi = q25, q75
    valid_wins = np.clip(valid, lo, hi)
    std = float(np.std(valid_wins, ddof=1))
    mean_wins = float(np.mean(valid_wins))
    cv = std / mean_wins if abs(mean_wins) > 0.01 * std else np.nan

    with np.errstate(all="ignore"):
        skewness = float(scipy_stats.skew(valid_wins, bias=False))
        kurtosis = float(scipy_stats.kurtosis(valid_wins, bias=False))  # excess kurtosis

    return {
        "mean": mean,
        "std": std,
        "q10": float(q10),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "q90": float(q90),
        "iqr": iqr,
        "cv": cv,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def _stats_bool(arr: np.ndarray) -> dict[str, float]:
    """Compute mean (proportion True) for a boolean metric column."""
    valid = arr[~np.isnan(arr.astype(float))]
    if len(valid) < MIN_VALID_FRAMES:
        return {"mean": np.nan}
    return {"mean": float(np.mean(valid.astype(float)))}


# ─────────────────────────────────────────────────────────────
# Per-subject loading
# ─────────────────────────────────────────────────────────────

def _load_one_subject(
    pose_records_dir: Path,
    stem: str,
    subject_id: str,
) -> tuple[str, pd.Series | None]:
    """Load frame-level CSVs for one subject and compute all statistics.

    Returns (subject_id, pd.Series of features) or (subject_id, None) on failure.
    """
    subj_dir = pose_records_dir / stem
    raw_path = subj_dir / "video_metrics_raw_2d.csv"
    norm_path = subj_dir / "video_metrics_normalised.csv"

    if not raw_path.exists():
        logger.warning(f"  Missing: {raw_path.name} for {stem}")
        return subject_id, None

    try:
        df_raw = pd.read_csv(raw_path, index_col=0)
    except Exception as exc:
        logger.warning(f"  Failed to read {raw_path}: {exc}")
        return subject_id, None

    df_norm = None
    if norm_path.exists():
        try:
            df_norm = pd.read_csv(norm_path, index_col=0)
        except Exception as exc:
            logger.warning(f"  Failed to read {norm_path}: {exc}")

    records: dict[str, float] = {}

    for col in df_raw.columns:
        if col in SKIP_COLS:
            continue

        dtype = df_raw[col].dtype
        arr_raw = df_raw[col].values

        if dtype == bool or str(dtype) == "bool":
            # Boolean: kp_set_changed → proportion only
            for stat_name, val in _stats_bool(arr_raw).items():
                records[f"{col}__raw__{stat_name}"] = val
            # norm version is identical for booleans → skip
            continue

        # Float metric — raw variant
        arr_float = arr_raw.astype(float)
        for stat_name, val in _stats_float(arr_float).items():
            records[f"{col}__raw__{stat_name}"] = val

        # Normalised variant
        if df_norm is not None and col in df_norm.columns:
            arr_norm = df_norm[col].values.astype(float)
            for stat_name, val in _stats_float(arr_norm).items():
                records[f"{col}__norm__{stat_name}"] = val

    return subject_id, pd.Series(records)


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────

def load_feature_matrix(
    csv_path: Path,
    pose_records_dir: Path,
    n_jobs: int = 4,
    debug_n: int | None = None,
) -> pd.DataFrame:
    """Load the clinical CSV, extract frame-level features for each subject,
    and return a merged DataFrame ready for ML.

    Args:
        csv_path: Path to child_for_humanlisbet_paper_with_paths.csv.
        pose_records_dir: Path to pose_records/ directory.
        n_jobs: Number of parallel workers for loading.
        debug_n: If set, limit to first N subjects (for quick testing).

    Returns:
        DataFrame with one row per subject.
        Clinical columns: uuid, diagnosis, gender, Ados_2_Age, ADOS_2_TOTAL, Ados_2_Module.
        Feature columns: {metric}__{raw|norm}__{stat_type}.
    """
    meta = pd.read_csv(csv_path)
    meta = meta.dropna(subset=["results_path"])
    meta = meta[meta["diagnosis"].isin(["ASD", "TD"])].reset_index(drop=True)
    logger.info(f"Clinical CSV: {len(meta)} subjects with ASD/TD diagnosis")

    if debug_n is not None:
        meta = meta.head(debug_n)
        logger.info(f"Debug mode: limited to first {debug_n} subjects")

    tasks = [
        (pose_records_dir, Path(row["results_path"]).stem, str(row.get("uuid", i)))
        for i, row in meta.iterrows()
    ]

    logger.info(f"Extracting features from {len(tasks)} subjects (n_jobs={n_jobs}) …")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_load_one_subject)(prd, stem, uid)
        for prd, stem, uid in tqdm(tasks, desc="Loading subjects")
    )

    rows: list[dict] = []
    skipped: list[str] = []

    clinical_cols = [
        "uuid", "diagnosis", "gender", "Ados_2_Age", "ADOS_2_TOTAL", "Ados_2_Module",
    ]

    for (subject_id, features), (_, meta_row) in zip(results, meta.iterrows()):
        if features is None:
            skipped.append(subject_id)
            continue
        record = {k: meta_row[k] for k in clinical_cols if k in meta_row.index}
        record.update(features.to_dict())
        rows.append(record)

    if skipped:
        logger.warning(
            f"Skipped {len(skipped)} subjects (no pose records): "
            + ", ".join(skipped[:5]) + ("…" if len(skipped) > 5 else "")
        )

    df = pd.DataFrame(rows)
    n_feat = len([c for c in df.columns if "__raw__" in c or "__norm__" in c])
    logger.info(f"Feature matrix: {len(df)} subjects × {n_feat} feature columns")
    return df


def get_feature_columns(
    df: pd.DataFrame,
    variant: str | None = None,
    stat_type: str | None = None,
) -> list[str]:
    """Return feature column names, optionally filtered by variant and stat_type."""
    cols = [c for c in df.columns if "__raw__" in c or "__norm__" in c]
    if variant is not None:
        cols = [c for c in cols if f"__{variant}__" in c]
    if stat_type is not None:
        cols = [c for c in cols if c.endswith(f"__{stat_type}")]
    return cols
