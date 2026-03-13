#!/usr/bin/env python3
"""
ML prediction pipeline for pose-based ASD classification and ADOS regression.

Usage
-----
    python -m ml_analysis.run_ml \\
        --csv            dataset/info/child_for_humanlisbet_paper_with_paths.csv \\
        --pose-records   dataset/pose_records \\
        --output-dir     ml_analysis/results \\
        [--n-jobs 16] \\
        [--use-gpu] \\
        [--corr-threshold 0.95] \\
        [--n-outer-folds 5] \\
        [--n-inner-folds 3] \\
        [--n-iter 50] \\
        [--skip-dimreduce] [--skip-classification] [--skip-regression] [--skip-explain] \\
        [--debug-n 5]

Steps
-----
  1. Feature extraction (frame-level → 11 stats × metric × variant)
  2. Preprocessing report (feature selection on full dataset, informational)
  3. Dimensionality reduction (PCA + UMAP)
  4. Classification (ASD vs TD, nested CV)
  5. Regression (ADOS-2 total, nested CV)
  6. SHAP explainability for best models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from within the ml_analysis directory
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "ml_analysis"

from .features import load_feature_matrix, get_feature_columns
from .preprocessing import (
    prepare_feature_matrix,
    generate_feature_selection_report,
)
from .dimreduce import run_dimensionality_reduction
from .classification import run_classification, build_strat_label
from .regression import run_regression, build_ados_strat
from .explain import run_explainability


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True, type=Path,
        help="Path to child_for_humanlisbet_paper_with_paths.csv",
    )
    parser.add_argument(
        "--pose-records", required=True, type=Path,
        help="Path to the pose_records/ directory",
    )
    parser.add_argument(
        "--output-dir", default=Path("ml_analysis/results"), type=Path,
        help="Output directory (created if needed). Default: ml_analysis/results",
    )
    parser.add_argument(
        "--n-jobs", default=4, type=int,
        help="Parallel workers for feature loading and outer CV folds (default: 4)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="Enable GPU support for XGBoost, LightGBM, and UMAP (requires CUDA)",
    )
    parser.add_argument(
        "--corr-threshold", default=0.95, type=float,
        help="Pearson |r| threshold for correlation filter (default: 0.95)",
    )
    parser.add_argument(
        "--n-outer-folds", default=5, type=int,
        help="Number of outer CV folds (default: 5)",
    )
    parser.add_argument(
        "--n-inner-folds", default=3, type=int,
        help="Number of inner CV folds for hyperparameter search (default: 3)",
    )
    parser.add_argument(
        "--n-iter", default=50, type=int,
        help="RandomizedSearchCV iterations per inner search (default: 50)",
    )
    parser.add_argument(
        "--umap-n-neighbors", default=15, type=int,
        help="UMAP n_neighbors parameter (default: 15)",
    )
    parser.add_argument(
        "--umap-min-dist", default=0.1, type=float,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    parser.add_argument(
        "--skip-dimreduce", action="store_true",
        help="Skip dimensionality reduction step",
    )
    parser.add_argument(
        "--skip-classification", action="store_true",
        help="Skip classification step",
    )
    parser.add_argument(
        "--skip-regression", action="store_true",
        help="Skip regression step",
    )
    parser.add_argument(
        "--skip-explain", action="store_true",
        help="Skip SHAP explainability step",
    )
    parser.add_argument(
        "--debug-n", default=None, type=int,
        help="Limit to first N subjects for quick testing",
    )
    parser.add_argument(
        "--random-state", default=42, type=int,
        help="Global random seed (default: 42)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level),
        stream=sys.stdout,
    )


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger("ml_analysis")

    # Validate inputs
    if not args.csv.exists():
        logger.error(f"CSV not found: {args.csv}")
        sys.exit(1)
    if not args.pose_records.exists():
        logger.error(f"pose_records directory not found: {args.pose_records}")
        sys.exit(1)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    random_state = args.random_state

    # ══════════════════════════════════════════════════════════
    # STEP 1: Feature extraction
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("STEP 1: Feature extraction (frame-level)")
    logger.info("=" * 65)

    df = load_feature_matrix(
        csv_path=args.csv,
        pose_records_dir=args.pose_records,
        n_jobs=args.n_jobs,
        debug_n=args.debug_n,
    )

    if df.empty:
        logger.error("No subjects loaded. Check paths and CSV contents.")
        sys.exit(1)

    # Save feature matrix
    feat_path = out / "feature_matrix.csv"
    df.to_csv(feat_path, index=False)
    logger.info(f"Feature matrix saved: {feat_path}  ({df.shape[0]} × {df.shape[1]})")

    # ── Prepare feature matrix (raw + norm, + gender + age) ───
    feature_cols = get_feature_columns(df)
    X_df, final_feature_cols = prepare_feature_matrix(
        df, feature_cols, include_gender=True, include_age=True
    )
    X_raw = X_df.values.astype(float)
    feature_names = list(X_df.columns)

    logger.info(f"Feature matrix: {X_raw.shape[0]} subjects × {X_raw.shape[1]} features")

    # Clinical meta for downstream use
    clinical_cols = ["uuid", "diagnosis", "gender", "Ados_2_Age", "ADOS_2_TOTAL", "Ados_2_Module"]
    df_meta = df[[c for c in clinical_cols if c in df.columns]].copy()

    # ── Binary labels (ASD=1, TD=0) ───────────────────────────
    y_clf_all = (df_meta["diagnosis"] == "ASD").astype(int).values if "diagnosis" in df_meta else None

    # ── Continuous ADOS scores ─────────────────────────────────
    if "ADOS_2_TOTAL" in df_meta.columns:
        ados = pd.to_numeric(df_meta["ADOS_2_TOTAL"], errors="coerce")
        valid_ados_mask = ados.notna().values
        y_reg_all = ados.values.astype(float)
    else:
        valid_ados_mask = np.zeros(len(df), dtype=bool)
        y_reg_all = None

    # ══════════════════════════════════════════════════════════
    # STEP 2: Feature selection report (informational, full dataset)
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("STEP 2: Feature selection report (informational)")
    logger.info("=" * 65)

    preproc_dir = out / "preprocessing"
    try:
        generate_feature_selection_report(
            X=X_df,
            output_dir=preproc_dir,
            corr_threshold=args.corr_threshold,
        )
    except Exception as exc:
        logger.warning(f"Feature selection report failed: {exc}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Dimensionality reduction
    # ══════════════════════════════════════════════════════════
    dimreduce_result = {}
    if not args.skip_dimreduce:
        logger.info("=" * 65)
        logger.info("STEP 3: Dimensionality reduction (PCA + UMAP)")
        logger.info("=" * 65)

        # Apply the full preprocessing pipeline (missingness → impute → near-zero-var
        # → correlation filter → RobustScaler) so that highly correlated features do
        # not artificially concentrate variance in the first PCA components.
        from .preprocessing import build_preprocessing_pipeline, get_pipeline_feature_names
        preproc_dimreduce = build_preprocessing_pipeline(corr_threshold=args.corr_threshold)
        X_scaled = preproc_dimreduce.fit_transform(X_raw)
        feature_names_filtered = get_pipeline_feature_names(preproc_dimreduce, feature_names)
        logger.info(
            f"  Preprocessing for dimreduce: {X_raw.shape[1]} → {X_scaled.shape[1]} features "
            f"({X_raw.shape[1] - X_scaled.shape[1]} removed by var/corr filters)"
        )

        try:
            dimreduce_result = run_dimensionality_reduction(
                X_scaled=X_scaled,
                df_meta=df_meta,
                feature_names=feature_names_filtered,
                output_dir=out,
                use_gpu=args.use_gpu,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist,
                random_state=random_state,
            )
        except Exception as exc:
            logger.error(f"Dimensionality reduction failed: {exc}", exc_info=True)
    else:
        logger.info("STEP 3: Skipped (--skip-dimreduce)")

    # ══════════════════════════════════════════════════════════
    # STEP 4: Classification (ASD vs TD)
    # ══════════════════════════════════════════════════════════
    df_cv_clf = None
    if not args.skip_classification and y_clf_all is not None:
        logger.info("=" * 65)
        logger.info("STEP 4: Classification (ASD vs TD)")
        logger.info("=" * 65)

        logger.info(
            f"  ASD: {(y_clf_all == 1).sum()}, TD: {(y_clf_all == 0).sum()}, "
            f"Total: {len(y_clf_all)}"
        )
        try:
            df_cv_clf = run_classification(
                X=X_raw,
                y=y_clf_all,
                df_meta=df_meta,
                feature_names=feature_names,
                output_dir=out,
                n_outer=args.n_outer_folds,
                n_inner=args.n_inner_folds,
                n_iter=args.n_iter,
                corr_threshold=args.corr_threshold,
                use_gpu=args.use_gpu,
                n_jobs=args.n_jobs,
                random_state=random_state,
            )
        except Exception as exc:
            logger.error(f"Classification failed: {exc}", exc_info=True)
    elif args.skip_classification:
        logger.info("STEP 4: Skipped (--skip-classification)")
    else:
        logger.info("STEP 4: Skipped (no diagnosis column found)")

    # ══════════════════════════════════════════════════════════
    # STEP 5: Regression (ADOS-2 total)
    # ══════════════════════════════════════════════════════════
    df_cv_reg = None
    if not args.skip_regression and y_reg_all is not None and valid_ados_mask.sum() >= 10:
        logger.info("=" * 65)
        logger.info("STEP 5: Regression (ADOS-2 total score)")
        logger.info("=" * 65)

        # Restrict to subjects with valid ADOS
        X_reg = X_raw[valid_ados_mask]
        y_reg = y_reg_all[valid_ados_mask]
        df_meta_reg = df_meta[valid_ados_mask].reset_index(drop=True)
        logger.info(f"  Subjects with valid ADOS: {valid_ados_mask.sum()}")

        try:
            df_cv_reg = run_regression(
                X=X_reg,
                y=y_reg,
                df_meta=df_meta_reg,
                feature_names=feature_names,
                output_dir=out,
                n_outer=args.n_outer_folds,
                n_inner=args.n_inner_folds,
                n_iter=args.n_iter,
                corr_threshold=args.corr_threshold,
                use_gpu=args.use_gpu,
                n_jobs=args.n_jobs,
                random_state=random_state,
            )
        except Exception as exc:
            logger.error(f"Regression failed: {exc}", exc_info=True)
    elif args.skip_regression:
        logger.info("STEP 5: Skipped (--skip-regression)")
    else:
        logger.info(f"STEP 5: Skipped (only {valid_ados_mask.sum()} subjects with valid ADOS)")

    # ══════════════════════════════════════════════════════════
    # STEP 6: SHAP explainability
    # ══════════════════════════════════════════════════════════
    if not args.skip_explain and df_cv_clf is not None:
        logger.info("=" * 65)
        logger.info("STEP 6: SHAP explainability")
        logger.info("=" * 65)

        y_reg_explain = None
        df_meta_reg_explain = None
        X_reg_explain = X_raw

        if df_cv_reg is not None and valid_ados_mask.sum() >= 10:
            y_reg_explain = y_reg_all[valid_ados_mask]
            X_reg_explain = X_raw[valid_ados_mask]
            df_meta_reg_explain = df_meta[valid_ados_mask].reset_index(drop=True)

        try:
            run_explainability(
                X=X_raw,
                y_clf=y_clf_all,
                y_reg=y_reg_explain if df_cv_reg is not None else None,
                df_meta=df_meta,
                feature_names=feature_names,
                df_cv_clf=df_cv_clf,
                df_cv_reg=df_cv_reg,
                output_dir=out,
                corr_threshold=args.corr_threshold,
                use_gpu=args.use_gpu,
                random_state=random_state,
                X_reg=X_reg_explain if df_cv_reg is not None else None,
                df_meta_reg=df_meta_reg_explain if df_cv_reg is not None else None,
            )
        except Exception as exc:
            logger.error(f"Explainability failed: {exc}", exc_info=True)
    elif args.skip_explain:
        logger.info("STEP 6: Skipped (--skip-explain)")
    else:
        logger.info("STEP 6: Skipped (classification step was not run)")

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("ML PIPELINE COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Results saved to: {out.resolve()}")

    if df_cv_clf is not None:
        best_clf = df_cv_clf.groupby("model")["auc_roc"].mean().idxmax()
        best_auc = df_cv_clf.groupby("model")["auc_roc"].mean().max()
        logger.info(f"  Classification best model: {best_clf} (mean AUC={best_auc:.3f})")

    if df_cv_reg is not None:
        best_reg = df_cv_reg.groupby("model")["rmse"].mean().idxmin()
        best_rmse = df_cv_reg.groupby("model")["rmse"].mean().min()
        best_r2 = df_cv_reg[df_cv_reg["model"] == best_reg]["r2"].mean()
        logger.info(
            f"  Regression best model: {best_reg} "
            f"(mean RMSE={best_rmse:.3f}, R²={best_r2:.3f})"
        )


if __name__ == "__main__":
    main()
