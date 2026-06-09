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
from .classification import run_classification, build_strat_label, build_strat_label_from_y
from .regression import run_regression, build_ados_strat
from .explain import run_explainability
from .config import load_config, validate_config, apply_config, exploratory_settings


# ─────────────────────────────────────────────────────────────
# All known target columns (from the CSV)
# ─────────────────────────────────────────────────────────────

ALL_TARGET_COLS = [
    # ADOS-2
    "ADOS_2_ADOS_G_revised_RRB_level_of_symptoms",
    "ADOS_2_ADOS_G_REVISED_RRB_SEVERITY_SCORE_new",
    "ADOS_2_ADOS_G_REVISED_SA_LEVEL_OF_SYMPTOMS",
    "ADOS_2_ADOS_G_REVISED_SA_SEVERITY_SCORE",
    "ADOS_2_SOCIAL_AFECT_TOTAL",
    "ADOS_2_TOTAL",
    "ADOS_G_ADOS_2_TOTAL_score_de_severite",
    "ADOS_G_REVISED_ADOS_2_TOTAL_Level_of_symptoms",
    # Diagnosis / demographics
    "diagnosis",
    "gender",
    # Vineland-II (VLDII)
    "VLDII_AdSS", "VLDII_MotorSS", "VLDII_gmsVS", "VLDII_fmsVS",
    "VLDII_SocSS", "VLDII_intVS", "VLDII_plaVS", "VLDII_copVS",
    "VLDII_DaiSS", "VLDII_perVS", "VLDII_comVS", "VLDII_domVS",
    "VLDII_ComSS", "VLDII_expVS", "VLDII_recVS",
    # Mullen (MSEL)
    "MSEL_TOTAL_DQ", "MSEL_FM_DQ", "MSEL_VR_DQ", "MSEL_LR_DQ",
    "MSEL_LE_DQ", "MSEL_NV_DQ", "MSEL_V_DQ", "MSEL_GM_DQ",
]


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
    parser.add_argument(
        "--config", default=None, type=Path,
        help=(
            "Path to a YAML (or JSON) config file. "
            "When provided, runs in config mode: only the specified methods and models "
            "are executed; fixed hyperparameters skip inner-CV search. "
            "Without --config the pipeline runs in exploratory mode (all models, "
            "full random search, PCA + UMAP) — identical to the original behaviour."
        ),
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


def detect_target_type(col: pd.Series) -> str:
    """Return 'classification' if col contains string labels, 'regression' otherwise.

    Detection logic:
      - If dtype is object (string) → classification
      - If coercing to numeric leaves all NaN → classification
      - Otherwise → regression
    """
    if col.dtype == object or col.dtype.name == "category":
        return "classification"
    numeric = pd.to_numeric(col.dropna(), errors="coerce")
    if numeric.isna().all():
        return "classification"
    return "regression"


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    logger = logging.getLogger("ml_analysis")

    # ── Mode selection ─────────────────────────────────────────
    if args.config is not None:
        logger.info(f"Mode: CONFIG  ({args.config})")
        cfg = load_config(args.config)
        validate_config(cfg)
        settings = apply_config(cfg, args)
        logger.info(
            f"  dimreduce methods : {settings['dimreduce_methods']}\n"
            f"  clf model filter  : {settings['clf_model_filter']}\n"
            f"  clf fixed params  : {list(settings['clf_fixed_params'])}\n"
            f"  reg model filter  : {settings['reg_model_filter']}\n"
            f"  reg fixed params  : {list(settings['reg_fixed_params'])}"
        )
    else:
        logger.info("Mode: EXPLORATORY  (all models, full random search, PCA + UMAP)")
        settings = exploratory_settings(args)

    # Unpack settings into local variables for readability
    n_jobs           = settings["n_jobs"]
    random_state     = settings["random_state"]
    corr_threshold   = settings["corr_threshold"]
    clf_n_outer_folds = settings["clf_n_outer_folds"]
    clf_n_inner_folds = settings["clf_n_inner_folds"]
    clf_n_iter        = settings["clf_n_iter"]
    clf_use_smote     = settings["clf_use_smote"]
    reg_n_outer_folds = settings["reg_n_outer_folds"]
    reg_n_inner_folds = settings["reg_n_inner_folds"]
    reg_n_iter        = settings["reg_n_iter"]
    umap_n_neighbors = settings["umap_n_neighbors"]
    umap_min_dist    = settings["umap_min_dist"]
    skip_dimreduce      = settings["skip_dimreduce"]
    skip_classification = settings["skip_classification"]
    skip_regression     = settings["skip_regression"]
    skip_explain        = settings["skip_explain"]
    dimreduce_methods   = settings["dimreduce_methods"]
    clf_model_filter    = settings["clf_model_filter"]
    clf_fixed_params    = settings["clf_fixed_params"]
    reg_model_filter    = settings["reg_model_filter"]
    reg_fixed_params    = settings["reg_fixed_params"]
    targets_list        = settings["targets_list"]
    explain_targets     = settings["explain_targets"]

    # Validate inputs
    if not args.csv.exists():
        logger.error(f"CSV not found: {args.csv}")
        sys.exit(1)
    if not args.pose_records.exists():
        logger.error(f"pose_records directory not found: {args.pose_records}")
        sys.exit(1)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # STEP 1: Feature extraction
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("STEP 1: Feature extraction (frame-level)")
    logger.info("=" * 65)

    df = load_feature_matrix(
        csv_path=args.csv,
        pose_records_dir=args.pose_records,
        n_jobs=n_jobs,
        debug_n=args.debug_n,
        extra_meta_cols=ALL_TARGET_COLS,
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

    # Clinical meta for downstream use — load all potential target columns (+identifiers)
    base_meta_cols = ["uuid", "Ados_2_Age", "Ados_2_Module"]
    present_target_cols = [c for c in ALL_TARGET_COLS if c in df.columns]
    clinical_cols = base_meta_cols + present_target_cols
    df_meta = df[[c for c in clinical_cols if c in df.columns]].copy()

    # gender and age are always prepended as input features (never used as targets)

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
            corr_threshold=corr_threshold,
        )
    except Exception as exc:
        logger.warning(f"Feature selection report failed: {exc}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: Dimensionality reduction
    # ══════════════════════════════════════════════════════════
    dimreduce_result = {}
    if not skip_dimreduce:
        logger.info("=" * 65)
        active_methods = " + ".join(m.upper() for m in dimreduce_methods)
        logger.info(f"STEP 3: Dimensionality reduction ({active_methods})")
        logger.info("=" * 65)

        # Build minimal df_meta for dimreduce (uuid + diagnosis needed for coloring)
        dimreduce_meta_cols = [
            "uuid", "diagnosis", "gender", "Ados_2_Age",
            "ADOS_2_TOTAL", "Ados_2_Module",
            "ADOS_2_ADOS_G_REVISED_RRB_SEVERITY_SCORE_new",
            "ADOS_2_ADOS_G_REVISED_SA_SEVERITY_SCORE",
            "ADOS_G_ADOS_2_TOTAL_score_de_severite",
        ]
        df_meta_dimreduce = df_meta[[c for c in dimreduce_meta_cols
                                     if c in df_meta.columns]].copy()

        # Apply the full preprocessing pipeline (missingness → impute → near-zero-var
        # → correlation filter → RobustScaler) so that highly correlated features do
        # not artificially concentrate variance in the first PCA components.
        from .preprocessing import build_preprocessing_pipeline, get_pipeline_feature_names
        preproc_dimreduce = build_preprocessing_pipeline(corr_threshold=corr_threshold)
        X_scaled = preproc_dimreduce.fit_transform(X_raw)
        feature_names_filtered = get_pipeline_feature_names(preproc_dimreduce, feature_names)
        logger.info(
            f"  Preprocessing for dimreduce: {X_raw.shape[1]} → {X_scaled.shape[1]} features "
            f"({X_raw.shape[1] - X_scaled.shape[1]} removed by var/corr filters)"
        )

        try:
            dimreduce_result = run_dimensionality_reduction(
                X_scaled=X_scaled,
                df_meta=df_meta_dimreduce,
                feature_names=feature_names_filtered,
                output_dir=out,
                use_gpu=args.use_gpu,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
                random_state=random_state,
                methods=dimreduce_methods,
            )
        except Exception as exc:
            logger.error(f"Dimensionality reduction failed: {exc}", exc_info=True)
    else:
        logger.info("STEP 3: Skipped (--skip-dimreduce)")

    # ══════════════════════════════════════════════════════════
    # STEPS 4-6: Multi-target loop (classification + regression + SHAP)
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info(f"STEPS 4-6: Multi-target loop ({len(targets_list)} targets)")
    logger.info("=" * 65)

    from sklearn.preprocessing import LabelEncoder

    # Store CV results per target for the final summary
    target_cv_results: dict[str, tuple[str, pd.DataFrame]] = {}

    for target_col in targets_list:
        logger.info("─" * 65)
        logger.info(f"Target: {target_col!r}")
        logger.info("─" * 65)

        if target_col not in df_meta.columns:
            logger.warning(f"  '{target_col}' not found in CSV — skipping")
            continue

        col_series = df_meta[target_col]
        task_type = detect_target_type(col_series)

        # ── Build valid-subject mask ───────────────────────────
        if task_type == "classification":
            valid_mask = (
                col_series.notna()
                & (col_series.astype(str).str.strip() != "")
                & (col_series.astype(str).str.lower() != "nan")
            )
        else:
            numeric_col = pd.to_numeric(col_series, errors="coerce")
            valid_mask = numeric_col.notna()

        n_valid = int(valid_mask.sum())
        if n_valid < 10:
            logger.warning(f"  '{target_col}': only {n_valid} valid subjects — skipping (need ≥10)")
            continue

        X_t = X_raw[valid_mask.values]
        df_meta_t = df_meta[valid_mask].reset_index(drop=True)

        # ── Classification ────────────────────────────────────
        if task_type == "classification":
            if skip_classification:
                logger.info(f"  '{target_col}': classification skipped (--skip-classification)")
                continue

            le = LabelEncoder()
            y_t = le.fit_transform(col_series[valid_mask].astype(str).values)
            label_names = list(le.classes_)
            n_classes = len(label_names)

            if n_classes < 2:
                logger.warning(
                    f"  '{target_col}': only {n_classes} class in this split — skipping"
                )
                continue

            logger.info(
                f"  '{target_col}': {n_classes}-class classification, n={n_valid}, "
                f"classes={label_names}"
            )

            # Save label-encoding map
            enc_path = out / "classification" / target_col
            enc_path.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"label": label_names, "encoded": range(n_classes)}).to_csv(
                enc_path / "label_encoding.csv", index=False
            )

            df_cv = None
            try:
                df_cv = run_classification(
                    X=X_t,
                    y=y_t,
                    df_meta=df_meta_t,
                    feature_names=feature_names,
                    output_dir=out,
                    target_name=target_col,
                    n_classes=n_classes,
                    label_names=label_names,
                    y_strat=build_strat_label_from_y(y_t),
                    n_outer=clf_n_outer_folds,
                    n_inner=clf_n_inner_folds,
                    n_iter=clf_n_iter,
                    corr_threshold=corr_threshold,
                    use_gpu=args.use_gpu,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    model_filter=clf_model_filter,
                    fixed_params=clf_fixed_params,
                    use_smote=clf_use_smote,
                )
                if df_cv is not None:
                    target_cv_results[target_col] = ("classification", df_cv)
            except Exception as exc:
                logger.error(f"  Classification '{target_col}' failed: {exc}", exc_info=True)

            # SHAP for this target
            if (
                not skip_explain
                and target_col in explain_targets
                and df_cv is not None
            ):
                try:
                    run_explainability(
                        X=X_t,
                        y_clf=y_t,
                        y_reg=None,
                        df_meta=df_meta_t,
                        feature_names=feature_names,
                        df_cv_clf=df_cv,
                        df_cv_reg=None,
                        output_dir=out,
                        target_name=target_col,
                        n_classes=n_classes,
                        corr_threshold=corr_threshold,
                        use_gpu=args.use_gpu,
                        random_state=random_state,
                    )
                except Exception as exc:
                    logger.error(f"  SHAP '{target_col}' failed: {exc}", exc_info=True)

        # ── Regression ────────────────────────────────────────
        else:
            if skip_regression:
                logger.info(f"  '{target_col}': regression skipped (--skip-regression)")
                continue

            y_t = pd.to_numeric(col_series[valid_mask], errors="coerce").values.astype(float)
            logger.info(
                f"  '{target_col}': regression, n={n_valid}, "
                f"range=[{y_t.min():.2f}, {y_t.max():.2f}]"
            )

            df_cv = None
            try:
                df_cv = run_regression(
                    X=X_t,
                    y=y_t,
                    df_meta=df_meta_t,
                    feature_names=feature_names,
                    output_dir=out,
                    target_name=target_col,
                    n_outer=reg_n_outer_folds,
                    n_inner=reg_n_inner_folds,
                    n_iter=reg_n_iter,
                    corr_threshold=corr_threshold,
                    use_gpu=args.use_gpu,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    model_filter=reg_model_filter,
                    fixed_params=reg_fixed_params,
                )
                if df_cv is not None:
                    target_cv_results[target_col] = ("regression", df_cv)
            except Exception as exc:
                logger.error(f"  Regression '{target_col}' failed: {exc}", exc_info=True)

            # SHAP for this target
            if (
                not skip_explain
                and target_col in explain_targets
                and df_cv is not None
            ):
                try:
                    run_explainability(
                        X=X_t,
                        y_clf=None,
                        y_reg=y_t,
                        df_meta=df_meta_t,
                        feature_names=feature_names,
                        df_cv_clf=None,
                        df_cv_reg=df_cv,
                        output_dir=out,
                        target_name=target_col,
                        corr_threshold=corr_threshold,
                        use_gpu=args.use_gpu,
                        random_state=random_state,
                    )
                except Exception as exc:
                    logger.error(f"  SHAP '{target_col}' failed: {exc}", exc_info=True)

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info("ML PIPELINE COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Results saved to: {out.resolve()}")
    logger.info(f"Completed {len(target_cv_results)}/{len(targets_list)} targets")

    for target_col, (task_type, df_cv) in target_cv_results.items():
        if task_type == "classification":
            best_m = df_cv.groupby("model")["auc_roc"].mean().idxmax()
            best_v = df_cv.groupby("model")["auc_roc"].mean().max()
            logger.info(f"  [{target_col}] best: {best_m} (AUC={best_v:.3f})")
        else:
            best_m = df_cv.groupby("model")["rmse"].mean().idxmin()
            best_rmse = df_cv.groupby("model")["rmse"].mean().min()
            best_r2 = df_cv[df_cv["model"] == best_m]["r2"].mean()
            logger.info(
                f"  [{target_col}] best: {best_m} (RMSE={best_rmse:.3f}, R²={best_r2:.3f})"
            )


if __name__ == "__main__":
    main()

