"""
Nested cross-validated binary classification: ASD vs TD.

Six models:
  dt    DecisionTreeClassifier(max_depth=4)          baseline, interpretable
  lr    LogisticRegression(elasticnet)               linear baseline
  svm   SVC(kernel='rbf', probability=True)          classic small-sample
  rf    RandomForestClassifier(n_estimators=500)     ensemble
  xgb   XGBClassifier                               gradient boosting
  lgbm  LGBMClassifier                              fast gradient boosting

CV scheme (joint stratification on diagnosis × ADOS tertile):
  Outer: StratifiedKFold(5, shuffle=True, random_state=42)
    └─ Inner: StratifiedKFold(3)
         └─ RandomizedSearchCV(n_iter=50, scoring='roc_auc')

SMOTE is applied inside each inner training fold.
Feature preprocessing is fit inside each inner training fold.

Metrics: AUC-ROC (primary), balanced accuracy, F1-macro, sensitivity, specificity.
Model comparison: Wilcoxon signed-rank on outer-fold AUC vectors (Bonferroni corrected).

Outputs (under output_dir/classification/):
  cv_results_all_models.csv    per-fold per-model scores
  model_comparison.csv         mean ± std + pairwise Wilcoxon p-values
  roc_curves.png               all models overlaid (mean ± std band)
  confusion_matrices.png       aggregated across folds, one per model
  learning_curves.png          RF + XGB train vs val AUC vs training size
"""

from __future__ import annotations

import logging
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from .preprocessing import build_preprocessing_pipeline

logger = logging.getLogger(__name__)

DPI = 150
PALETTE_MODELS = {
    "dt":   "#95A5A6",
    "lr":   "#3498DB",
    "svm":  "#9B59B6",
    "rf":   "#27AE60",
    "xgb":  "#E74C3C",
    "lgbm": "#F39C12",
}


# ─────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────

def _build_models(use_gpu: bool = False, n_jobs: int = -1, random_state: int = 42,
                  model_filter: list[str] | None = None, n_classes: int = 2):
    """Return dict of (model_id → (estimator, param_distributions)).

    Args:
        model_filter: If given, only models whose id is in this list are returned.
                      None (default) returns all available models.
    """
    try:
        from xgboost import XGBClassifier
        xgb_device = "cuda" if use_gpu else "cpu"
        xgb_est = XGBClassifier(
            device=xgb_device, eval_metric="logloss",
            random_state=random_state, nthread=max(1, n_jobs),
        )
        xgb_params = {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
            "model__gamma": [0, 0.1, 0.3],
        }
    except ImportError:
        logger.warning("xgboost not available; skipping XGB model")
        xgb_est = None
        xgb_params = None

    try:
        from lightgbm import LGBMClassifier
        lgbm_device = "gpu" if use_gpu else "cpu"
        # Binary: use is_unbalance (LightGBM doc recommends against class_weight for binary).
        # Multi-class: use class_weight='balanced' (is_unbalance is binary-only).
        if n_classes == 2:
            lgbm_est = LGBMClassifier(
                device=lgbm_device,
                is_unbalance=True,
                random_state=random_state, n_jobs=n_jobs, verbose=-1,
            )
        else:
            lgbm_est = LGBMClassifier(
                device=lgbm_device,
                class_weight="balanced",
                random_state=random_state, n_jobs=n_jobs, verbose=-1,
            )
        lgbm_params = {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [3, 5, 7, -1],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__num_leaves": [15, 31, 63],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_samples": [5, 10, 20],
        }
    except ImportError:
        logger.warning("lightgbm not available; skipping LGBM model")
        lgbm_est = None
        lgbm_params = None

    models = {
        "dt": (
            DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
            {
                "model__max_depth": [2, 3, 4, 5, 6],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "lr": (
            LogisticRegression(
                penalty="elasticnet", solver="saga", class_weight="balanced",
                max_iter=2000, random_state=random_state,
            ),
            {
                "model__C": np.logspace(-3, 2, 20),
                "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        ),
        "svm": (
            SVC(kernel="rbf", class_weight="balanced", probability=True,
                random_state=random_state),
            {
                "model__C": np.logspace(-2, 3, 20),
                "model__gamma": ["scale", "auto"] + list(np.logspace(-4, 0, 10)),
            },
        ),
        "rf": (
            RandomForestClassifier(
                n_estimators=500, class_weight="balanced",
                n_jobs=n_jobs, random_state=random_state,
            ),
            {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
            },
        ),
    }

    if xgb_est is not None:
        models["xgb"] = (xgb_est, xgb_params)
    if lgbm_est is not None:
        models["lgbm"] = (lgbm_est, lgbm_params)

    if model_filter is not None:
        missing = set(model_filter) - set(models)
        if missing:
            logger.warning(
                f"Config requested model(s) not available: {sorted(missing)}. "
                "They will be skipped."
            )
        models = {k: v for k, v in models.items() if k in model_filter}
        if not models:
            raise ValueError(
                f"No valid classification models after filtering. "
                f"Requested: {model_filter}"
            )

    return models


# ─────────────────────────────────────────────────────────────
# Stratification label
# ─────────────────────────────────────────────────────────────

def build_strat_label(df: pd.DataFrame) -> pd.Series:
    """
    Build combined stratification label: {diagnosis}_{ados_tertile}.
    Subjects missing ADOS get label {diagnosis}_noados.
    """
    ados = pd.to_numeric(df.get("ADOS_2_TOTAL", pd.Series(np.nan, index=df.index)), errors="coerce")
    diag = df.get("diagnosis", pd.Series("unknown", index=df.index)).astype(str)

    has_ados = ados.notna()
    try:
        ados_bins = pd.qcut(ados[has_ados], q=3, labels=["low", "med", "high"], duplicates="drop")
    except ValueError:
        ados_bins = pd.cut(ados[has_ados], bins=3, labels=["low", "med", "high"])

    strat = pd.Series("noados", index=df.index)
    strat[has_ados] = ados_bins.astype(str)
    strat = diag + "_" + strat
    return strat


def build_strat_label_from_y(y: np.ndarray) -> np.ndarray:
    """Build stratification array directly from integer-encoded class labels.

    For classification targets where diagnosis/ADOS meta is unavailable,
    we simply stratify on the label values themselves.
    """
    return y.astype(int)


# ─────────────────────────────────────────────────────────────
# SMOTE helper
# ─────────────────────────────────────────────────────────────

def _make_pipeline_with_smote(estimator, preprocessing_pipeline,
                               random_state=42, use_smote=True, k_neighbors=5):
    """Build a pipeline with optional SMOTE oversampling + preprocessing + model.

    Args:
        use_smote: When True (default) include SMOTE in an imbalanced-learn Pipeline.
                   Set to False to use a plain sklearn Pipeline (no oversampling).
    """
    from sklearn.pipeline import Pipeline as SkPipeline
    base_steps = list(preprocessing_pipeline.steps) + [("model", estimator)]

    if not use_smote:
        return SkPipeline(base_steps)

    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        steps = list(preprocessing_pipeline.steps) + [
            ("smote", smote),
            ("model", estimator),
        ]
        return ImbPipeline(steps)
    except ImportError:
        logger.warning("imbalanced-learn not available; SMOTE disabled")
        return SkPipeline(base_steps)


# ─────────────────────────────────────────────────────────────
# One outer fold
# ─────────────────────────────────────────────────────────────

def _run_one_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    strat: np.ndarray,
    model_defs: dict,
    n_inner: int,
    n_iter: int,
    corr_threshold: float,
    random_state: int,
    n_jobs_inner: int,
    fixed_params: dict | None = None,
    use_smote: bool = True,
    n_classes: int = 2,
) -> list[dict]:
    """Run all models for one outer fold. Returns list of result dicts.

    Args:
        fixed_params: Optional mapping of model_id → {param: value}.
                      When present for a model, those hyperparameters are applied
                      directly and inner-CV search is skipped for that model.
        use_smote: When False, SMOTE oversampling is disabled (useful when class
                   sizes are too small for the default k_neighbors=5).
        n_classes: Number of target classes (2 = binary, >2 = multi-class).
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    strat_train = strat[train_idx]

    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    fixed_params = fixed_params or {}
    fold_results = []

    # Adaptive SMOTE: k_neighbors must be < smallest minority class size in this fold
    min_class_count = int(np.bincount(y_train.astype(int)).min())
    smote_k = max(1, min(5, min_class_count - 1))
    if smote_k < 1:
        use_smote = False

    # Choose inner-CV scoring based on number of classes
    if n_classes > 2:
        inner_scoring = make_scorer(
            roc_auc_score, needs_proba=True, multi_class="ovr", average="macro"
        )
    else:
        inner_scoring = "roc_auc"

    for model_id, (estimator, param_dist) in model_defs.items():
        import copy
        est = copy.deepcopy(estimator)
        preproc = build_preprocessing_pipeline(corr_threshold=corr_threshold)
        pipe = _make_pipeline_with_smote(est, preproc, random_state=random_state,
                                         use_smote=use_smote and model_id != "lgbm",
                                         k_neighbors=smote_k)

        if model_id in fixed_params:
            # Config mode: apply fixed hyperparameters, skip inner CV
            for param, value in fixed_params[model_id].items():
                pipe.set_params(**{f"model__{param}": value})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_train, y_train)
            best = pipe
            best_params_str = str({f"model__{k}": v
                                    for k, v in fixed_params[model_id].items()})
        else:
            search = RandomizedSearchCV(
                pipe, param_dist,
                n_iter=n_iter, scoring=inner_scoring, cv=inner_cv,
                n_jobs=n_jobs_inner, random_state=random_state,
                refit=True, error_score=np.nan,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train, **_smote_fit_params(pipe))
            best = search.best_estimator_
            best_params_str = str(search.best_params_)


        # Predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_prob_full = best.predict_proba(X_test)  # shape (n_test, n_classes)
            y_pred = best.predict(X_test)

        # AUC: binary uses 1-D prob of positive class; multi-class uses OvR macro
        if n_classes == 2:
            y_prob = y_prob_full[:, 1]  # 1-D for binary (backward-compatible)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc = np.nan
        else:
            y_prob = y_prob_full  # 2-D for multi-class
            try:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            except Exception:
                auc = np.nan

        bacc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # sensitivity / specificity only meaningful for binary
        if n_classes == 2:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        else:
            sensitivity = np.nan
            specificity = np.nan

        fold_results.append({
            "fold": fold_idx,
            "model": model_id,
            "n_classes": n_classes,
            "auc_roc": auc,
            "balanced_acc": bacc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "best_params": best_params_str,
            # y_prob: 1-D list for binary, 2-D list for multi-class
            "y_prob": y_prob.tolist() if isinstance(y_prob, np.ndarray) else y_prob_full.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "test_indices": test_idx.tolist(),  # row indices in original X / df_meta
        })
        logger.debug(
            f"  Fold {fold_idx} | {model_id:5s}: "
            f"AUC={auc:.3f}, BAcc={bacc:.3f}"
        )

    return fold_results


def _smote_fit_params(pipe):
    """Return fit params to pass sample weights if SMOTE step present."""
    # imblearn Pipeline requires no special params; sklearn Pipeline ignores extra steps
    return {}


# ─────────────────────────────────────────────────────────────
# Main classification runner
# ─────────────────────────────────────────────────────────────

def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    df_meta: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    target_name: str = "default",
    n_classes: int = 2,
    label_names: list[str] | None = None,
    y_strat: np.ndarray | None = None,
    n_outer: int = 5,
    n_inner: int = 3,
    n_iter: int = 50,
    corr_threshold: float = 0.95,
    use_gpu: bool = False,
    n_jobs: int = 4,
    random_state: int = 42,
    model_filter: list[str] | None = None,
    fixed_params: dict | None = None,
    use_smote: bool = True,
) -> pd.DataFrame:
    """
    Run nested cross-validated classification for all (or a subset of) models.

    Args:
        X: Feature matrix (n_subjects × n_features), already imputed/scaled.
           Pass the RAW (un-preprocessed) features; preprocessing is done inside CV.
        y: Integer-encoded label array (0, 1, …, n_classes-1).
        df_meta: Clinical metadata (same row order as X), used for stratification
                 fallback (diagnosis × ADOS tertile) when y_strat is None.
        feature_names: Feature column names.
        output_dir: Root output directory.
        target_name: Name of the target column; used as output sub-directory name.
        n_classes: Number of distinct classes in y.
        label_names: Human-readable class names aligned to encoded integers.
        y_strat: Pre-built stratification array (same length as y). When None,
                 falls back to build_strat_label(df_meta).
        n_outer / n_inner: Number of folds.
        n_iter: RandomizedSearchCV iterations.
        corr_threshold: Correlation filter threshold (passed to preprocessing pipeline).
        use_gpu: Enable GPU for XGB/LGBM.
        n_jobs: Parallel jobs for outer fold loop.
        random_state: Random seed.
        model_filter: List of model IDs to run. None = all available models (exploratory).
        fixed_params: Dict mapping model_id → {param: value}. Models listed here use
                      fixed hyperparameters instead of inner-CV random search.
        use_smote: When False, SMOTE oversampling is disabled (set smote: false in config).

    Returns:
        DataFrame of per-fold per-model scores.
    """
    logger.info("Starting classification analysis with nested CV of {} outer folds and {} inner folds".format(n_outer, n_inner))

    out = output_dir / "classification" / target_name
    out.mkdir(parents=True, exist_ok=True)

    # Stratification: use provided y_strat, else fall back to diagnosis×ADOS label
    if y_strat is not None:
        strat_arr = y_strat
    else:
        strat_label = build_strat_label(df_meta)
        strat_arr = strat_label.values

    # Always use CPU for XGB/LGBM during CV:  with 119 subjects the GPU
    # data-transfer overhead far exceeds compute time, and running 5 concurrent
    # loky forks each with a LightGBM GPU instance exhausts host memory before
    # the first tree is trained (LightGBM allocates large host-side bin buffers
    # per Booster regardless of dataset size when device='gpu').
    model_defs = _build_models(use_gpu=False, n_jobs=1, random_state=random_state,
                               model_filter=model_filter, n_classes=n_classes)
    fixed_params = fixed_params or {}
    logger.info(f"Classification: {len(model_defs)} models, {n_outer}×{n_inner} nested CV")
    if fixed_params:
        logger.info(f"  Fixed hyperparams (no inner CV) for: {sorted(fixed_params)}")

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    splits = list(outer_cv.split(X, strat_arr))

    if not use_smote:
        logger.info("  SMOTE disabled (use_smote=False)")

    # Run folds in parallel (n_jobs outer folds)
    all_fold_results = Parallel(n_jobs=min(n_jobs, n_outer), backend="loky")(
        delayed(_run_one_fold)(
            fold_idx=i,
            train_idx=train,
            test_idx=test,
            X=X,
            y=y,
            strat=strat_arr,
            model_defs=model_defs,
            n_inner=n_inner,
            n_iter=n_iter,
            corr_threshold=corr_threshold,
            random_state=random_state,
            n_jobs_inner=-1,
            fixed_params=fixed_params,
            use_smote=use_smote,
            n_classes=n_classes,
        )
        for i, (train, test) in enumerate(splits)
    )

    # Flatten results
    rows = []
    for fold_res in all_fold_results:
        rows.extend(fold_res)

    # ── Save: per-subject predictions ────────────────────────
    # Link each prediction to the original subject uuid via test_indices.
    uuids = df_meta["uuid"].values if "uuid" in df_meta.columns else np.arange(len(df_meta))
    pred_rows = []
    roc_data_rows = []
    raw_preds = {r["model"]: [] for r in rows}

    for r in rows:
        model_id = r["model"]
        fold_idx = r["fold"]
        test_idx_fold = np.array(r["test_indices"])
        y_test_fold = np.array(r["y_test"])
        y_prob_raw = r["y_prob"]
        y_pred_fold = np.array(r["y_pred"])
        nc = r.get("n_classes", n_classes)

        # y_prob: 1-D for binary, 2-D for multi-class
        y_prob_fold = np.array(y_prob_raw)
        # derive scalar probability for the per-subject CSV
        if nc == 2:
            y_prob_pos = y_prob_fold  # probability of class 1
        else:
            y_prob_pos = y_prob_fold.max(axis=1) if y_prob_fold.ndim == 2 else y_prob_fold

        raw_preds[model_id].append({
            "y_test": r["y_test"],
            "y_prob": y_prob_raw,
            "y_pred": r["y_pred"],
            "n_classes": nc,
        })

        # Per-subject prediction rows
        for idx, yt, yp, ypr in zip(test_idx_fold, y_test_fold, y_pred_fold, y_prob_pos):
            pred_rows.append({
                "fold": fold_idx,
                "model": model_id,
                "uuid": uuids[idx],
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_prob_pos": float(ypr),
            })

        # ROC curve data per fold (binary only)
        if nc == 2 and len(np.unique(y_test_fold)) > 1:
            fpr_arr, tpr_arr, thr_arr = roc_curve(y_test_fold, y_prob_fold)
            for f, t, thr in zip(fpr_arr, tpr_arr, thr_arr):
                roc_data_rows.append({
                    "fold": fold_idx, "model": model_id,
                    "fpr": float(f), "tpr": float(t), "threshold": float(thr),
                })

    pd.DataFrame(pred_rows).to_csv(out / "predictions_per_subject.csv", index=False)
    logger.info("  Saved data: predictions_per_subject.csv")

    pd.DataFrame(roc_data_rows).to_csv(out / "roc_curve_data.csv", index=False)
    logger.info("  Saved data: roc_curve_data.csv")

    # ── Summary CV results (drop raw arrays) ─────────────────
    rows_summary = [
        {k: v for k, v in r.items() if k not in ("y_test", "y_prob", "y_pred", "test_indices")}
        for r in rows
    ]
    df_cv = pd.DataFrame(rows_summary)
    df_cv.to_csv(out / "cv_results_all_models.csv", index=False)
    logger.info(f"CV results: {len(df_cv)} rows → {out / 'cv_results_all_models.csv'}")

    # ── Model comparison (Wilcoxon signed-rank) ───────────────
    df_cmp = _model_comparison(df_cv, metric="auc_roc")
    df_cmp.to_csv(out / "model_comparison.csv", index=False)

    # ── Figures ───────────────────────────────────────────────
    _plot_roc_curves(raw_preds, out, n_classes=n_classes)
    _plot_confusion_matrices(raw_preds, y, list(model_defs.keys()), out,
                             n_classes=n_classes, label_names=label_names)
    _plot_learning_curves(X, y, strat_arr, model_defs, out,
                          corr_threshold=corr_threshold, random_state=random_state)

    # Print summary
    summary = df_cv.groupby("model")["auc_roc"].agg(["mean", "std"]).reset_index()
    summary.columns = ["model", "auc_mean", "auc_std"]
    summary = summary.sort_values("auc_mean", ascending=False)
    logger.info("\n── Classification AUC-ROC summary ──\n" + summary.to_string(index=False))

    return df_cv


# ─────────────────────────────────────────────────────────────
# Model comparison
# ─────────────────────────────────────────────────────────────

def _model_comparison(df_cv: pd.DataFrame, metric: str = "auc_roc") -> pd.DataFrame:
    """Compute mean±std and pairwise Wilcoxon signed-rank p-values (Bonferroni)."""
    models = df_cv["model"].unique()
    auc_by_model = {m: df_cv[df_cv["model"] == m][metric].values for m in models}

    rows = []
    pairs = list(combinations(models, 2))
    n_pairs = len(pairs)
    pvalues = {}

    for m1, m2 in pairs:
        v1, v2 = auc_by_model[m1], auc_by_model[m2]
        if len(v1) < 2 or np.allclose(v1, v2):
            p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = wilcoxon(v1, v2, alternative="two-sided")
        p_bonf = min(p * n_pairs, 1.0)
        pvalues[(m1, m2)] = p_bonf

    for m in models:
        vals = auc_by_model[m]
        row = {
            "model": m,
            f"{metric}_mean": float(np.mean(vals)),
            f"{metric}_std": float(np.std(vals)),
        }
        for m2 in models:
            if m == m2:
                continue
            key = (m, m2) if (m, m2) in pvalues else (m2, m)
            row[f"p_vs_{m2}_bonf"] = pvalues.get(key, np.nan)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(f"{metric}_mean", ascending=False)


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────

def _plot_roc_curves(raw_preds: dict, out: Path, n_classes: int = 2) -> None:
    """Plot mean ROC curve with ±1 std band for each model (binary only)."""
    if n_classes != 2:
        logger.info("  ROC curve skipped for multi-class target (n_classes=%d)", n_classes)
        return
    fig, ax = plt.subplots(figsize=(7, 6))

    base_fpr = np.linspace(0, 1, 101)

    for model_id, fold_data in raw_preds.items():
        tprs = []
        aucs = []
        for fd in fold_data:
            y_test = np.array(fd["y_test"])
            y_prob_raw = fd["y_prob"]
            y_prob = np.array(y_prob_raw)
            # For binary, y_prob is 1-D
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            if len(np.unique(y_test)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            interp_tpr = np.interp(base_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc_score(y_test, y_prob))
        if not tprs:
            continue

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        color = PALETTE_MODELS.get(model_id, "grey")
        ax.plot(base_fpr, mean_tpr, color=color, lw=2,
                label=f"{model_id.upper()} (AUC={mean_auc:.3f}±{std_auc:.3f})")
        ax.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=color, alpha=0.15)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves (Nested CV)", fontsize=12)
    ax.legend(fontsize=8, framealpha=0.8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: roc_curves.png")


def _plot_confusion_matrices(raw_preds: dict, y: np.ndarray, model_ids: list, out: Path,
                              n_classes: int = 2,
                              label_names: list[str] | None = None) -> None:
    """Aggregated confusion matrices across all outer folds, one per model."""
    class_labels = list(range(n_classes))
    tick_labels = label_names if label_names and len(label_names) == n_classes else [
        str(i) for i in class_labels
    ]
    n_models = len(model_ids)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for idx, model_id in enumerate(model_ids):
        ax = axes[idx // ncols][idx % ncols]
        fold_data = raw_preds.get(model_id, [])

        cm_total = np.zeros((n_classes, n_classes), dtype=int)
        for fd in fold_data:
            y_test = np.array(fd["y_test"])
            y_pred = np.array(fd["y_pred"])
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            cm_total += cm

        im = ax.imshow(cm_total, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(model_id.upper(), fontsize=11)
        tick_marks = list(range(n_classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=45 if n_classes > 4 else 0,
                           ha="right" if n_classes > 4 else "center")
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm_total[i, j]),
                        ha="center", va="center", fontsize=max(6, 13 - n_classes),
                        color="white" if cm_total[i, j] > cm_total.max() / 2 else "black")

    # Hide unused axes
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Aggregated Confusion Matrices (All Folds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: confusion_matrices.png")


def _plot_learning_curves(X, y, strat_arr, model_defs, out,
                          corr_threshold=0.95, random_state=42):
    """Learning curves (train vs val AUC vs n_training_samples) for RF and XGB."""
    lc_models = {k: v for k, v in model_defs.items() if k in ("rf", "xgb")}
    if not lc_models:
        return

    fig, axes = plt.subplots(1, len(lc_models), figsize=(6 * len(lc_models), 5), squeeze=False)

    train_sizes_rel = np.linspace(0.2, 1.0, 8)

    for idx, (model_id, (estimator, _)) in enumerate(lc_models.items()):
        ax = axes[0][idx]
        import copy
        est = copy.deepcopy(estimator)
        preproc = build_preprocessing_pipeline(corr_threshold=corr_threshold)

        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.over_sampling import SMOTE
            pipe = ImbPipeline(
                list(preproc.steps) + [("smote", SMOTE(random_state=random_state)), ("model", est)]
            )
        except ImportError:
            from sklearn.pipeline import Pipeline
            pipe = Pipeline(list(preproc.steps) + [("model", est)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_sizes, train_scores, val_scores = learning_curve(
                pipe, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=random_state),
                train_sizes=train_sizes_rel, scoring="roc_auc",
                n_jobs=-1, error_score=np.nan,
            )

        train_mean = np.nanmean(train_scores, axis=1)
        train_std  = np.nanstd(train_scores, axis=1)
        val_mean   = np.nanmean(val_scores, axis=1)
        val_std    = np.nanstd(val_scores, axis=1)

        color = PALETTE_MODELS.get(model_id, "grey")
        ax.plot(train_sizes, train_mean, "o-", color=color, label="Train AUC", lw=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        color=color, alpha=0.2)
        ax.plot(train_sizes, val_mean, "s--", color=color, label="Val AUC (CV)", lw=2, alpha=0.7)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        color=color, alpha=0.1)
        ax.axhline(0.5, color="grey", linestyle=":", lw=1)
        ax.set_xlabel("Training set size", fontsize=10)
        ax.set_ylabel("AUC-ROC", fontsize=10)
        ax.set_title(f"Learning Curve — {model_id.upper()}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 1.02)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: learning_curves.png")
