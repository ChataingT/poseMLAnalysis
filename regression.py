"""
Nested cross-validated regression: predict ADOS-2 total score.

Six models:
  dt    DecisionTreeRegressor(max_depth=4)        baseline
  ridge ElasticNet                                linear baseline
  svr   SVR(kernel='rbf')                         classic
  rf    RandomForestRegressor(n_estimators=500)   ensemble
  xgb   XGBRegressor                             gradient boosting
  lgbm  LGBMRegressor                            fast gradient boosting

CV scheme (ADOS quintile stratification):
  Outer: StratifiedKFold(5) on ados_quintile
    └─ Inner: StratifiedKFold(3) on ados_quintile
         └─ RandomizedSearchCV(n_iter=50, scoring='neg_root_mean_squared_error')

Subjects with missing ADOS are excluded.

Metrics: RMSE (primary), MAE, R², Spearman ρ, Pearson r.
Model comparison: Wilcoxon signed-rank on outer-fold RMSE vectors (Bonferroni corrected).

Outputs (under output_dir/regression/):
  cv_results_all_models.csv
  model_comparison.csv
  predicted_vs_actual_{model}.png    out-of-fold predicted vs true ADOS
  residuals_{model}.png              residual distribution
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
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from .preprocessing import build_preprocessing_pipeline

logger = logging.getLogger(__name__)

DPI = 150
PALETTE_DIAGNOSIS = {"ASD": "#E74C3C", "TD": "#2E86AB"}
PALETTE_MODELS = {
    "dt":    "#95A5A6",
    "ridge": "#3498DB",
    "svr":   "#9B59B6",
    "rf":    "#27AE60",
    "xgb":   "#E74C3C",
    "lgbm":  "#F39C12",
}


# ─────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────

def _build_models(use_gpu: bool = False, n_jobs: int = -1, random_state: int = 42,
                  model_filter: list[str] | None = None):
    """Return dict of (model_id → (estimator, param_distributions)).

    Args:
        model_filter: If given, only models whose id is in this list are returned.
                      None (default) returns all available models.
    """
    try:
        from xgboost import XGBRegressor
        xgb_device = "cuda" if use_gpu else "cpu"
        xgb_est = XGBRegressor(
            device=xgb_device, random_state=random_state, nthread=max(1, n_jobs),
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
        from lightgbm import LGBMRegressor
        lgbm_device = "gpu" if use_gpu else "cpu"
        lgbm_est = LGBMRegressor(
            device=lgbm_device, random_state=random_state, n_jobs=n_jobs, verbose=-1,
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
            DecisionTreeRegressor(random_state=random_state),
            {
                "model__max_depth": [2, 3, 4, 5, 6],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__criterion": ["squared_error", "absolute_error"],
            },
        ),
        "ridge": (
            ElasticNet(max_iter=5000, random_state=random_state),
            {
                "model__alpha": np.logspace(-3, 3, 20),
                "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
        ),
        "svr": (
            SVR(kernel="rbf"),
            {
                "model__C": np.logspace(-2, 3, 20),
                "model__gamma": ["scale", "auto"] + list(np.logspace(-4, 0, 10)),
                "model__epsilon": [0.01, 0.1, 0.5, 1.0],
            },
        ),
        "rf": (
            RandomForestRegressor(n_estimators=500, n_jobs=n_jobs, random_state=random_state),
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
                f"No valid regression models after filtering. "
                f"Requested: {model_filter}"
            )

    return models


# ─────────────────────────────────────────────────────────────
# ADOS stratification
# ─────────────────────────────────────────────────────────────

def build_ados_strat(ados: pd.Series, n_bins: int = 5) -> np.ndarray:
    """
    Bin continuous scores into quintiles for stratified CV.
    Returns integer bin labels array aligned to ados.index.
    """
    try:
        bins = pd.qcut(ados, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(ados, bins=n_bins, labels=False)
    return bins.fillna(-1).astype(int).values


# Alias for backward compatibility
build_reg_strat = build_ados_strat


# ─────────────────────────────────────────────────────────────
# One outer fold
# ─────────────────────────────────────────────────────────────

def _run_one_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_defs: dict,
    n_inner: int,
    n_iter: int,
    ados_strat: np.ndarray,
    corr_threshold: float,
    random_state: int,
    n_jobs_inner: int,
    fixed_params: dict | None = None,
) -> list[dict]:
    """Run all models for one outer fold. Returns list of result dicts.

    Args:
        fixed_params: Optional mapping of model_id → {param: value}.
                      When present for a model, those hyperparameters are applied
                      directly and inner-CV search is skipped for that model.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    strat_train = ados_strat[train_idx]

    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=random_state)
    fixed_params = fixed_params or {}
    fold_results = []

    for model_id, (estimator, param_dist) in model_defs.items():
        import copy
        from sklearn.pipeline import Pipeline
        est = copy.deepcopy(estimator)
        preproc = build_preprocessing_pipeline(corr_threshold=corr_threshold)
        pipe = Pipeline(list(preproc.steps) + [("model", est)])

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
                n_iter=n_iter, scoring="neg_root_mean_squared_error", cv=inner_cv,
                n_jobs=n_jobs_inner, random_state=random_state,
                refit=True, error_score=np.nan,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params_str = str(search.best_params_)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = best.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spearman_r, spearman_p = spearmanr(y_test, y_pred)
            pearson_r, pearson_p = pearsonr(y_test, y_pred)

        fold_results.append({
            "fold": fold_idx,
            "model": model_id,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "best_params": best_params_str,
            "y_pred": y_pred.tolist(),
            "y_test": y_test.tolist(),
            "test_indices": test_idx.tolist(),  # row indices in original X / df_meta
        })
        logger.debug(
            f"  Fold {fold_idx} | {model_id:5s}: "
            f"RMSE={rmse:.3f}, R²={r2:.3f}, ρ={spearman_r:.3f}"
        )

    return fold_results


# ─────────────────────────────────────────────────────────────
# Main regression runner
# ─────────────────────────────────────────────────────────────

def run_regression(
    X: np.ndarray,
    y: np.ndarray,
    df_meta: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    target_name: str = "target",
    n_outer: int = 5,
    n_inner: int = 3,
    n_iter: int = 50,
    corr_threshold: float = 0.95,
    use_gpu: bool = False,
    n_jobs: int = 4,
    random_state: int = 42,
    model_filter: list[str] | None = None,
    fixed_params: dict | None = None,
) -> pd.DataFrame:
    """
    Run nested cross-validated regression for all (or a subset of) models.

    Args:
        X: Feature matrix (n_subjects × n_features), raw (preprocessing inside CV).
        y: Continuous target score array.
        df_meta: Clinical metadata (same row order as X).
        feature_names: Feature column names.
        output_dir: Root output directory.
        target_name: Name of the target column; used as output sub-directory name.
        n_outer / n_inner: Number of folds.
        n_iter: RandomizedSearchCV iterations.
        corr_threshold: Correlation filter threshold.
        use_gpu: Enable GPU for XGB/LGBM.
        n_jobs: Parallel outer fold jobs.
        random_state: Random seed.
        model_filter: List of model IDs to run. None = all available models (exploratory).
        fixed_params: Dict mapping model_id → {param: value}. Models listed here use
                      fixed hyperparameters instead of inner-CV random search.

    Returns:
        DataFrame of per-fold per-model scores.
    """

    logger.info("Starting regression analysis with nested CV of {} outer folds and {} inner folds".format(n_outer, n_inner))
    out = output_dir / "regression" / target_name
    out.mkdir(parents=True, exist_ok=True)

    # Stratification: bin y directly into quintiles
    y_series = pd.Series(y)
    ados_strat = build_ados_strat(y_series)
    diag_series = df_meta.get("diagnosis", pd.Series("unknown", index=df_meta.index))

    # Always use CPU for XGB/LGBM during CV (same rationale as classification).
    model_defs = _build_models(use_gpu=False, n_jobs=1, random_state=random_state,
                               model_filter=model_filter)
    fixed_params = fixed_params or {}
    logger.info(f"Regression: {len(model_defs)} models, {n_outer}×{n_inner} nested CV")
    if fixed_params:
        logger.info(f"  Fixed hyperparams (no inner CV) for: {sorted(fixed_params)}")

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    splits = list(outer_cv.split(X, ados_strat))

    all_fold_results = Parallel(n_jobs=min(n_jobs, n_outer), backend="loky")(
        delayed(_run_one_fold)(
            fold_idx=i,
            train_idx=train,
            test_idx=test,
            X=X,
            y=y,
            model_defs=model_defs,
            n_inner=n_inner,
            n_iter=n_iter,
            ados_strat=ados_strat,
            corr_threshold=corr_threshold,
            random_state=random_state,
            n_jobs_inner=-1,
            fixed_params=fixed_params,
        )
        for i, (train, test) in enumerate(splits)
    )

    # ── Flatten and extract raw predictions ──────────────────
    uuids = df_meta["uuid"].values if "uuid" in df_meta.columns else np.arange(len(df_meta))
    diag_vals = df_meta["diagnosis"].values if "diagnosis" in df_meta.columns else None
    rows = []
    pred_rows = []
    raw_preds = {m: {"y_pred": [], "y_test": []} for m in model_defs}

    for fold_res in all_fold_results:
        for r in fold_res:
            model_id = r["model"]
            test_idx_fold = np.array(r["test_indices"])
            y_pred_fold = np.array(r["y_pred"])
            y_test_fold = np.array(r["y_test"])

            raw_preds[model_id]["y_pred"].extend(y_pred_fold.tolist())
            raw_preds[model_id]["y_test"].extend(y_test_fold.tolist())

            # Per-subject prediction rows
            for idx, yt, yp in zip(test_idx_fold, y_test_fold, y_pred_fold):
                row_dict = {
                    "fold": r["fold"],
                    "model": model_id,
                    "uuid": uuids[idx],
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "residual": float(yp - yt),
                }
                if diag_vals is not None:
                    row_dict["diagnosis"] = diag_vals[idx]
                pred_rows.append(row_dict)

            rows.append({k: v for k, v in r.items()
                         if k not in ("y_pred", "y_test", "test_indices")})

    # ── Save: per-subject predictions ────────────────────────
    pd.DataFrame(pred_rows).to_csv(out / "predictions_per_subject.csv", index=False)
    logger.info("  Saved data: predictions_per_subject.csv")

    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(out / "cv_results_all_models.csv", index=False)
    logger.info(f"CV results: {len(df_cv)} rows → {out / 'cv_results_all_models.csv'}")

    # ── Model comparison ──────────────────────────────────────
    df_cmp = _model_comparison(df_cv, metric="rmse")
    df_cmp.to_csv(out / "model_comparison.csv", index=False)

    # ── Figures ───────────────────────────────────────────────
    for model_id, preds in raw_preds.items():
        y_pred_all = np.array(preds["y_pred"])
        y_test_all = np.array(preds["y_test"])
        _plot_pred_vs_actual(y_test_all, y_pred_all, model_id, out, target_name=target_name)
        _plot_residuals(y_test_all, y_pred_all, model_id, out, target_name=target_name)

    summary = df_cv.groupby("model")[["rmse", "r2", "spearman_r"]].mean().reset_index()
    logger.info("\n── Regression summary ──\n" + summary.to_string(index=False))

    return df_cv


# ─────────────────────────────────────────────────────────────
# Model comparison
# ─────────────────────────────────────────────────────────────

def _model_comparison(df_cv: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    models = df_cv["model"].unique()
    vals_by_model = {m: df_cv[df_cv["model"] == m][metric].values for m in models}

    pairs = list(combinations(models, 2))
    n_pairs = len(pairs)
    pvalues = {}

    for m1, m2 in pairs:
        v1, v2 = vals_by_model[m1], vals_by_model[m2]
        if len(v1) < 2 or np.allclose(v1, v2):
            p = 1.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p = wilcoxon(v1, v2, alternative="two-sided")
        pvalues[(m1, m2)] = min(p * n_pairs, 1.0)

    rows = []
    asc = metric == "rmse"  # lower is better for RMSE
    for m in models:
        vals = vals_by_model[m]
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

    return pd.DataFrame(rows).sort_values(f"{metric}_mean", ascending=asc)


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────

def _plot_pred_vs_actual(y_true, y_pred, model_id, out, target_name: str = "target"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.65, s=40, color=PALETTE_MODELS.get(model_id, "steelblue"),
               edgecolors="white", linewidths=0.4)
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "k--", lw=1.2, alpha=0.6, label="Ideal")

    # Regression line
    try:
        m, b = np.polyfit(y_true, y_pred, 1)
        xs = np.linspace(lims[0], lims[1], 50)
        ax.plot(xs, m * xs + b, "r-", lw=1.5, alpha=0.7, label=f"Fit (slope={m:.2f})")
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = pearsonr(y_true, y_pred)
        rho, _ = spearmanr(y_true, y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    ax.set_xlabel(f"True {target_name}", fontsize=11)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=11)
    ax.set_title(
        f"{model_id.upper()} — Predicted vs Actual\n"
        f"RMSE={rmse:.2f}, R²={r2:.3f}, ρ={rho:.3f}, r={r:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"predicted_vs_actual_{model_id}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: predicted_vs_actual_{model_id}.png")


def _plot_residuals(y_true, y_pred, model_id, out, target_name: str = "target"):
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Residual distribution
    ax = axes[0]
    ax.hist(residuals, bins=20, color=PALETTE_MODELS.get(model_id, "steelblue"),
            edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", lw=1.2)
    ax.set_xlabel("Residual (Predicted − True)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"{model_id.upper()} — Residual Distribution", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # Residual vs predicted
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.65, s=40,
               color=PALETTE_MODELS.get(model_id, "steelblue"),
               edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="black", linestyle="--", lw=1.2)
    ax.set_xlabel(f"Predicted {target_name}", fontsize=10)
    ax.set_ylabel("Residual", fontsize=10)
    ax.set_title(f"{model_id.upper()} — Residuals vs Predicted", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / f"residuals_{model_id}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: residuals_{model_id}.png")
