"""
SHAP explainability for the best-performing classification and regression models.

Models are retrained on the full dataset (all subjects) using the best
hyperparameters found during nested CV (determined by mean outer-fold score).

SHAP explainer selection:
  RF / XGB / LGBM → shap.TreeExplainer   (exact, fast)
  LR / Ridge      → shap.LinearExplainer
  SVM / SVR       → shap.KernelExplainer (background = kmeans(X, 50))

Global explanations:
  - Beeswarm plot (top 20 by mean |SHAP|)
  - Bar chart of mean |SHAP| for top 20 features
  - Dependence plots for top 5 features

Local explanations (classification only):
  - Waterfall plots for 4 representative subjects: TP, TN, FP, FN

Outputs under output_dir/explain/:
  shap_beeswarm_{task}_{model}.png
  shap_bar_{task}_{model}.png
  shap_dependence_{rank}_{feature}_{task}_{model}.png
  shap_waterfall_{case_type}_{task}_{model}.png
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DPI = 150
TOP_N_FEATURES = 20
TOP_N_DEPENDENCE = 5
PALETTE_DIAGNOSIS = {"ASD": "#E74C3C", "TD": "#2E86AB"}


# ─────────────────────────────────────────────────────────────
# Model retraining on full dataset
# ─────────────────────────────────────────────────────────────

def _get_best_model_id(df_cv: pd.DataFrame, metric: str, lower_is_better: bool = False) -> str:
    """Return the model_id with best mean metric across outer folds."""
    summary = df_cv.groupby("model")[metric].mean()
    return summary.idxmin() if lower_is_better else summary.idxmax()


def _retrain_best(
    df_cv: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    lower_is_better: bool,
    model_builder,
    corr_threshold: float,
    random_state: int,
    use_gpu: bool,
    input_feature_names: list[str] | None = None,
) -> tuple[str, object, np.ndarray, list[str]]:
    """
    Identify best model by mean outer-fold metric, retrain on full dataset,
    and return (model_id, fitted_pipeline, X_transformed, feature_names_out).

    Args:
        input_feature_names: Real names of the columns in X (before preprocessing).
            If provided, the returned feature_names_out will use them; otherwise
            falls back to generic names (f0, f1, …).
    """
    import copy
    from sklearn.pipeline import Pipeline

    best_model_id = _get_best_model_id(df_cv, metric, lower_is_better)
    logger.info(f"  Best model by {metric}: {best_model_id}")

    # Get best hyperparams from the most common best_params string across folds
    fold_rows = df_cv[df_cv["model"] == best_model_id]
    best_params_str = fold_rows["best_params"].mode()[0]
    try:
        import ast
        best_params_raw = ast.literal_eval(best_params_str)
    except Exception:
        best_params_raw = {}

    # Build model with best params (strip "model__" prefix)
    model_defs = model_builder(use_gpu=use_gpu, n_jobs=-1, random_state=random_state)
    estimator, _ = model_defs[best_model_id]
    est = copy.deepcopy(estimator)

    model_params = {k.replace("model__", ""): v for k, v in best_params_raw.items()
                    if k.startswith("model__")}
    try:
        est.set_params(**model_params)
    except Exception as e:
        logger.warning(f"  Could not set best params: {e}")

    from .preprocessing import build_preprocessing_pipeline, get_pipeline_feature_names
    preproc = build_preprocessing_pipeline(corr_threshold=corr_threshold)
    pipe = Pipeline(list(preproc.steps) + [("model", est)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)

    # Transform X through all steps except the final model
    X_transformed = _transform_without_model(pipe, X)

    # Recover real feature names via the fitted preprocessing masks.
    # get_pipeline_feature_names traces keep_mask_ through each filter step,
    # giving the correct metric names (e.g. "child_speed_centroid__raw__mean").
    # Without input_feature_names we fall back to generic indices.
    if input_feature_names is not None:
        feat_names = get_pipeline_feature_names(preproc, input_feature_names)
    else:
        feat_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        logger.warning(
            "  input_feature_names not provided to _retrain_best; "
            "SHAP output will use generic feature names (feature_0, feature_1, …)"
        )

    return best_model_id, pipe, X_transformed, feat_names


def _transform_without_model(pipe, X: np.ndarray) -> np.ndarray:
    """Apply all pipeline steps except the last (model) step."""
    X_t = X.copy()
    steps = list(pipe.named_steps.items())
    for name, step in steps[:-1]:  # skip 'model'
        if hasattr(step, "transform"):
            X_t = step.transform(X_t)
        elif hasattr(step, "fit_resample"):
            # SMOTE — skip at inference
            pass
    return X_t


# ─────────────────────────────────────────────────────────────
# SHAP computation
# ─────────────────────────────────────────────────────────────

def _get_shap_values(model_id: str, fitted_model, X_transformed: np.ndarray,
                     task: str, random_state: int = 42):
    """
    Compute SHAP values using the appropriate explainer.

    Returns:
        shap_values: ndarray of shape (n_samples, n_features)
                     For binary classification: SHAP values for the positive class (ASD).
        explainer: fitted SHAP explainer object.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap package is required. Install with: pip install shap")

    tree_models = {"rf", "xgb", "lgbm", "dt"}
    linear_models = {"lr", "ridge"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if model_id in tree_models:
            explainer = shap.TreeExplainer(fitted_model.named_steps["model"])
            sv = explainer.shap_values(X_transformed)
            # For classifiers, shap_values may be a list [class0, class1]
            if isinstance(sv, list) and len(sv) == 2:
                sv = sv[1]  # positive class (ASD)
            # For multi-output: take first output
            if sv.ndim == 3:
                sv = sv[:, :, 1]

        elif model_id in linear_models:
            background = shap.maskers.Independent(X_transformed, max_samples=100)
            explainer = shap.LinearExplainer(fitted_model.named_steps["model"], background)
            sv = explainer.shap_values(X_transformed)
            if isinstance(sv, list):
                sv = sv[1]

        else:
            # Kernel explainer (SVM, SVR)
            background = shap.kmeans(X_transformed, min(50, X_transformed.shape[0]))
            if task == "classification":
                def predict_fn(x):
                    return fitted_model.named_steps["model"].predict_proba(x)[:, 1]
            else:
                def predict_fn(x):
                    return fitted_model.named_steps["model"].predict(x)
            explainer = shap.KernelExplainer(predict_fn, background)
            sv = explainer.shap_values(X_transformed, nsamples=200)

    return np.array(sv), explainer


# ─────────────────────────────────────────────────────────────
# Global explanation figures
# ─────────────────────────────────────────────────────────────

def _plot_beeswarm(shap_values, X_transformed, feature_names, task, model_id, out):
    try:
        import shap
    except ImportError:
        return

    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"shap_beeswarm_{task}_{model_id}.png"

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:TOP_N_FEATURES]

    sv_top = shap_values[:, top_idx]
    X_top = X_transformed[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]

    # Shorten feature names for display
    short_names = [_shorten_name(n) for n in names_top]

    fig, ax = plt.subplots(figsize=(10, 0.4 * TOP_N_FEATURES + 2))

    # Manual beeswarm (matplotlib-based, no shap.plots dependency issues)
    _manual_beeswarm(ax, sv_top, X_top, short_names)

    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    ax.set_title(f"SHAP Beeswarm — {task.capitalize()} [{model_id.upper()}]", fontsize=11)
    ax.axvline(0, color="black", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {fname.name}")


def _manual_beeswarm(ax, shap_values, X_data, feature_names):
    """Simple beeswarm: features on y-axis, SHAP on x-axis, colored by feature value."""
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdBu_r
    n_feat = shap_values.shape[1]

    for j in range(n_feat):
        sv_j = shap_values[:, j]
        fv_j = X_data[:, j]

        # Normalize feature values for coloring
        fv_min, fv_max = np.nanmin(fv_j), np.nanmax(fv_j)
        if fv_max > fv_min:
            fv_norm = (fv_j - fv_min) / (fv_max - fv_min)
        else:
            fv_norm = np.full_like(fv_j, 0.5)

        # Jitter y-positions to create beeswarm effect
        y_base = n_feat - j - 1
        y_jitter = y_base + np.random.uniform(-0.3, 0.3, size=len(sv_j))

        colors = cmap(fv_norm)
        ax.scatter(sv_j, y_jitter, c=colors, s=12, alpha=0.7, linewidths=0)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(reversed(feature_names), fontsize=7)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("Feature value\n(low → high)", fontsize=7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])


def _plot_bar(shap_values, feature_names, task, model_id, out):
    """Bar chart of mean |SHAP| for top N features."""
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"shap_bar_{task}_{model_id}.png"

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:TOP_N_FEATURES]
    top_names = [_shorten_name(feature_names[i]) for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(9, 0.35 * TOP_N_FEATURES + 2))
    colors = [f"C{i % 10}" for i in range(len(top_idx))]
    ax.barh(range(len(top_idx)), top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(f"SHAP Feature Importance — {task.capitalize()} [{model_id.upper()}]", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {fname.name}")


def _plot_dependence(shap_values, X_transformed, feature_names, task, model_id, out):
    """Dependence plots for top-N features."""
    out.mkdir(parents=True, exist_ok=True)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:TOP_N_DEPENDENCE]

    for rank, feat_idx in enumerate(top_idx):
        sv = shap_values[:, feat_idx]
        fv = X_transformed[:, feat_idx]

        # Find best interaction feature (highest |corr| with SHAP values of this feature)
        correlations = np.array([
            abs(np.corrcoef(X_transformed[:, j], sv)[0, 1])
            if np.std(X_transformed[:, j]) > 0 else 0
            for j in range(X_transformed.shape[1])
        ])
        correlations[feat_idx] = 0  # exclude self
        interact_idx = int(np.argmax(correlations))
        interact_fv = X_transformed[:, interact_idx]

        # Color by interaction feature
        interact_min, interact_max = np.nanmin(interact_fv), np.nanmax(interact_fv)
        if interact_max > interact_min:
            interact_norm = (interact_fv - interact_min) / (interact_max - interact_min)
        else:
            interact_norm = np.full_like(interact_fv, 0.5)

        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(fv, sv, c=interact_norm, cmap="RdBu_r",
                        s=40, alpha=0.75, edgecolors="white", linewidths=0.3)
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label(f"Interact: {_shorten_name(feature_names[interact_idx])}", fontsize=7)

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(_shorten_name(feature_names[feat_idx]), fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(
            f"SHAP Dependence (rank {rank+1}) — {task.capitalize()} [{model_id.upper()}]\n"
            f"{_shorten_name(feature_names[feat_idx])}",
            fontsize=10,
        )
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        safe_name = feature_names[feat_idx].replace("/", "_").replace(" ", "_")[:40]
        fname = out / f"shap_dependence_{rank+1}_{safe_name}_{task}_{model_id}.png"
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {fname.name}")


# ─────────────────────────────────────────────────────────────
# Local explanations (classification waterfall)
# ─────────────────────────────────────────────────────────────

def _plot_waterfalls(shap_values, X_transformed, feature_names, y_true, y_prob,
                     model_id, out, n_top=15):
    """
    Waterfall plots for representative subjects: TP, TN, FP, FN.
    """
    out.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= 0.5).astype(int)

    cases = {
        "TP": np.where((y_true == 1) & (y_pred == 1))[0],
        "TN": np.where((y_true == 0) & (y_pred == 0))[0],
        "FP": np.where((y_true == 0) & (y_pred == 1))[0],
        "FN": np.where((y_true == 1) & (y_pred == 0))[0],
    }

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:n_top]
    top_names = [_shorten_name(feature_names[i]) for i in top_idx]

    for case_type, indices in cases.items():
        if len(indices) == 0:
            logger.debug(f"  No {case_type} examples found; skipping waterfall")
            continue

        # Pick the subject closest to the median probability in that group
        probs = y_prob[indices]
        pick_idx = indices[np.argmin(np.abs(probs - np.median(probs)))]

        sv_subj = shap_values[pick_idx, top_idx]
        fv_subj = X_transformed[pick_idx, top_idx]

        # Sort by absolute SHAP (ascending for bottom-up waterfall)
        order = np.argsort(np.abs(sv_subj))
        sv_ord = sv_subj[order]
        names_ord = [top_names[i] for i in order]

        # Compute cumulative sum for waterfall
        # Base value is mean model output ≈ mean(y_prob) but we just show relative
        fig, ax = plt.subplots(figsize=(9, 0.4 * n_top + 2))

        running = 0.0
        for k, (sv_k, name_k) in enumerate(zip(sv_ord, names_ord)):
            color = "#E74C3C" if sv_k > 0 else "#2E86AB"
            ax.barh(k, sv_k, left=running, color=color, alpha=0.85, height=0.6)
            ax.text(running + sv_k / 2, k, f"{sv_k:+.3f}", va="center", ha="center",
                    fontsize=6, color="white" if abs(sv_k) > 0.02 else "black")
            running += sv_k

        ax.set_yticks(range(n_top))
        ax.set_yticklabels(names_ord, fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Cumulative SHAP contribution", fontsize=10)
        ax.set_title(
            f"SHAP Waterfall — {case_type} | {model_id.upper()}\n"
            f"True={y_true[pick_idx]}, Predicted prob={y_prob[pick_idx]:.2f}",
            fontsize=10,
        )
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        fname = out / f"shap_waterfall_{case_type}_classification_{model_id}.png"
        fig.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved: {fname.name}")


# ─────────────────────────────────────────────────────────────
# Data saving
# ─────────────────────────────────────────────────────────────

def _save_shap_data(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list[str],
    df_meta: pd.DataFrame,
    task: str,
    model_id: str,
    out: Path,
) -> None:
    """
    Save SHAP values and corresponding feature values as CSV files so
    plots can be fully recreated without re-running the model.

    Files saved:
      shap_values_{task}_{model}.csv       — one row per subject, one col per feature
      shap_feature_values_{task}_{model}.csv — scaled feature values passed to SHAP

    Both files also include clinical columns (uuid, diagnosis, gender, age, ados)
    prepended as the first columns.
    """
    out.mkdir(parents=True, exist_ok=True)

    # Clinical prefix columns
    uuids = df_meta["uuid"].values if "uuid" in df_meta.columns else np.arange(len(df_meta))
    diag = df_meta["diagnosis"].values if "diagnosis" in df_meta.columns else [""] * len(df_meta)
    gender = df_meta["gender"].values if "gender" in df_meta.columns else [""] * len(df_meta)
    age = df_meta["Ados_2_Age"].values if "Ados_2_Age" in df_meta.columns else [np.nan] * len(df_meta)
    ados = df_meta["ADOS_2_TOTAL"].values if "ADOS_2_TOTAL" in df_meta.columns else [np.nan] * len(df_meta)

    prefix = pd.DataFrame({
        "uuid": uuids,
        "diagnosis": diag,
        "gender": gender,
        "Ados_2_Age": age,
        "ADOS_2_TOTAL": ados,
    })

    # SHAP values
    sv_df = pd.DataFrame(shap_values, columns=feature_names)
    pd.concat([prefix, sv_df], axis=1).to_csv(
        out / f"shap_values_{task}_{model_id}.csv", index=False
    )
    logger.info(f"  Saved data: shap_values_{task}_{model_id}.csv")

    # Feature values (scaled, after preprocessing)
    fv_df = pd.DataFrame(X_transformed, columns=feature_names)
    pd.concat([prefix, fv_df], axis=1).to_csv(
        out / f"shap_feature_values_{task}_{model_id}.csv", index=False
    )
    logger.info(f"  Saved data: shap_feature_values_{task}_{model_id}.csv")

    # Mean |SHAP| ranking
    mean_abs = np.abs(shap_values).mean(axis=0)
    rank_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    rank_df.insert(0, "rank", range(1, len(rank_df) + 1))
    rank_df.to_csv(out / f"shap_importance_{task}_{model_id}.csv", index=False)
    logger.info(f"  Saved data: shap_importance_{task}_{model_id}.csv")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _shorten_name(name: str, maxlen: int = 45) -> str:
    """Shorten long feature names for plot labels."""
    return name[:maxlen] + "…" if len(name) > maxlen else name


# ─────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────

def run_explainability(
    X: np.ndarray,
    y_clf: np.ndarray,
    y_reg: np.ndarray | None,
    df_meta: pd.DataFrame,
    feature_names: list[str],
    df_cv_clf: pd.DataFrame,
    df_cv_reg: pd.DataFrame | None,
    output_dir: Path,
    corr_threshold: float = 0.95,
    use_gpu: bool = False,
    random_state: int = 42,
    X_reg: np.ndarray | None = None,
    df_meta_reg: pd.DataFrame | None = None,
) -> None:
    """
    Run SHAP explainability for best classification and (optionally) regression models.

    Args:
        X: Raw feature matrix for classification (all subjects, preprocessing done inside).
        y_clf: Binary labels (0=TD, 1=ASD).
        y_reg: Continuous ADOS scores for regression (None to skip regression explain).
            Must be restricted to subjects with valid ADOS scores; ``X_reg`` must match.
        df_meta: Clinical metadata aligned to ``X`` (classification subjects).
        feature_names: Names of columns in ``X`` (and ``X_reg``).
        df_cv_clf: Classification CV results DataFrame (from ``run_classification``).
        df_cv_reg: Regression CV results DataFrame (from ``run_regression``, or None).
        output_dir: Root output directory.
        corr_threshold: Preprocessing correlation threshold.
        use_gpu: Use GPU for tree models.
        random_state: Random seed.
        X_reg: Raw feature matrix restricted to subjects with valid ADOS scores.
            Required when ``y_reg`` is not None.  Defaults to ``X`` if not provided,
            but callers must ensure ``X_reg.shape[0] == len(y_reg)``.
        df_meta_reg: Clinical metadata aligned to ``X_reg`` (ADOS-valid subjects only).
            Defaults to ``df_meta`` if not provided.
    """
    out = output_dir / "explain"
    out.mkdir(parents=True, exist_ok=True)

    from .classification import _build_models as clf_models
    from .regression import _build_models as reg_models

    # ── Classification ────────────────────────────────────────
    logger.info("── SHAP: Classification ──")
    try:
        best_clf_id, pipe_clf, X_clf_t, names_clf = _retrain_best(
            df_cv=df_cv_clf,
            X=X,
            y=y_clf,
            metric="auc_roc",
            lower_is_better=False,
            model_builder=clf_models,
            corr_threshold=corr_threshold,
            random_state=random_state,
            use_gpu=use_gpu,
            input_feature_names=feature_names,
        )
        shap_clf, explainer_clf = _get_shap_values(
            best_clf_id, pipe_clf, X_clf_t, task="classification", random_state=random_state
        )
        _save_shap_data(shap_clf, X_clf_t, names_clf, df_meta,
                        "classification", best_clf_id, out)
        _plot_beeswarm(shap_clf, X_clf_t, names_clf, "classification", best_clf_id, out)
        _plot_bar(shap_clf, names_clf, "classification", best_clf_id, out)
        _plot_dependence(shap_clf, X_clf_t, names_clf, "classification", best_clf_id, out)

        # Local: need probability predictions
        y_prob_clf = pipe_clf.predict_proba(X)[:, 1]
        _plot_waterfalls(shap_clf, X_clf_t, names_clf, y_clf, y_prob_clf,
                         best_clf_id, out)
        logger.info(f"  Classification SHAP complete (model: {best_clf_id})")

    except Exception as exc:
        logger.error(f"  Classification SHAP failed: {exc}", exc_info=True)

    # ── Regression ────────────────────────────────────────────
    if y_reg is not None and df_cv_reg is not None:
        logger.info("── SHAP: Regression ──")
        # Use the ADOS-valid-only feature matrix and metadata if provided.
        # Falling back to the full X / df_meta is only safe when all subjects
        # have valid ADOS scores; in the general case callers must supply X_reg.
        X_for_reg = X_reg if X_reg is not None else X
        df_meta_for_reg = df_meta_reg if df_meta_reg is not None else df_meta
        try:
            best_reg_id, pipe_reg, X_reg_t, names_reg = _retrain_best(
                df_cv=df_cv_reg,
                X=X_for_reg,
                y=y_reg,
                metric="rmse",
                lower_is_better=True,
                model_builder=reg_models,
                corr_threshold=corr_threshold,
                random_state=random_state,
                use_gpu=use_gpu,
                input_feature_names=feature_names,
            )
            shap_reg, explainer_reg = _get_shap_values(
                best_reg_id, pipe_reg, X_reg_t, task="regression", random_state=random_state
            )
            _save_shap_data(shap_reg, X_reg_t, names_reg, df_meta_for_reg,
                            "regression", best_reg_id, out)
            _plot_beeswarm(shap_reg, X_reg_t, names_reg, "regression", best_reg_id, out)
            _plot_bar(shap_reg, names_reg, "regression", best_reg_id, out)
            _plot_dependence(shap_reg, X_reg_t, names_reg, "regression", best_reg_id, out)
            logger.info(f"  Regression SHAP complete (model: {best_reg_id})")

        except Exception as exc:
            logger.error(f"  Regression SHAP failed: {exc}", exc_info=True)
