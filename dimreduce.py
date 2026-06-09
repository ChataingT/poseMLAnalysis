"""
Dimensionality reduction visualisations for the ML pipeline.

Produces PCA and UMAP 2-D embeddings of the preprocessed feature matrix,
coloured by clinical variables (diagnosis, gender, age, ADOS).

PCA outputs:
    dimreduce/pca_explained_variance.png
    dimreduce/pca_scatter_{diagnosis|gender|age|ados}.png
    dimreduce/pca_biplot.png

UMAP outputs:
    dimreduce/umap_trustworthiness.png
    dimreduce/umap_n_neighbors_sensitivity.png
    dimreduce/umap_scatter_{diagnosis|gender|age|ados}.png

Combined panels:
    dimreduce/combined_{diagnosis|gender|age|ados}.png   (PCA + UMAP side by side)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants / style
# ─────────────────────────────────────────────────────────────

DPI = 150
PALETTE_DIAGNOSIS = {"ASD": "#E74C3C", "TD": "#2E86AB"}
CMAP_CONTINUOUS = "viridis"
POINT_SIZE = 50
POINT_ALPHA = 0.8

CLINICAL_VARS = [
    ("diagnosis", "categorical"),
    ("gender",    "categorical"),
    ("age",       "continuous"),
    ("ados",      "continuous"),
]

GENDER_PALETTE = {"Male": "#2980B9", "Female": "#C0392B", "M": "#2980B9", "F": "#C0392B"}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _scatter_categorical(ax, emb, labels, palette, title):
    categories = sorted(labels.dropna().unique())
    for cat in categories:
        mask = labels == cat
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=palette.get(str(cat), "grey"),
            label=str(cat),
            s=POINT_SIZE, alpha=POINT_ALPHA, edgecolors="white", linewidths=0.3,
        )
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.7)


def _scatter_continuous(ax, emb, values, title, cmap=CMAP_CONTINUOUS):
    valid = ~values.isna()
    sc = ax.scatter(
        emb[valid, 0], emb[valid, 1],
        c=values[valid].values, cmap=cmap,
        s=POINT_SIZE, alpha=POINT_ALPHA, edgecolors="white", linewidths=0.3,
    )
    if (~valid).any():
        ax.scatter(
            emb[~valid, 0], emb[~valid, 1],
            c="lightgrey", s=POINT_SIZE, alpha=0.4, edgecolors="white", linewidths=0.3,
            label="missing",
        )
        ax.legend(fontsize=9, framealpha=0.7)
    plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    ax.set_title(title, fontsize=11)


def _style_ax(ax, xlabel="Component 1", ylabel="Component 2"):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────
# Clinical variable extraction
# ─────────────────────────────────────────────────────────────

def _get_clinical_series(df_meta: pd.DataFrame):
    """
    Return named pd.Series aligned to df_meta.index for each clinical variable.
    """
    diag = df_meta.get("diagnosis", pd.Series(np.nan, index=df_meta.index))
    gender_raw = df_meta.get("gender", pd.Series(np.nan, index=df_meta.index))
    age = pd.to_numeric(df_meta.get("Ados_2_Age", pd.Series(np.nan, index=df_meta.index)), errors="coerce")
    ados = pd.to_numeric(df_meta.get("ADOS_G_ADOS_2_TOTAL_score_de_severite", pd.Series(np.nan, index=df_meta.index)), errors="coerce")
    return diag, gender_raw, age, ados


# ─────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────

def run_pca(
    X_scaled: np.ndarray,
    df_meta: pd.DataFrame,
    output_dir: Path,
    n_top_biplot: int = 10,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA on X_scaled, save all figures.

    Args:
        X_scaled: (n_subjects, n_features) scaled numpy array.
        df_meta: DataFrame with clinical columns (same row order as X_scaled).
        output_dir: Root output directory; figures go into output_dir/dimreduce/.
        n_top_biplot: Number of loading vectors shown in biplot.
        feature_names: Feature column names (for biplot). Optional.

    Returns:
        (embedding_2d, fitted_pca)
    """
    out = output_dir / "dimreduce"
    out.mkdir(parents=True, exist_ok=True)

    n_components = min(X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    diag, gender, age, ados = _get_clinical_series(df_meta)
    ados_rrb = pd.to_numeric(df_meta.get("ADOS_2_ADOS_G_REVISED_RRB_SEVERITY_SCORE_new",
                                          pd.Series(np.nan, index=df_meta.index)), errors="coerce")
    ados_sa  = pd.to_numeric(df_meta.get("ADOS_2_ADOS_G_REVISED_SA_SEVERITY_SCORE",
                                          pd.Series(np.nan, index=df_meta.index)), errors="coerce")
    ados_total = pd.to_numeric(df_meta.get("ADOS_G_ADOS_2_TOTAL_score_de_severite",
                                            pd.Series(np.nan, index=df_meta.index)), errors="coerce")

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    # ── Save: explained variance data ────────────────────────
    evr_df = pd.DataFrame({
        "component": range(1, len(evr) + 1),
        "explained_variance_ratio": evr,
        "cumulative_variance_ratio": cumulative,
        "explained_variance_pct": evr * 100,
        "cumulative_variance_pct": cumulative * 100,
    })
    evr_df.to_csv(out / "pca_explained_variance.csv", index=False)
    logger.info("  Saved data: pca_explained_variance.csv")

    # ── Explained variance curve ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(evr) + 1), evr * 100, alpha=0.6, color="steelblue", label="Individual")
    ax.plot(range(1, len(cumulative) + 1), cumulative * 100, "o-", color="darkred",
            markersize=3, linewidth=1.5, label="Cumulative")
    idx_90 = np.searchsorted(cumulative, 0.90)
    if idx_90 < len(cumulative):
        ax.axvline(idx_90 + 1, color="darkred", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(90, color="darkred", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(idx_90 + 1.5, 91, f"90% @ PC{idx_90 + 1}", fontsize=8, color="darkred")
    ax.set_xlabel("Principal Component", fontsize=10)
    ax.set_ylabel("Explained Variance (%)", fontsize=10)
    ax.set_title("PCA — Explained Variance", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, min(50, len(evr)) + 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out / "pca_explained_variance.png")

    emb2 = coords[:, :2]
    pc1_var = evr[0] * 100
    pc2_var = evr[1] * 100 if len(evr) > 1 else 0

    xlabel = f"PC1 ({pc1_var:.1f}%)"
    ylabel = f"PC2 ({pc2_var:.1f}%)"

    # ── Save: PCA embedding (all components up to first 20) ──
    n_save = min(20, coords.shape[1])
    emb_df = pd.DataFrame(
        coords[:, :n_save],
        columns=[f"PC{i+1}" for i in range(n_save)],
        index=df_meta.index,
    )
    emb_df.insert(0, "uuid", df_meta.get("uuid", pd.Series(range(len(df_meta)), index=df_meta.index)))
    emb_df.insert(1, "diagnosis", diag.values)
    emb_df.insert(2, "gender", gender.values)
    emb_df.insert(3, "Ados_2_Age", age.values)
    emb_df.insert(4, "ADOS_G_ADOS_2_TOTAL_score_de_severite", ados.values)
    emb_df.to_csv(out / "pca_embedding.csv", index=False)
    logger.info("  Saved data: pca_embedding.csv")

    # ── Save: top-30 loadings for PC1 and PC2 ────────────────
    if feature_names is not None and len(feature_names) == X_scaled.shape[1]:
        loadings_full = pca.components_[:2].T  # (n_features, 2)
        loading_magnitudes = np.linalg.norm(loadings_full, axis=1)
        top50 = np.argsort(-loading_magnitudes)[:50]
        load_df = pd.DataFrame({
            "feature": [feature_names[i] for i in top50],
            "loading_PC1": loadings_full[top50, 0],
            "loading_PC2": loadings_full[top50, 1],
            "loading_magnitude": loading_magnitudes[top50],
        })
        load_df.to_csv(out / "pca_loadings_top50.csv", index=False)
        logger.info("  Saved data: pca_loadings_top50.csv")

    # ── Scatter by diagnosis ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_categorical(ax, emb2, diag, PALETTE_DIAGNOSIS, "PCA — Diagnosis")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "pca_scatter_diagnosis.png")

    # ── Scatter by gender ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_categorical(ax, emb2, gender, GENDER_PALETTE, "PCA — Gender")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "pca_scatter_gender.png")

    # ── Scatter by age ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_continuous(ax, emb2, age, "PCA — Age (months)")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "pca_scatter_age.png")

    # ── Scatter by ADOS ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_continuous(ax, emb2, ados, "PCA — ADOS-G Severity Score")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "pca_scatter_ados.png")

    # ── Scatter by ADOS RRB severity score ───────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_continuous(ax, emb2, ados_rrb, "PCA — ADOS-2 RRB Severity Score")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "pca_scatter_ados_rrb_severity.png")

    # ── Scatter by ADOS SA severity score ────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_continuous(ax, emb2, ados_sa, "PCA — ADOS-2 SA Severity Score")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "pca_scatter_ados_sa_severity.png")

    # ── Scatter by ADOS-G total severity score ───────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_continuous(ax, emb2, ados_total, "PCA — ADOS-G Total Severity Score")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "pca_scatter_ados_total_severity.png")

    # ── Biplot (top N loading vectors) ────────────────────────
    if feature_names is not None and len(feature_names) == X_scaled.shape[1]:
        fig, ax = plt.subplots(figsize=(8, 7))
        _scatter_categorical(ax, emb2, diag, PALETTE_DIAGNOSIS, "PCA Biplot — Top Loadings")
        loadings = pca.components_[:2].T  # (n_features, 2)
        loading_magnitudes = np.linalg.norm(loadings, axis=1)
        top_idx = np.argsort(-loading_magnitudes)[:n_top_biplot]
        scale = np.max(np.abs(emb2)) / np.max(loading_magnitudes) * 0.8
        for i in top_idx:
            ax.annotate(
                "", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            )
            ax.text(
                loadings[i, 0] * scale * 1.05, loadings[i, 1] * scale * 1.05,
                feature_names[i], fontsize=6, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
            )
        _style_ax(ax, xlabel, ylabel)
        _save(fig, out / "pca_biplot.png")

    logger.info(
        f"PCA: {n_components} components, top 2 explain "
        f"{(pc1_var + pc2_var):.1f}% variance"
    )
    return emb2, pca


# ─────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────

def run_umap(
    X_scaled: np.ndarray,
    df_meta: pd.DataFrame,
    output_dir: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    use_gpu: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fit UMAP on X_scaled, save all figures.

    Args:
        X_scaled: (n_subjects, n_features) scaled numpy array.
        df_meta: DataFrame with clinical columns.
        output_dir: Root output directory; figures go into output_dir/dimreduce/.
        n_neighbors: UMAP n_neighbors (default 15).
        min_dist: UMAP min_dist (default 0.1).
        use_gpu: Try cuml.manifold.UMAP if True.
        random_state: Random seed.

    Returns:
        embedding_2d: (n_subjects, 2) UMAP embedding.
    """
    out = output_dir / "dimreduce"
    out.mkdir(parents=True, exist_ok=True)

    diag, gender, age, ados = _get_clinical_series(df_meta)

    # ── UMAP fit (primary, 2-D) ──────────────────────────────
    umap_model = _fit_umap(X_scaled, n_neighbors=n_neighbors, min_dist=min_dist,
                           n_components=2, use_gpu=use_gpu, random_state=random_state)
    emb2 = umap_model.embedding_ if hasattr(umap_model, "embedding_") else umap_model

    xlabel, ylabel = "UMAP 1", "UMAP 2"

    # ── Scatter by diagnosis ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_categorical(ax, emb2, diag, PALETTE_DIAGNOSIS, "UMAP — Diagnosis")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "umap_scatter_diagnosis.png")

    # ── Scatter by gender ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_categorical(ax, emb2, gender, GENDER_PALETTE, "UMAP — Gender")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "umap_scatter_gender.png")

    # ── Scatter by age ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_continuous(ax, emb2, age, "UMAP — Age (months)")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "umap_scatter_age.png")

    # ── Scatter by ADOS ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_continuous(ax, emb2, ados, "UMAP — ADOS-G Severity Score")
    _style_ax(ax, xlabel, ylabel)
    _save(fig, out / "umap_scatter_ados.png")

    # ── Save: UMAP embedding ──────────────────────────────────
    umap_df = pd.DataFrame(
        {"UMAP1": emb2[:, 0], "UMAP2": emb2[:, 1]},
        index=df_meta.index,
    )
    umap_df.insert(0, "uuid", df_meta.get("uuid", pd.Series(range(len(df_meta)), index=df_meta.index)))
    umap_df.insert(1, "diagnosis", diag.values)
    umap_df.insert(2, "gender", gender.values)
    umap_df.insert(3, "Ados_2_Age", age.values)
    umap_df.insert(4, "ADOS_G_ADOS_2_TOTAL_score_de_severite", ados.values)
    umap_df.to_csv(out / "umap_embedding.csv", index=False)
    logger.info("  Saved data: umap_embedding.csv")

    # ── Trustworthiness curve (n_components = 1..8) ───────────
    _plot_trustworthiness(X_scaled, out, n_neighbors=n_neighbors,
                          use_gpu=use_gpu, random_state=random_state)

    # ── n_neighbors sensitivity grid ─────────────────────────
    _plot_n_neighbors_sensitivity(X_scaled, diag, out,
                                  use_gpu=use_gpu, random_state=random_state)

    return emb2


def _fit_umap(X, n_neighbors=15, min_dist=0.1, n_components=2, use_gpu=False, random_state=42):
    """Fit UMAP (CPU or GPU). Returns fitted model with .embedding_ attribute or array."""
    if use_gpu:
        try:
            from cuml.manifold import UMAP as cumlUMAP
            model = cumlUMAP(
                n_neighbors=n_neighbors, min_dist=min_dist,
                n_components=n_components, random_state=random_state,
            )
            emb = model.fit_transform(X)
            logger.info("  UMAP: using cuml GPU backend")
            return emb  # cuml returns array directly
        except ImportError:
            logger.warning("  cuml not available; falling back to CPU UMAP")

    import umap as umap_lib
    model = umap_lib.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components,
        random_state=random_state, n_jobs=-1,
    )
    model.fit(X)
    logger.info(f"  UMAP (CPU): n_neighbors={n_neighbors}, min_dist={min_dist}")
    return model


def _plot_trustworthiness(X_scaled, out, n_neighbors=15, use_gpu=False, random_state=42,
                          max_components=8):
    """Trustworthiness curve for n_components = 1..max_components."""
    n_components_range = range(1, min(max_components + 1, X_scaled.shape[1] + 1))
    trust_scores = []
    for nc in n_components_range:
        model = _fit_umap(X_scaled, n_neighbors=n_neighbors, min_dist=0.1,
                          n_components=nc, use_gpu=use_gpu, random_state=random_state)
        emb = model.embedding_ if hasattr(model, "embedding_") else model
        t = trustworthiness(X_scaled, emb, n_neighbors=n_neighbors)
        trust_scores.append(t)
        logger.debug(f"    Trustworthiness (n_components={nc}): {t:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(n_components_range), trust_scores, "o-", color="steelblue",
            markersize=6, linewidth=2)
    ax.axhline(0.90, color="darkred", linestyle="--", linewidth=1.2, label="0.90 threshold")
    ax.set_xlabel("UMAP n_components", fontsize=10)
    ax.set_ylabel("Trustworthiness", fontsize=10)
    ax.set_title("UMAP — Trustworthiness Curve", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(list(n_components_range))
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out / "umap_trustworthiness.png")


def _plot_n_neighbors_sensitivity(X_scaled, diag, out, use_gpu=False, random_state=42):
    """2×2 grid of UMAP embeddings for n_neighbors ∈ {5, 15, 30, 50}."""
    n_neighbors_values = [5, 15, 30, 50]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()

    for ax, nn in zip(axes, n_neighbors_values):
        model = _fit_umap(X_scaled, n_neighbors=nn, min_dist=0.1, n_components=2,
                          use_gpu=use_gpu, random_state=random_state)
        emb = model.embedding_ if hasattr(model, "embedding_") else model
        _scatter_categorical(ax, emb, diag, PALETTE_DIAGNOSIS, f"n_neighbors={nn}")
        _style_ax(ax, "UMAP 1", "UMAP 2")

    fig.suptitle("UMAP — n_neighbors Sensitivity (colored by Diagnosis)", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out / "umap_n_neighbors_sensitivity.png")


# ─────────────────────────────────────────────────────────────
# Combined panels (PCA + UMAP side by side)
# ─────────────────────────────────────────────────────────────

def plot_combined_panels(
    pca_emb: np.ndarray,
    umap_emb: np.ndarray,
    df_meta: pd.DataFrame,
    pca_var: tuple[float, float],
    output_dir: Path,
) -> None:
    """
    2×4 (2 rows = PCA/UMAP, 4 cols = diagnosis/gender/age/ados) combined figure.
    One combined .png per clinical variable.

    Args:
        pca_emb:  (n, 2) PCA embedding.
        umap_emb: (n, 2) UMAP embedding.
        df_meta:  DataFrame with clinical columns.
        pca_var:  (pc1_var_pct, pc2_var_pct) explained variance percentages.
        output_dir: Root output directory.
    """
    out = output_dir / "dimreduce"
    out.mkdir(parents=True, exist_ok=True)

    diag, gender, age, ados = _get_clinical_series(df_meta)

    var_configs = [
        ("diagnosis", diag, "categorical", PALETTE_DIAGNOSIS, "Diagnosis"),
        ("gender",    gender, "categorical", GENDER_PALETTE,    "Gender"),
        ("age",       age,    "continuous",  None,               "Age (months)"),
        ("ados",      ados,   "continuous",  None,               "ADOS-2 Total"),
    ]

    pca_xlabel = f"PC1 ({pca_var[0]:.1f}%)"
    pca_ylabel = f"PC2 ({pca_var[1]:.1f}%)"

    for var_name, series, kind, palette, label in var_configs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, emb, method, xl, yl in [
            (axes[0], pca_emb,  "PCA",  pca_xlabel, pca_ylabel),
            (axes[1], umap_emb, "UMAP", "UMAP 1",   "UMAP 2"),
        ]:
            title = f"{method} — {label}"
            if kind == "categorical":
                _scatter_categorical(ax, emb, series, palette, title)
            else:
                _scatter_continuous(ax, emb, series, title)
            _style_ax(ax, xl, yl)

        fig.suptitle(label, fontsize=14, y=1.02)
        fig.tight_layout()
        _save(fig, out / f"combined_{var_name}.png")


# ─────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────

def run_dimensionality_reduction(
    X_scaled: np.ndarray,
    df_meta: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
    use_gpu: bool = False,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
    methods: list[str] | None = None,
) -> dict:
    """
    Run PCA and/or UMAP, save all figures, return embeddings.

    Args:
        X_scaled: Preprocessed, scaled feature matrix (n_subjects × n_features).
        df_meta: Clinical metadata DataFrame (same row order as X_scaled).
        feature_names: Names for the feature columns in X_scaled.
        output_dir: Root output directory.
        use_gpu: Try cuml for UMAP.
        umap_n_neighbors, umap_min_dist: UMAP hyperparameters.
        random_state: Random seed.
        methods: Subset of ["pca", "umap"] to run. Defaults to both.

    Returns:
        dict with keys "pca_emb", "umap_emb", "pca_model" (absent keys are None).
    """
    if methods is None:
        methods = ["pca", "umap"]
    methods = [m.lower() for m in methods]

    result: dict = {"pca_emb": None, "umap_emb": None, "pca_model": None}

    pca_emb = None
    pca_var = (0.0, 0.0)

    if "pca" in methods:
        logger.info("── Dimensionality reduction: PCA ──")
        pca_emb, pca_model = run_pca(
            X_scaled, df_meta, output_dir,
            feature_names=feature_names,
        )
        evr = pca_model.explained_variance_ratio_
        pca_var = (evr[0] * 100, evr[1] * 100 if len(evr) > 1 else 0.0)
        result["pca_emb"] = pca_emb
        result["pca_model"] = pca_model
    else:
        logger.info("── Dimensionality reduction: PCA skipped (not in methods) ──")

    umap_emb = None
    if "umap" in methods:
        logger.info("── Dimensionality reduction: UMAP ──")
        umap_emb = run_umap(
            X_scaled, df_meta, output_dir,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            use_gpu=use_gpu,
            random_state=random_state,
        )
        result["umap_emb"] = umap_emb
    else:
        logger.info("── Dimensionality reduction: UMAP skipped (not in methods) ──")

    if pca_emb is not None and umap_emb is not None:
        logger.info("── Combined PCA + UMAP panels ──")
        plot_combined_panels(pca_emb, umap_emb, df_meta, pca_var, output_dir)

    return result
