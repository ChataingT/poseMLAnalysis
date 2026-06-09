"""
Config-mode support for the poseMLAnalysis pipeline.

When the user passes ``--config path/to/run.yaml`` (or ``.json``), this module
loads the file, validates the keys, and returns a ``settings`` dict that
run_ml.py uses to filter models and methods.

Without ``--config`` the pipeline runs in **exploratory mode**: all models,
both PCA and UMAP, full random-search CV — identical to the original behaviour.

Config file format (YAML, comments allowed; JSON also accepted)
---------------------------------------------------------------

    # All sections are optional; omitted sections use exploratory defaults.

    global:
      n_jobs: 4
      random_state: 42
      corr_threshold: 0.95

    dimreduce:
      methods: [pca]           # subset of ["pca", "umap"]
      umap:
        n_neighbors: 15
        min_dist: 0.1

    classification:
      enabled: true
      n_outer_folds: 5
      n_inner_folds: 3
      n_iter: 50
      models:
        dt:
          enabled: true
          params:              # fixed hyperparams → skip inner-CV search
            max_depth: 4
            min_samples_leaf: 2
        rf:
          enabled: true        # no params → full random search

    regression:
      enabled: true
      n_outer_folds: 5
      n_inner_folds: 3
      n_iter: 50
      models:
        ridge:
          enabled: true
        rf:
          enabled: true
          params:
            max_depth: 10
            min_samples_leaf: 2

    explain:
      enabled: true
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Valid keys (for validation)
# ─────────────────────────────────────────────────────────────

_VALID_TOP_LEVEL = {"global", "dimreduce", "classification", "regression", "explain", "targets"}

_VALID_GLOBAL = {"n_jobs", "random_state", "corr_threshold"}

_VALID_DIMREDUCE = {"methods", "umap"}
_VALID_DIMREDUCE_UMAP = {"n_neighbors", "min_dist"}

_VALID_TASK = {"enabled", "n_outer_folds", "n_inner_folds", "n_iter", "models", "smote"}
_VALID_MODEL = {"enabled", "params"}

_VALID_CLF_MODELS = {"dt", "lr", "svm", "rf", "xgb", "lgbm"}
_VALID_REG_MODELS = {"dt", "ridge", "svr", "rf", "xgb", "lgbm"}

_VALID_EXPLAIN = {"enabled", "targets"}


# ─────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    """
    Load a YAML or JSON config file.

    YAML is tried first; if PyYAML is not installed the file is parsed as JSON.
    Raises FileNotFoundError if the path does not exist.
    Raises ValueError if the file cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Try YAML first
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(text)
        logger.info(f"Config loaded (YAML): {path}")
        return cfg if cfg is not None else {}
    except ImportError:
        logger.debug("PyYAML not available; trying JSON parser")
    except Exception as exc:
        raise ValueError(f"Failed to parse config as YAML: {exc}") from exc

    # Fallback: JSON
    try:
        cfg = json.loads(text)
        logger.info(f"Config loaded (JSON): {path}")
        return cfg
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse config as YAML or JSON: {exc}\n"
            "Install PyYAML for YAML support: pip install pyyaml"
        ) from exc


# ─────────────────────────────────────────────────────────────
# Validate
# ─────────────────────────────────────────────────────────────

def validate_config(cfg: dict) -> None:
    """
    Validate config keys and basic value types.
    Raises ValueError with a descriptive message on the first problem found.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a YAML/JSON mapping (dict).")

    unknown = set(cfg) - _VALID_TOP_LEVEL
    if unknown:
        raise ValueError(
            f"Unknown top-level config key(s): {sorted(unknown)}. "
            f"Valid keys: {sorted(_VALID_TOP_LEVEL)}"
        )

    # global
    if "global" in cfg:
        g = cfg["global"]
        unknown = set(g) - _VALID_GLOBAL
        if unknown:
            raise ValueError(f"Unknown key(s) in global: {sorted(unknown)}")

    # dimreduce
    if "dimreduce" in cfg:
        dr = cfg["dimreduce"]
        unknown = set(dr) - _VALID_DIMREDUCE
        if unknown:
            raise ValueError(f"Unknown key(s) in dimreduce: {sorted(unknown)}")
        if "methods" in dr:
            methods = dr["methods"]
            if not isinstance(methods, list):
                raise ValueError("dimreduce.methods must be a list, e.g. [pca, umap]")
            invalid = set(methods) - {"pca", "umap"}
            if invalid:
                raise ValueError(
                    f"Invalid dimreduce.methods value(s): {sorted(invalid)}. "
                    "Allowed: pca, umap"
                )
        if "umap" in dr:
            unknown = set(dr["umap"]) - _VALID_DIMREDUCE_UMAP
            if unknown:
                raise ValueError(f"Unknown key(s) in dimreduce.umap: {sorted(unknown)}")

    # classification / regression
    for task, valid_models in [("classification", _VALID_CLF_MODELS),
                                ("regression",     _VALID_REG_MODELS)]:
        if task not in cfg:
            continue
        t = cfg[task]
        unknown = set(t) - _VALID_TASK
        if unknown:
            raise ValueError(f"Unknown key(s) in {task}: {sorted(unknown)}")
        if "models" in t:
            for model_id, mcfg in t["models"].items():
                if model_id not in valid_models:
                    raise ValueError(
                        f"Unknown model '{model_id}' in {task}.models. "
                        f"Valid models: {sorted(valid_models)}"
                    )
                if not isinstance(mcfg, dict):
                    raise ValueError(
                        f"{task}.models.{model_id} must be a mapping "
                        f"(e.g. {{enabled: true}})"
                    )
                unknown = set(mcfg) - _VALID_MODEL
                if unknown:
                    raise ValueError(
                        f"Unknown key(s) in {task}.models.{model_id}: {sorted(unknown)}"
                    )

    # targets (top-level list)
    if "targets" in cfg:
        t = cfg["targets"]
        if not isinstance(t, list):
            raise ValueError("top-level 'targets' must be a list of column name strings")
        if not all(isinstance(x, str) for x in t):
            raise ValueError("All entries in top-level 'targets' must be strings")

    # explain
    if "explain" in cfg:
        unknown = set(cfg["explain"]) - _VALID_EXPLAIN
        if unknown:
            raise ValueError(f"Unknown key(s) in explain: {sorted(unknown)}")
        if "targets" in cfg["explain"]:
            et = cfg["explain"]["targets"]
            if not isinstance(et, list) or not all(isinstance(x, str) for x in et):
                raise ValueError("explain.targets must be a list of column name strings")


# ─────────────────────────────────────────────────────────────
# Apply
# ─────────────────────────────────────────────────────────────

def apply_config(cfg: dict, args: "argparse.Namespace") -> dict:  # noqa: F821
    """
    Merge a validated config dict with the parsed CLI args.

    Returns a ``settings`` dict with all the keys that run_ml.py needs:

    Global pipeline controls (may override CLI):
      n_jobs, random_state, corr_threshold
      umap_n_neighbors, umap_min_dist
      skip_dimreduce, skip_classification, skip_regression, skip_explain

    Per-task CV controls (independent for classification and regression):
      clf_n_outer_folds, clf_n_inner_folds, clf_n_iter
      reg_n_outer_folds, reg_n_inner_folds, reg_n_iter

    Per-step controls:
      dimreduce_methods  : list[str] — ["pca"] / ["umap"] / ["pca","umap"]
      clf_model_filter   : list[str] | None — None means "all"
      clf_fixed_params   : dict  {model_id: {param: value, ...}}
      reg_model_filter   : list[str] | None
      reg_fixed_params   : dict
    """
    # Default targets (backward-compatible)
    _DEFAULT_TARGETS = ["diagnosis", "ADOS_2_TOTAL"]

    # Start from CLI args (always present); CV defaults shared across tasks
    s: dict[str, Any] = {
        # global
        "n_jobs":          args.n_jobs,
        "random_state":    args.random_state,
        "corr_threshold":  args.corr_threshold,
        # per-task CV (initialised from CLI defaults, overridden independently below)
        "clf_n_outer_folds": args.n_outer_folds,
        "clf_n_inner_folds": args.n_inner_folds,
        "clf_n_iter":        args.n_iter,
        "clf_use_smote":     True,
        "reg_n_outer_folds": args.n_outer_folds,
        "reg_n_inner_folds": args.n_inner_folds,
        "reg_n_iter":        args.n_iter,
        # UMAP
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist":   args.umap_min_dist,
        # skip flags
        "skip_dimreduce":       args.skip_dimreduce,
        "skip_classification":  args.skip_classification,
        "skip_regression":      args.skip_regression,
        "skip_explain":         args.skip_explain,
        # method/model filters (exploratory defaults)
        "dimreduce_methods":  ["pca", "umap"],
        "clf_model_filter":   None,
        "clf_fixed_params":   {},
        "reg_model_filter":   None,
        "reg_fixed_params":   {},
        # multi-target
        "targets_list":       _DEFAULT_TARGETS,
        "explain_targets":    _DEFAULT_TARGETS,
    }

    # Override with global section
    g = cfg.get("global", {})
    if "n_jobs"          in g: s["n_jobs"]         = int(g["n_jobs"])
    if "random_state"    in g: s["random_state"]   = int(g["random_state"])
    if "corr_threshold"  in g: s["corr_threshold"] = float(g["corr_threshold"])

    # dimreduce section
    dr = cfg.get("dimreduce", {})
    if "methods" in dr:
        s["dimreduce_methods"] = [m.lower() for m in dr["methods"]]
    if not s["dimreduce_methods"]:
        # Empty list → skip dimreduce entirely
        s["skip_dimreduce"] = True
    umap_cfg = dr.get("umap", {})
    if "n_neighbors" in umap_cfg: s["umap_n_neighbors"] = int(umap_cfg["n_neighbors"])
    if "min_dist"    in umap_cfg: s["umap_min_dist"]    = float(umap_cfg["min_dist"])

    # classification section — CV params written to clf_* keys only
    clf = cfg.get("classification", {})
    if "enabled" in clf and not clf["enabled"]:
        s["skip_classification"] = True
    if "n_outer_folds" in clf: s["clf_n_outer_folds"] = int(clf["n_outer_folds"])
    if "n_inner_folds" in clf: s["clf_n_inner_folds"] = int(clf["n_inner_folds"])
    if "n_iter"        in clf: s["clf_n_iter"]         = int(clf["n_iter"])
    if "smote"         in clf: s["clf_use_smote"]      = bool(clf["smote"])
    if "models" in clf:
        enabled_models = []
        fixed = {}
        for mid, mcfg in clf["models"].items():
            if mcfg.get("enabled", True):
                enabled_models.append(mid)
                if "params" in mcfg and mcfg["params"]:
                    fixed[mid] = dict(mcfg["params"])
        s["clf_model_filter"] = enabled_models if enabled_models else None
        s["clf_fixed_params"] = fixed

    # regression section — CV params written to reg_* keys only
    reg = cfg.get("regression", {})
    if "enabled" in reg and not reg["enabled"]:
        s["skip_regression"] = True
    if "n_outer_folds" in reg: s["reg_n_outer_folds"] = int(reg["n_outer_folds"])
    if "n_inner_folds" in reg: s["reg_n_inner_folds"] = int(reg["n_inner_folds"])
    if "n_iter"        in reg: s["reg_n_iter"]         = int(reg["n_iter"])
    if "models" in reg:
        enabled_models = []
        fixed = {}
        for mid, mcfg in reg["models"].items():
            if mcfg.get("enabled", True):
                enabled_models.append(mid)
                if "params" in mcfg and mcfg["params"]:
                    fixed[mid] = dict(mcfg["params"])
        s["reg_model_filter"] = enabled_models if enabled_models else None
        s["reg_fixed_params"] = fixed

    # explain section
    exp = cfg.get("explain", {})
    if "enabled" in exp and not exp["enabled"]:
        s["skip_explain"] = True

    # targets section: override default targets list
    if "targets" in cfg:
        s["targets_list"] = list(cfg["targets"])
    # explain.targets: which targets get SHAP analysis
    if "targets" in exp:
        s["explain_targets"] = list(exp["targets"])
    elif "targets" in cfg:
        # if targets given but no explain.targets, keep explain for first clf target
        s["explain_targets"] = list(cfg["targets"])

    return s


# ─────────────────────────────────────────────────────────────
# Convenience: build exploratory-mode settings (no config file)
# ─────────────────────────────────────────────────────────────

def exploratory_settings(args: "argparse.Namespace") -> dict:  # noqa: F821
    """Return settings dict for exploratory mode (identical to old behaviour)."""
    _default_targets = ["diagnosis", "ADOS_2_TOTAL"]
    return {
        "n_jobs":           args.n_jobs,
        "random_state":     args.random_state,
        "corr_threshold":   args.corr_threshold,
        "clf_n_outer_folds": args.n_outer_folds,
        "clf_n_inner_folds": args.n_inner_folds,
        "clf_n_iter":        args.n_iter,
        "clf_use_smote":     True,
        "reg_n_outer_folds": args.n_outer_folds,
        "reg_n_inner_folds": args.n_inner_folds,
        "reg_n_iter":        args.n_iter,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_min_dist":    args.umap_min_dist,
        "skip_dimreduce":       args.skip_dimreduce,
        "skip_classification":  args.skip_classification,
        "skip_regression":      args.skip_regression,
        "skip_explain":         args.skip_explain,
        "dimreduce_methods":  ["pca", "umap"],
        "clf_model_filter":   None,
        "clf_fixed_params":   {},
        "reg_model_filter":   None,
        "reg_fixed_params":   {},
        "targets_list":       _default_targets,
        "explain_targets":    _default_targets,
    }
