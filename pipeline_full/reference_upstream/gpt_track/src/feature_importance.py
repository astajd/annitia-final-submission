"""Feature importance for the strongest models.

For tree-based models we read native importances; for the binary GBMs we also
compute SHAP values on the train set when ``shap`` is available. Output is a
single CSV per (experiment, endpoint, model) and a roll-up
``reports/phase2_feature_importance.md`` flagging suspicious top features
(follow-up duration, last age, visit count, missingness count).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .features import build_feature_set
from .models import build_model
from .models._preprocess import fill_for_tree
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger

_LOG = get_logger(__name__)

SUSPICIOUS_PATTERNS = (
    "n_visits",
    "followup",
    "follow_up",
    "last_observed_age",
    "age_last",
    "time_since_baseline",
    "miss_total",
    "miss_visit_",
    "_count",
    "gap_max",
    "gap_min",
    "gap_mean",
)


def _is_suspicious(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in SUSPICIOUS_PATTERNS)


def _train_full_model(model_name: str, params: dict, fs, endpoint) -> object:
    Xtr = fs.X_train
    mask = pd.Series(np.asarray(endpoint.mask, dtype=bool), index=Xtr.index)
    m = build_model(model_name, params)
    m.fit(Xtr, endpoint, mask=mask)
    return m


def _native_importance(model, feature_names: list[str]) -> pd.Series:
    inner = getattr(model, "_model", None)
    if inner is None:
        return pd.Series(dtype=float)
    try:
        imp_attr = getattr(inner, "feature_importances_", None)
        imp = np.asarray(imp_attr, dtype=float) if imp_attr is not None else None
        if imp is not None and len(imp) == len(feature_names):
            return pd.Series(imp, index=feature_names).sort_values(ascending=False)
    except (NotImplementedError, AttributeError):
        pass
    if hasattr(inner, "coef_"):
        try:
            coef = np.asarray(inner.coef_, dtype=float).ravel()
            if len(coef) == len(feature_names):
                return pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False)
        except Exception:
            pass
    return pd.Series(dtype=float)


def compute_for(
    feature_set: str,
    model_name: str,
    endpoint: str,
    params: dict | None = None,
    use_shap: bool = True,
    out_dir: Path | None = None,
) -> Path:
    """Train one model on the full train set and persist a CSV of importances."""
    ds = load_dataset()
    fs = build_feature_set(feature_set, ds)
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    ep = hep if endpoint == "hepatic" else death

    model = _train_full_model(model_name, params or {}, fs, ep)
    cols = list(fill_for_tree(fs.X_train).columns)
    importance = _native_importance(model, cols)

    out_dir = out_dir or (cfg.EXPERIMENT_OUTPUTS / "phase2_feature_importance")
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{feature_set}__{endpoint}__{model_name}"
    csv = out_dir / f"{label}.csv"
    df = importance.reset_index()
    df.columns = ["feature", "importance"]
    df["suspicious"] = df["feature"].map(_is_suspicious)
    df.to_csv(csv, index=False)

    if use_shap and model_name in {"lgbm_binary", "xgb_binary", "catboost_binary", "xgb_cox"}:
        try:
            import shap

            inner = model._model
            X_pre = fill_for_tree(fs.X_train).reindex(columns=model._cols, fill_value=0)
            sample = X_pre.sample(min(len(X_pre), 500), random_state=0)
            explainer = shap.TreeExplainer(inner)
            shap_values = explainer.shap_values(sample.values)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            mean_abs = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                "feature": model._cols,
                "shap_mean_abs": mean_abs,
            }).sort_values("shap_mean_abs", ascending=False)
            shap_df["suspicious"] = shap_df["feature"].map(_is_suspicious)
            shap_csv = out_dir / f"{label}__shap.csv"
            shap_df.to_csv(shap_csv, index=False)
        except Exception as e:  # noqa: BLE001
            _LOG.warning("SHAP failed for %s: %s", label, e)

    _LOG.info("wrote %s (%d features)", csv, len(df))
    return csv


def write_top_report(out_path: Path | None = None, top_k: int = 20) -> Path:
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase2_feature_importance"
    out_path = out_path or (cfg.REPORTS_DIR / "phase2_feature_importance.md")
    if not out_root.exists():
        out_path.write_text("# Feature importance\n\n(no importance files yet — run `python -m src.feature_importance`)\n")
        return out_path

    lines = ["# Phase 2 feature importance\n"]
    lines.append(
        "Top features per (feature_set, endpoint, model). "
        "`suspicious=True` flags features that match known follow-up / missingness / cutoff patterns "
        f"({', '.join(SUSPICIOUS_PATTERNS)}). Treat any model whose top-{top_k} contains such features with caution.\n"
    )
    for csv in sorted(out_root.glob("*.csv")):
        if csv.name.endswith("__shap.csv"):
            continue
        df = pd.read_csv(csv).head(top_k)
        if df.empty:
            continue
        n_susp = int(df["suspicious"].sum())
        lines.append(f"## {csv.stem}  (top-{top_k}, {n_susp} suspicious)\n")
        lines.append(df.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        shap_csv = out_root / f"{csv.stem}__shap.csv"
        if shap_csv.exists():
            shap_df = pd.read_csv(shap_csv).head(top_k)
            n_s = int(shap_df["suspicious"].sum())
            lines.append(f"### SHAP top-{top_k} ({n_s} suspicious)\n")
            lines.append(shap_df.to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    _LOG.info("wrote %s", out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--feature-set", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", choices=["hepatic", "death"], required=True)
    args = p.parse_args()
    compute_for(args.feature_set, args.model, args.endpoint)
    write_top_report()
