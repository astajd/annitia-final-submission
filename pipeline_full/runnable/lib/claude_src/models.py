"""Model wrappers — uniform (X_tr, e_tr, t_tr, X_va) -> risk_va interface."""
from __future__ import annotations
import numpy as np
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest

from .data import survival_y

_TIE_BREAKER = 1e-9


def _add_jitter(arr, seed=0):
    rng = np.random.default_rng(seed)
    return arr + _TIE_BREAKER * rng.standard_normal(len(arr))


def make_coxnet(l1_ratio=0.5, alpha_min_ratio=0.01):
    def fit_predict(X_tr, e_tr, t_tr, X_va):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("cox", CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio,
                max_iter=2000, fit_baseline_model=False)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_tr, survival_y(e_tr, t_tr))
        return _add_jitter(np.asarray(pipe.predict(X_va)).ravel())
    return fit_predict


def make_rsf(n_estimators=200, min_samples_leaf=20, min_samples_split=40,
             max_features="sqrt", seed=42):
    def fit_predict(X_tr, e_tr, t_tr, X_va):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rsf", RandomSurvivalForest(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split, max_features=max_features,
                n_jobs=-1, random_state=seed)),
        ])
        pipe.fit(X_tr, survival_y(e_tr, t_tr))
        return _add_jitter(np.asarray(pipe.predict(X_va)).ravel(), seed=seed)
    return fit_predict


def make_xgb_cox(n_estimators=300, learning_rate=0.03, max_depth=4,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5.0,
                 reg_lambda=1.0, seed=42, n_jobs=1):
    # On this data (~1k rows) XGB's default thread auto-pick is ~75× slower
    # than n_jobs=1 because the per-tree work is too small to amortize the
    # OpenMP fork/join overhead. n_jobs=1 keeps a single-fit fast; outer
    # parallelism (e.g. multiple folds) is what should provide concurrency.
    import xgboost as xgb

    def fit_predict(X_tr, e_tr, t_tr, X_va):
        y_tr = np.where(e_tr, t_tr, -t_tr)
        X_tr_arr = X_tr.to_numpy(dtype=float)
        X_va_arr = X_va.to_numpy(dtype=float)
        model = xgb.XGBRegressor(
            objective="survival:cox",
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, subsample=subsample,
            colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
            reg_lambda=reg_lambda, tree_method="hist",
            n_jobs=n_jobs, verbosity=0, random_state=seed)
        model.fit(X_tr_arr, y_tr)
        return _add_jitter(np.asarray(model.predict(X_va_arr)).ravel(), seed=seed)
    return fit_predict


def _binary_label_with_horizon(e, t, horizon):
    e = np.asarray(e, dtype=bool); t = np.asarray(t, dtype=float)
    label = np.zeros(len(e), dtype=int)
    mask = np.ones(len(e), dtype=bool)
    label[e & (t <= horizon)] = 1
    mask[(~e) & (t < horizon)] = False  # censored before horizon = unknown, drop
    return label, mask


def make_lgbm_binary(horizon=5.0, n_estimators=300, learning_rate=0.03,
                     num_leaves=31, min_child_samples=20, reg_lambda=0.0, seed=42):
    import lightgbm as lgb

    def fit_predict(X_tr, e_tr, t_tr, X_va):
        y_tr, m_tr = _binary_label_with_horizon(e_tr, t_tr, horizon)
        if m_tr.sum() < 20 or y_tr[m_tr].sum() < 3:
            return _add_jitter(np.zeros(len(X_va)), seed=seed)
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            reg_lambda=reg_lambda, class_weight="balanced",
            random_state=seed, verbose=-1)
        clf.fit(X_tr.iloc[m_tr], y_tr[m_tr])
        return _add_jitter(clf.predict_proba(X_va)[:, 1], seed=seed)
    return fit_predict


def make_catboost_binary(horizon=5.0, iterations=300, learning_rate=0.03,
                         depth=5, l2_leaf_reg=3.0, seed=42):
    from catboost import CatBoostClassifier

    def fit_predict(X_tr, e_tr, t_tr, X_va):
        y_tr, m_tr = _binary_label_with_horizon(e_tr, t_tr, horizon)
        if m_tr.sum() < 20 or y_tr[m_tr].sum() < 3:
            return _add_jitter(np.zeros(len(X_va)), seed=seed)
        clf = CatBoostClassifier(
            iterations=iterations, learning_rate=learning_rate, depth=depth,
            l2_leaf_reg=l2_leaf_reg, auto_class_weights="Balanced",
            random_seed=seed, verbose=False, allow_writing_files=False)
        clf.fit(X_tr.iloc[m_tr], y_tr[m_tr])
        return _add_jitter(clf.predict_proba(X_va)[:, 1], seed=seed)
    return fit_predict


def make_logreg_binary(horizon=5.0, C=1.0, seed=42):
    def fit_predict(X_tr, e_tr, t_tr, X_va):
        y_tr, m_tr = _binary_label_with_horizon(e_tr, t_tr, horizon)
        if m_tr.sum() < 20 or y_tr[m_tr].sum() < 3:
            return _add_jitter(np.zeros(len(X_va)), seed=seed)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("lr",  LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                       solver="liblinear", random_state=seed))])
        pipe.fit(X_tr.iloc[m_tr], y_tr[m_tr])
        return _add_jitter(pipe.predict_proba(X_va)[:, 1], seed=seed)
    return fit_predict


MODEL_REGISTRY = {
    "coxnet":      lambda **kw: make_coxnet(**kw),
    "rsf":         lambda **kw: make_rsf(**kw),
    "xgb_cox":     lambda **kw: make_xgb_cox(**kw),
    "lgbm_bin":    lambda **kw: make_lgbm_binary(**kw),
    "catboost_bin":lambda **kw: make_catboost_binary(**kw),
    "logreg_bin":  lambda **kw: make_logreg_binary(**kw),
}
