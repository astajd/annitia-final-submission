"""Shared preprocessing for tabular survival/binary models.

We fit a simple median-impute + standardize pipeline on the *training fold*
only, then apply it to validation/test. Categorical columns are detected as
``object`` or ``category`` dtype; we one-hot encode them with ``handle_unknown=ignore``.

Tree models can usually handle NaN themselves, but we still produce a clean
matrix so the same X representation works for every model in the registry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FittedPreprocessor:
    pipeline: ColumnTransformer
    feature_names: list[str]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Reorder/fill missing columns so the transformer always sees the same schema.
        X = X.reindex(columns=self.feature_names_input)
        out = self.pipeline.transform(X)
        # Failsafe: any NaN produced by zero-variance scalers or unseen-only
        # columns gets zeroed before estimators see it. The model wrappers all
        # expect finite arrays.
        if np.isnan(out).any():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    feature_names_input: list[str] = None  # set after fit


def fit_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> FittedPreprocessor:
    """Fit median-impute + (optional) standardize + one-hot encode.

    Drops columns that are all-NaN or constant after imputation so downstream
    estimators (esp. CoxNet and StandardScaler) never see NaN/zero-variance
    inputs.
    """
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Drop numeric columns that have no observed values (median is undefined).
    num_cols = [c for c in num_cols if X[c].notna().any()]
    # Drop categorical columns with zero non-null values too.
    cat_cols = [c for c in cat_cols if X[c].notna().any()]

    num_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=False))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler(with_mean=True, with_std=True)))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    ct.fit(X)

    feature_names = []
    for name, _, cols in transformers:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe: OneHotEncoder = ct.named_transformers_["cat"].named_steps["ohe"]
            feature_names.extend(ohe.get_feature_names_out(cols).tolist())

    fp = FittedPreprocessor(pipeline=ct, feature_names=feature_names)
    fp.feature_names_input = list(X.columns)
    return fp


def fill_for_tree(X: pd.DataFrame) -> pd.DataFrame:
    """Cheap NaN-tolerant transform for tree models that handle NaN natively.

    Just one-hot encodes categoricals; numeric NaNs are kept.
    """
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    if not cat_cols:
        return X.astype(float, errors="ignore")
    return pd.get_dummies(X, columns=cat_cols, dummy_na=True)
