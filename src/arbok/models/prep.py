"""Feature-store preparation: feature selection, NaN handling, split helpers.

Modeling-time concerns intentionally separated from feature-engineering: by the
time we get to this module the feature store is a wide (zip, year_month, ...)
DataFrame, and our job is to turn it into (X_train, y_train, X_test, y_test)
ready for sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arbok.config import HORIZONS_MONTHS

# Columns we always exclude from X regardless of source.
NON_FEATURE_PREFIXES = ("is_train_", "is_test_")
NON_FEATURE_EXACT = {
    "zip", "cbsa", "year_month", "zhvi", "zori",
    "split", "fwd_6m", "fwd_1y", "fwd_3y", "fwd_5y", "fwd_10y",
}

# Categorical / textual / identifier cols we drop (could one-hot later if useful).
CATEGORICAL_TO_DROP = {
    "bps__cbsa_name",
    "acs__state_x", "acs__state_y", "acs__state",
    "nri__state", "nri__state_fips", "nri__county", "nri__county_fips",
    "rdc__months_supply",  # known empty at zip tier
}


@dataclass(frozen=True)
class Split:
    name: str            # "pre_2008" or "post_2012"
    train_mask_col: str  # "is_train_pre_2008"
    test_mask_col: str   # "is_test_pre_2008"


SPLITS_AVAIL = (
    Split("pre_2008",  "is_train_pre_2008",  "is_test_pre_2008"),
    Split("post_2012", "is_train_post_2012", "is_test_post_2012"),
)


def feature_columns(fs: pd.DataFrame, min_coverage: float = 0.10) -> list[str]:
    """Return the column list to use as X, filtering low-coverage + categorical."""
    cols = []
    for c in fs.columns:
        if c in NON_FEATURE_EXACT or c in CATEGORICAL_TO_DROP:
            continue
        if any(c.startswith(p) for p in NON_FEATURE_PREFIXES):
            continue
        if fs[c].dtype == object or pd.api.types.is_datetime64_any_dtype(fs[c]):
            continue
        if fs[c].notna().mean() < min_coverage:
            continue
        cols.append(c)
    return cols


def build_xy(
    fs: pd.DataFrame,
    target: str,
    split: Split,
    min_coverage: float = 0.10,
    impute: str | None = "median",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Construct (X_train, y_train, X_test, y_test, feature_names) for one split.

    Drops rows where target is NaN. Filters features by min coverage on the
    TRAIN side only (so the test set isn't peeked at). Imputes per-feature
    medians from train statistics.
    """
    if target not in fs.columns:
        raise ValueError(f"{target} not in feature store")

    keep = fs[target].notna()
    train_mask = fs[split.train_mask_col].fillna(False) & keep
    test_mask = fs[split.test_mask_col].fillna(False) & keep

    train = fs.loc[train_mask].copy()
    test = fs.loc[test_mask].copy()

    feats = feature_columns(train, min_coverage=min_coverage)

    X_train = train[feats]
    y_train = train[target].astype(float)
    X_test = test[feats]
    y_test = test[target].astype(float)

    if impute == "median":
        med = X_train.median(numeric_only=True)
        X_train = X_train.fillna(med)
        X_test = X_test.fillna(med)
        # Drop columns whose train median is still NaN (column was all-NaN in train).
        still_na = X_train.columns[X_train.isna().any()].tolist()
        if still_na:
            X_train = X_train.drop(columns=still_na)
            X_test = X_test.drop(columns=still_na, errors="ignore")
            feats = [f for f in feats if f not in still_na]
        # Also drop near-zero-variance columns (constant after imputation = useless).
        nunique = X_train.nunique(dropna=False)
        zero_var = nunique[nunique <= 1].index.tolist()
        if zero_var:
            X_train = X_train.drop(columns=zero_var)
            X_test = X_test.drop(columns=zero_var, errors="ignore")
            feats = [f for f in feats if f not in zero_var]

    return X_train, y_train, X_test, y_test, feats


def spatial_cv_folds(
    cbsa_series: pd.Series, n_folds: int = 5, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold-out-CBSA cross-validation. Returns list of (train_idx, val_idx) tuples.

    Each fold's validation set is one block of CBSAs (no leakage from neighboring
    zips inside the same CBSA into both sides).
    """
    cbsas = cbsa_series.unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(cbsas)
    chunks = np.array_split(cbsas, n_folds)
    pos = pd.Series(np.arange(len(cbsa_series)), index=cbsa_series.index)
    folds = []
    for chunk in chunks:
        val_mask = cbsa_series.isin(chunk)
        val_idx = pos[val_mask].values
        train_idx = pos[~val_mask].values
        folds.append((train_idx, val_idx))
    return folds
