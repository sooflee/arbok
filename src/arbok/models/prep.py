"""Feature-store preparation: feature selection, NaN handling, split helpers.

Modeling-time concerns intentionally separated from feature-engineering: by the
time we get to this module the feature store is a wide (zip, year_month, ...)
DataFrame, and our job is to turn it into (X_train, y_train, X_test, y_test)
ready for sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

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


def demean_by_cbsa_month(
    fs: pd.DataFrame,
    cols: list[str],
    group_keys: tuple[str, ...] = ("cbsa", "year_month"),
) -> pd.DataFrame:
    """Subtract per-(cbsa, year_month) mean from each col. Returns a new DataFrame.

    Why: features broadcast from national/county/CBSA level (FRED rates, county wages,
    CBSA permits) have the SAME value for every zip in the group, so they zero out and
    will be dropped downstream by the near-zero-variance filter. What's left is the
    zip-specific deviation — what makes this zip different from its metro neighbors.
    Forces the model to learn spatial discrimination instead of time effects.
    """
    out = fs.copy()
    group = fs.groupby(list(group_keys), observed=True)
    for c in cols:
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = fs[c] - group[c].transform("mean")
    return out


def spatial_demean_features(
    fs: pd.DataFrame, target_cols: list[str] | None = None
) -> pd.DataFrame:
    """Wrapper: demean both modeling features AND the listed targets, by (cbsa, year_month)."""
    feats = feature_columns(fs, min_coverage=0.0)  # keep all numerics; near-zero-var dropped later
    cols = list(feats)
    if target_cols:
        cols += [c for c in target_cols if c in fs.columns]
    return demean_by_cbsa_month(fs, cols)


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


def spatial_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    cbsa_col: str = "cbsa",
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Group-by-CBSA k-fold split. Yields ``(train_idx, test_idx)`` integer-position arrays.

    Why: temporal CV catches "the future is different from the past" leakage, but it
    does NOT catch "the zip next door, same metro, same school district, same labor
    market, was in train" leakage. Holding out entire CBSAs forces the model to
    generalize across metros — the harder, more honest test for cross-sectional
    return prediction.

    Implementation: shuffles the unique CBSAs with ``random_state`` for reproducibility,
    then delegates to :class:`sklearn.model_selection.GroupKFold` so partitioning is
    balanced by row count (not by CBSA count). This matters because Phoenix has ~200
    zips and a small CBSA might have 5 — naive equal-CBSA-per-fold partitioning would
    produce wildly unequal test sizes.

    Yields integer positional indices (compatible with ``df.iloc[...]`` and sklearn
    estimator ``fit``/``predict`` flows) regardless of the input frame's index.
    """
    if cbsa_col not in df.columns:
        raise ValueError(f"{cbsa_col!r} not in DataFrame columns")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    groups = df[cbsa_col].to_numpy()
    unique_cbsas = pd.unique(groups)
    if len(unique_cbsas) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique CBSAs, found {len(unique_cbsas)}"
        )

    # Shuffle CBSAs with the given seed, then remap each row's group to its new
    # shuffled position. GroupKFold itself is deterministic and does NOT shuffle,
    # so we have to inject randomness ourselves.
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(unique_cbsas)
    cbsa_to_order = {c: i for i, c in enumerate(perm)}
    shuffled_groups = np.array([cbsa_to_order[c] for c in groups])

    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold needs an X just for length; we don't actually use the features here.
    for train_idx, test_idx in gkf.split(np.zeros(len(df)), groups=shuffled_groups):
        yield train_idx, test_idx


def spatiotemporal_split(
    df: pd.DataFrame,
    train_until: str | pd.Timestamp,
    test_cbsas: list[str] | set[str] | np.ndarray | pd.Series,
    cbsa_col: str = "cbsa",
    time_col: str = "month",
) -> tuple[np.ndarray, np.ndarray]:
    """Single (train_idx, test_idx) split that stresses BOTH time and space.

    - train = rows with ``time_col <= train_until`` AND ``cbsa_col NOT in test_cbsas``
    - test  = rows with ``time_col >  train_until`` AND ``cbsa_col IN test_cbsas``

    This deliberately excludes the "future of training CBSAs" and "history of test
    CBSAs" buckets from both sides — the model never sees the test metros at all
    (spatial holdout) AND must predict a future period (temporal holdout). It is
    the strictest realistic generalization test: a new metro you've never modeled,
    in a future you haven't observed.

    Returns integer positional indices into ``df`` (suitable for ``df.iloc[...]``).
    """
    if cbsa_col not in df.columns:
        raise ValueError(f"{cbsa_col!r} not in DataFrame columns")
    if time_col not in df.columns:
        raise ValueError(f"{time_col!r} not in DataFrame columns")

    cutoff = pd.Timestamp(train_until)
    test_set = set(pd.Index(test_cbsas).astype(str))

    times = pd.to_datetime(df[time_col])
    cbsas = df[cbsa_col].astype(str)
    in_test_cbsas = cbsas.isin(test_set).to_numpy()
    is_past = (times <= cutoff).to_numpy()
    is_future = (times > cutoff).to_numpy()

    train_mask = is_past & ~in_test_cbsas
    test_mask = is_future & in_test_cbsas

    pos = np.arange(len(df))
    return pos[train_mask], pos[test_mask]
