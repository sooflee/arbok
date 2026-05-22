"""Cheap baselines the real models must beat.

A model that doesn't beat metro-mean is not learning anything spatial; one that
doesn't beat hedonic regression is not learning anything beyond demographics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BaselineResult:
    name: str
    y_pred: np.ndarray


def metro_mean_predictor(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> BaselineResult:
    """Predict each test zip's target as the in-sample mean of its CBSA's targets.

    train and test must have `cbsa` and `target` columns.
    """
    cbsa_means = train.groupby("cbsa")[target].mean()
    overall = train[target].mean()
    y_pred = test["cbsa"].map(cbsa_means).fillna(overall).to_numpy(dtype=float)
    return BaselineResult("metro_mean", y_pred)


def overall_mean_predictor(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> BaselineResult:
    """Predict the global train-set mean for every test row (cheapest baseline)."""
    return BaselineResult("overall_mean", np.full(len(test), train[target].mean()))


def hedonic_predictor(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> BaselineResult:
    """OLS on ACS-only features (income / age / education / home value / etc.).

    Demographic predictors are the textbook baseline for residential RE; if our
    fancy model doesn't beat this, the macro/amenity/climate features aren't earning
    their seat.
    """
    from sklearn.linear_model import Ridge

    acs_cols = [c for c in X_train.columns if c.startswith("acs__")]
    if not acs_cols:
        return BaselineResult("hedonic_acs", np.full(len(X_test), float(y_train.mean())))
    # Ridge (not OLS) — ACS columns are highly collinear (pop_25_29 + pop_30_34 ...)
    # so OLS coefficients explode. alpha=1 is conservative.
    m = Ridge(alpha=1.0)
    m.fit(X_train[acs_cols], y_train.values)
    pred = m.predict(X_test[acs_cols])
    # Clip to plausible-return range (matches training y range with small headroom).
    lo, hi = y_train.quantile(0.001), y_train.quantile(0.999)
    return BaselineResult("hedonic_acs", np.clip(pred, lo, hi))
