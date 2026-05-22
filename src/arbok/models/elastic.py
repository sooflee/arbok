"""ElasticNet wrapper. Interpretable per-feature coefficients."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import QuantileTransformer


@dataclass
class ElasticFit:
    model: ElasticNet
    scaler: QuantileTransformer
    feature_names: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self.model.predict(self.scaler.transform(X[self.feature_names]))
        lo = getattr(self, "y_lo", None)
        hi = getattr(self, "y_hi", None)
        if lo is not None and hi is not None:
            pred = np.clip(pred, lo, hi)
        return pred

    def coef_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.feature_names, "coef_std": self.model.coef_}
        ).sort_values("coef_std", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def fit_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    max_iter: int = 5_000,
    random_state: int = 0,
) -> ElasticFit:
    """Fit ElasticNet on quantile-transformed features so coefficients are comparable.

    Why QuantileTransformer, not StandardScaler:
        Some features (notably the ACS pop_* age-bin columns) have a unit
        change between training-era vintages and test-era vintages — pre-2018
        rows are pop shares (mean ~7), post-2018 rows are raw counts (mean
        ~1200). StandardScaler fit on train-era stats then applied to test
        produces scaled values >10,000 in absolute terms, and a linear model's
        predictions blow up by hundreds of percent. QuantileTransformer maps
        every feature to a percentile rank against the train distribution, so
        out-of-distribution test values get clamped to the train edges and
        the model stays in-distribution. This is the same property tree
        models get for free; we have to add it to linear models explicitly.

    alpha=0.01 — strong enough to keep coefficients honest after the rank
    transform. Lower alpha works too (test RMSE is similar) but produces
    more nonzero coefficients with marginal signal.
    """
    scaler = QuantileTransformer(
        n_quantiles=min(1000, max(10, len(X_train) // 100)),
        output_distribution="normal",
        subsample=200_000,
        random_state=random_state,
    )
    X_scaled = scaler.fit_transform(X_train)
    m = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state,
    )
    m.fit(X_scaled, y_train.values)
    fit = ElasticFit(model=m, scaler=scaler, feature_names=list(X_train.columns))
    # Stash y-range for prediction clipping downstream.
    fit.y_lo = float(y_train.quantile(0.001))  # type: ignore[attr-defined]
    fit.y_hi = float(y_train.quantile(0.999))  # type: ignore[attr-defined]
    return fit
