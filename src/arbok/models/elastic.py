"""ElasticNet wrapper. Interpretable per-feature coefficients."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


@dataclass
class ElasticFit:
    model: ElasticNet
    scaler: StandardScaler
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
    """Fit ElasticNet on standardized features so coefficients are comparable.

    alpha=0.01 — a previous run with alpha=1e-3 produced runaway predictions
    (RMSE 0.5 on fwd_1y vs LightGBM 0.07). Strong-magnitude features (M2,
    Case-Shiller) need real regularization even after standardization.
    """
    scaler = StandardScaler()
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
