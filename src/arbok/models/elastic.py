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
        return self.model.predict(self.scaler.transform(X[self.feature_names]))

    def coef_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.feature_names, "coef_std": self.model.coef_}
        ).sort_values("coef_std", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def fit_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1e-3,
    l1_ratio: float = 0.5,
    max_iter: int = 5_000,
    random_state: int = 0,
) -> ElasticFit:
    """Fit ElasticNet on standardized features so coefficients are comparable.

    Defaults: tiny alpha (data is large + most features have non-zero info), 50/50
    L1/L2. Tune via spatial CV in a sweep wrapper if desired.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    m = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state,
    )
    m.fit(X_scaled, y_train.values)
    return ElasticFit(model=m, scaler=scaler, feature_names=list(X_train.columns))
