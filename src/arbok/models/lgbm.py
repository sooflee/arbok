"""LightGBM wrapper with SHAP feature importance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("lightgbm required (uv add lightgbm)") from e


@dataclass
class LgbmFit:
    model: lgb.Booster
    feature_names: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])

    def gain_importance(self) -> pd.DataFrame:
        gain = self.model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self.feature_names, "gain": gain})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )

    def shap_top_k(self, X: pd.DataFrame, k: int = 20, sample: int = 50_000) -> pd.DataFrame:
        """Mean |SHAP| over a sampled subset of X. Sample is fine — SHAP is slow."""
        import shap
        Xs = X.sample(min(sample, len(X)), random_state=0) if len(X) > sample else X
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(Xs[self.feature_names])
        if isinstance(sv, list):
            sv = sv[0]
        mean_abs = np.abs(sv).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 50,
) -> LgbmFit:
    """LightGBM with reasonable defaults for tabular regression on big panels."""
    base_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
    }
    if params:
        base_params.update(params)

    feat_names = list(X_train.columns)
    train_set = lgb.Dataset(X_train, label=y_train.values, feature_name=feat_names)
    valid_sets = [train_set]
    valid_names = ["train"]
    if X_val is not None and y_val is not None:
        val_set = lgb.Dataset(X_val, label=y_val.values, reference=train_set, feature_name=feat_names)
        valid_sets.append(val_set)
        valid_names.append("val")

    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if X_val is not None else []

    booster = lgb.train(
        base_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return LgbmFit(model=booster, feature_names=feat_names)
