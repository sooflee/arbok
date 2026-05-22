"""LightGBM wrapper with SHAP feature importance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("lightgbm required (uv add lightgbm)") from e


# Shared base params so point and quantile fits stay comparable.
_BASE_LGBM_PARAMS: dict = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
}


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
        **_BASE_LGBM_PARAMS,
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


# ---------------------------------------------------------------------------
# Quantile LightGBM
# ---------------------------------------------------------------------------


def fit_quantile_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 50,
) -> LgbmFit:
    """Fit one quantile LightGBM at level `alpha` (in (0, 1)).

    Uses the same tree-shape / sampling params as `fit_lightgbm` so the
    point and quantile fits remain comparable; only the loss switches to
    `objective='quantile'` with the requested alpha.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    base_params = {
        "objective": "quantile",
        "alpha": float(alpha),
        "metric": "quantile",
        **_BASE_LGBM_PARAMS,
    }
    if params:
        base_params.update(params)

    feat_names = list(X_train.columns)
    train_set = lgb.Dataset(X_train, label=y_train.values, feature_name=feat_names)
    valid_sets = [train_set]
    valid_names = ["train"]
    if X_val is not None and y_val is not None:
        val_set = lgb.Dataset(
            X_val, label=y_val.values, reference=train_set, feature_name=feat_names
        )
        valid_sets.append(val_set)
        valid_names.append("val")

    callbacks = (
        [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        if X_val is not None and y_val is not None
        else []
    )

    booster = lgb.train(
        base_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return LgbmFit(model=booster, feature_names=feat_names)


def fit_lgbm_quantiles(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: Sequence[float] = (0.1, 0.5, 0.9),
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 50,
) -> dict[float, LgbmFit]:
    """Fit one quantile model per alpha. Returns {alpha: LgbmFit}.

    Each model is independent — LightGBM does not support a single joint
    quantile fit, so we train three separate boosters. Quantile crossing
    (e.g. p10 > p50) can occur on out-of-distribution rows; downstream
    callers should clip if monotonicity is required.
    """
    models: dict[float, LgbmFit] = {}
    for alpha in alphas:
        models[float(alpha)] = fit_quantile_lgbm(
            X_train,
            y_train,
            alpha=float(alpha),
            X_val=X_val,
            y_val=y_val,
            params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
    return models


def predict_quantiles(
    models: Mapping[float, LgbmFit],
    X: pd.DataFrame,
    enforce_monotone: bool = True,
) -> pd.DataFrame:
    """Predict from a {alpha: LgbmFit} mapping into a DataFrame with named columns.

    Columns are named `pred_p{int(alpha*100)}` (e.g. `pred_p10`, `pred_p50`,
    `pred_p90`). Index matches `X.index`. When `enforce_monotone=True`,
    cumulative-max across the sorted-alpha columns prevents quantile crossing
    while preserving each alpha's monotone ordering against its neighbors.
    """
    if not models:
        raise ValueError("models is empty")

    sorted_alphas = sorted(models.keys())
    cols: dict[str, np.ndarray] = {}
    for alpha in sorted_alphas:
        name = f"pred_p{int(round(float(alpha) * 100))}"
        cols[name] = models[alpha].predict(X)

    out = pd.DataFrame(cols, index=X.index)
    if enforce_monotone and out.shape[1] > 1:
        # Walk lower -> upper alphas, clamp each column to be >= previous.
        ordered = list(out.columns)
        for i in range(1, len(ordered)):
            prev, curr = ordered[i - 1], ordered[i]
            out[curr] = np.maximum(out[curr].values, out[prev].values)
    return out
