"""Evaluation metrics. RMSE, R^2, decile spread, rank correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str = "model") -> dict:
    """Compute RMSE, R^2, Spearman rho, and decile spread."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    rho, _ = spearmanr(yt, yp)

    # Decile spread — the most actionable metric. Sort by predicted, take top/bottom
    # 10%, compare the MEAN realized returns. This is what an investor cares about.
    order = np.argsort(yp)
    n = len(yp)
    bot = order[: n // 10]
    top = order[-(n // 10):]
    decile_spread = float(yt[top].mean() - yt[bot].mean())
    top_decile_mean = float(yt[top].mean())
    bot_decile_mean = float(yt[bot].mean())

    return {
        "model": name,
        "n": int(mask.sum()),
        "rmse": rmse,
        "r2": r2,
        "spearman": float(rho),
        "decile_spread": decile_spread,
        "top_decile_mean": top_decile_mean,
        "bot_decile_mean": bot_decile_mean,
    }


def results_table(results: list[dict]) -> pd.DataFrame:
    """Stack a list of evaluate() dicts into a tidy DataFrame."""
    return pd.DataFrame(results).set_index("model").round(4)
