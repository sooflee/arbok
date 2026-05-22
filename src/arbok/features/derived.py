"""Derived features computed from the panel itself — no new data sources.

These three features are pure transforms of ZHVI and ZORI and are the most
zip-native predictors we can produce without external data. They're especially
valuable for the spatial-demeaned model since the macro/CBSA features get
zeroed out.

- ``derived__yield_gross``: 12 * ZORI / ZHVI. Annualized gross rental yield.
  A cap-rate proxy investors care about; tends to be higher in lower-priced metros.
- ``derived__vol_24m``: rolling 24-month standard deviation of monthly ZHVI return.
  Volatility = downside risk proxy.
- ``derived__drawdown``: current ZHVI / trailing-60mo max ZHVI - 1.
  Negative when zip is below its trailing peak; near-zero at all-time highs.
"""
from __future__ import annotations

import pandas as pd

from arbok.config import PROCESSED


def compute_derived(panel: pd.DataFrame) -> pd.DataFrame:
    """Return (zip, year_month, derived__*) DataFrame ready to join."""
    df = panel[["zip", "year_month", "zhvi", "zori"]].sort_values(["zip", "year_month"]).copy()
    # Convert Period[M] -> month-start Timestamp so downstream merges work.
    if str(df["year_month"].dtype).startswith("period"):
        df["year_month"] = df["year_month"].dt.to_timestamp()

    df["derived__yield_gross"] = 12.0 * df["zori"] / df["zhvi"]

    df["_mret"] = df.groupby("zip", observed=True)["zhvi"].pct_change()
    df["derived__vol_24m"] = (
        df.groupby("zip", observed=True)["_mret"]
        .transform(lambda s: s.rolling(24, min_periods=12).std())
    )

    df["_rmax"] = (
        df.groupby("zip", observed=True)["zhvi"]
        .transform(lambda s: s.rolling(60, min_periods=12).max())
    )
    df["derived__drawdown"] = df["zhvi"] / df["_rmax"] - 1.0

    return df[["zip", "year_month", "derived__yield_gross", "derived__vol_24m", "derived__drawdown"]]


def build_and_save() -> pd.DataFrame:
    panel = pd.read_parquet(PROCESSED / "panel_zip_month.parquet")
    out = compute_derived(panel)
    out.to_parquet(PROCESSED / "derived_zip_month.parquet", index=False)
    print(f"Saved derived features: {out.shape} -> data/processed/derived_zip_month.parquet")
    return out


def load_derived() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "derived_zip_month.parquet")
