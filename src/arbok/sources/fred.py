"""FRED macro-data loader for timing / "Money & rates" predictors.

Uses the FRED API if ``FRED_API_KEY`` is in the environment, otherwise falls
back to the public CSV endpoint at ``fredgraph.csv?id=<ID>``. Raw responses
are cached under ``data/raw/fred/`` so re-runs are cheap and offline-safe.

Friendly-name mapping is in :data:`SERIES`. Verified IDs (HTTP 200 against
the public CSV endpoint on 2026-05-21): all eight current entries. If a
series ID is renamed / retired upstream, :func:`fetch_series` raises with
the offending ID so the caller can drop it from :data:`SERIES`.
"""
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

# Friendly name -> FRED series ID. Add new IDs here; downstream code keys
# off the friendly names so the wide panel stays stable across renames.
SERIES: dict[str, str] = {
    "mortgage_30y": "MORTGAGE30US",      # 30Y fixed-rate mortgage avg, weekly
    "real_rate_10y": "DFII10",           # 10Y TIPS yield, daily
    "m2_sa": "M2SL",                     # M2 money stock, monthly SA
    "treasury_10y": "DGS10",             # 10Y Treasury constant maturity, daily
    "housing_affordability": "FIXHAI",   # NAR housing affordability, monthly
    "case_shiller_us": "CSUSHPISA",      # S&P/Case-Shiller US national, monthly SA
    "lumber": "WPU081",                  # PPI: lumber & wood products, monthly
    "unemployment_us": "UNRATE",         # US unemployment rate, monthly
}

# Rate-class series for which YoY/QoQ deltas are meaningful predictors.
RATE_SERIES: tuple[str, ...] = ("mortgage_30y", "real_rate_10y", "treasury_10y")

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CACHE_DIR = RAW / "fred"
OUT_PATH = PROCESSED / "macro_monthly.parquet"

REQUEST_TIMEOUT = 30


def _cache_path(series_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{series_id}.csv"


def _fetch_via_api(series_id: str, api_key: str) -> pd.DataFrame:
    """Pull observations via the JSON API and normalise to date/value."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    resp = requests.get(FRED_API_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        raise ValueError(f"FRED API returned no observations for {series_id!r}")
    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).reset_index(drop=True)


def _fetch_via_csv(series_id: str) -> pd.DataFrame:
    """Public CSV fallback — no API key required."""
    resp = requests.get(
        FRED_CSV_URL, params={"id": series_id}, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    if not resp.text.lstrip().lower().startswith(("observation_date", "date")):
        raise ValueError(
            f"FRED CSV for {series_id!r} did not return a data table — "
            "the ID may be retired or renamed."
        )
    df = pd.read_csv(StringIO(resp.text))
    # Public CSV uses 'observation_date' (newer) or 'DATE' (older); col 2 is the series.
    date_col = df.columns[0]
    val_col = df.columns[1]
    df = df.rename(columns={date_col: "date", val_col: "value"})
    df["date"] = pd.to_datetime(df["date"])
    # FRED sentinel for missing is '.'
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).reset_index(drop=True)


def fetch_series(series_id: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch one FRED series.

    Uses the JSON API if ``api_key`` (or ``FRED_API_KEY`` env) is provided,
    else the public CSV. Raw response is cached at
    ``data/raw/fred/<series_id>.csv``. Returns columns ``date``, ``value``.
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if key:
        df = _fetch_via_api(series_id, key)
    else:
        df = _fetch_via_csv(series_id)
    # Cache the normalised frame; cheaper than round-tripping raw API JSON.
    df.to_csv(_cache_path(series_id), index=False)
    return df


def _to_monthly(df: pd.DataFrame) -> pd.Series:
    """Last-of-month value, indexed by month start (Period[M] -> Timestamp)."""
    s = df.set_index("date")["value"].sort_index()
    monthly = s.resample("ME").last()
    monthly.index = monthly.index.to_period("M").to_timestamp()
    return monthly


def fetch_all(series: dict[str, str] | None = None) -> pd.DataFrame:
    """Fetch every series and return a wide monthly panel.

    Daily / weekly series are downsampled to monthly via last-of-month value;
    monthly series are passed through. Output is indexed by ``year_month``
    (Timestamp on month start) with one column per friendly name.
    """
    series = series or SERIES
    cols: dict[str, pd.Series] = {}
    for name, sid in series.items():
        raw = fetch_series(sid)
        cols[name] = _to_monthly(raw)
    panel = pd.DataFrame(cols).sort_index()
    panel.index.name = "year_month"
    # Forward-fill across months so daily/weekly series without a stamp in
    # a given month inherit their last observation (e.g. mortgage_30y when
    # the survey skipped the final week of the month).
    panel = panel.ffill()
    return panel


def add_delta_columns(
    df: pd.DataFrame, periods: tuple[int, ...] = (1, 3, 12)
) -> pd.DataFrame:
    """Add ``<col>_d{n}m`` first-difference columns for rate-class series.

    Differences (not pct-change) since the columns are already in percent
    units. Only :data:`RATE_SERIES` that are present in ``df`` are touched.
    """
    out = df.copy()
    for col in RATE_SERIES:
        if col not in out.columns:
            continue
        for n in periods:
            out[f"{col}_d{n}m"] = out[col].diff(n)
    return out


def build_and_save() -> Path:
    """Fetch everything and persist to ``data/processed/macro_monthly.parquet``."""
    panel = fetch_all()
    panel = add_delta_columns(panel)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH)
    return OUT_PATH


def load_macro_monthly() -> pd.DataFrame:
    """Read the persisted monthly macro panel."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} not found — run build_and_save() first."
        )
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    out = build_and_save()
    print(f"wrote {out}")
    print(load_macro_monthly().tail())
