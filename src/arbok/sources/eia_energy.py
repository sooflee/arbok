"""EIA state-level energy prices: residential electricity + regional gasoline.

Pulls from the EIA v2 Open Data API. A free API key is required; register at
https://www.eia.gov/opendata/register.php and export the result as
``EIA_API_KEY`` in your environment. Calls without a key raise a clear
``RuntimeError`` pointing at the signup page.

Endpoint shape (verified against https://www.eia.gov/opendata/documentation.php
on 2026-05-21): legacy v1 series IDs are addressable through the v2 convenience
route ``GET https://api.eia.gov/v2/seriesid/<SERIES_ID>?api_key=...``. The
response body is ``{"response": {"data": [{"period", "value", ...}, ...]}}``.

Two granularities, by design of the underlying series:
* **Electricity** -- monthly per-state series ``ELEC.PRICE.<USPS>-RES.M`` in
  cents per kWh. We emit one row per (state, month).
* **Gasoline** -- weekly retail regular-grade in $/gal. EIA publishes US +
  five PADD aggregates; per-state retail gasoline is *not* free, so we stay at
  PADD granularity and downsample weekly -> monthly mean. Map PADDs to states
  downstream when joining onto the zip-month panel.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

EIA_BASE_URL = "https://api.eia.gov/v2/seriesid"
SIGNUP_URL = "https://www.eia.gov/opendata/register.php"
CACHE_DIR = RAW / "eia"
ELEC_OUT = PROCESSED / "eia_energy_state_month.parquet"
GAS_OUT = PROCESSED / "eia_energy_gas_padd_month.parquet"

REQUEST_TIMEOUT = 60

# USPS -> state FIPS (2-digit, zero-padded). 50 states + DC. Matches the
# convention used by bls_laus / fema_disasters / irs_soi in this package.
STATE_USPS_TO_FIPS: dict[str, str] = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}

# PADD region -> EIA weekly retail-gasoline series ID (regular grade, $/gal).
# Source: EIA series catalog; verified series-ID shape against the v2 docs.
PADD_GAS_SERIES: dict[str, str] = {
    "US":    "PET.EMM_EPMR_PTE_NUS_DPG.W",  # U.S. average
    "PADD1": "PET.EMM_EPMR_PTE_R10_DPG.W",  # East Coast
    "PADD2": "PET.EMM_EPMR_PTE_R20_DPG.W",  # Midwest
    "PADD3": "PET.EMM_EPMR_PTE_R30_DPG.W",  # Gulf Coast
    "PADD4": "PET.EMM_EPMR_PTE_R40_DPG.W",  # Rocky Mountain
    "PADD5": "PET.EMM_EPMR_PTE_R50_DPG.W",  # West Coast
}


def _require_api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError(
            "EIA_API_KEY not set. Register for a free key at "
            f"{SIGNUP_URL} and export EIA_API_KEY=<your-key>."
        )
    return key


def _cache_path(series_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{series_id}.csv"


def fetch_series(series_id: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch one EIA series via the v2 ``/seriesid/`` route.

    Returns columns ``period`` (str, source-native granularity) and
    ``value`` (float). Cached at ``data/raw/eia/<SERIES_ID>.csv``.
    """
    key = api_key or _require_api_key()
    url = f"{EIA_BASE_URL}/{series_id}"
    resp = requests.get(url, params={"api_key": key}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json().get("response", {}).get("data", [])
    if not data:
        raise ValueError(f"EIA returned no rows for {series_id!r}")
    df = pd.DataFrame(data)
    # EIA v2 has migrated from `value` to per-dataset value column names
    # (`price` for electricity, `value` for retail gasoline). Pick whichever
    # numeric value column is present.
    value_col = next((c for c in ("value", "price") if c in df.columns), None)
    if "period" not in df.columns or value_col is None:
        raise ValueError(
            f"EIA response for {series_id!r} missing period/value cols: "
            f"{list(df.columns)}"
        )
    df = df[["period", value_col]].rename(columns={value_col: "value"}).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    df.to_csv(_cache_path(series_id), index=False)
    return df


def _to_month_start(period: pd.Series) -> pd.Series:
    """Coerce EIA ``period`` (YYYY-MM or YYYY-MM-DD) to month-start Timestamp."""
    return pd.to_datetime(period, errors="coerce").dt.to_period("M").dt.to_timestamp()


def fetch_electricity_residential(start_year: int = 2010) -> pd.DataFrame:
    """Monthly avg residential electricity price per state, cents/kWh.

    Columns: ``state`` (USPS), ``state_fips``, ``year_month``,
    ``elec_residential_cents_per_kwh``.
    """
    key = _require_api_key()
    frames: list[pd.DataFrame] = []
    for usps, fips in STATE_USPS_TO_FIPS.items():
        sid = f"ELEC.PRICE.{usps}-RES.M"
        raw = fetch_series(sid, api_key=key)
        frame = pd.DataFrame({
            "state": usps,
            "state_fips": fips,
            "year_month": _to_month_start(raw["period"]),
            "elec_residential_cents_per_kwh": raw["value"].astype(float),
        })
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel["year_month"].dt.year >= start_year]
    return panel.sort_values(["state", "year_month"]).reset_index(drop=True)


def fetch_gasoline_regional(start_year: int = 2010) -> pd.DataFrame:
    """Monthly avg retail regular-grade gasoline per PADD region, $/gal.

    Weekly source series are downsampled to month-mean. Columns: ``padd``,
    ``year_month``, ``gas_regular_usd_per_gal``.
    """
    key = _require_api_key()
    frames: list[pd.DataFrame] = []
    for padd, sid in PADD_GAS_SERIES.items():
        raw = fetch_series(sid, api_key=key)
        weekly = pd.DataFrame({
            "date": pd.to_datetime(raw["period"], errors="coerce"),
            "value": raw["value"].astype(float),
        }).dropna(subset=["date"])
        weekly = weekly.set_index("date").sort_index()
        monthly = weekly["value"].resample("MS").mean()
        frame = pd.DataFrame({
            "padd": padd,
            "year_month": monthly.index,
            "gas_regular_usd_per_gal": monthly.values,
        })
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True).dropna(
        subset=["gas_regular_usd_per_gal"]
    )
    panel = panel[panel["year_month"].dt.year >= start_year]
    return panel.sort_values(["padd", "year_month"]).reset_index(drop=True)


def build_and_save() -> tuple[Path, Path]:
    """Fetch both panels and persist as parquet under ``data/processed/``."""
    elec = fetch_electricity_residential()
    ELEC_OUT.parent.mkdir(parents=True, exist_ok=True)
    elec.to_parquet(ELEC_OUT, index=False)

    gas = fetch_gasoline_regional()
    gas.to_parquet(GAS_OUT, index=False)
    return ELEC_OUT, GAS_OUT


def load_eia_electricity() -> pd.DataFrame:
    """Read the persisted state-month residential-electricity panel."""
    if not ELEC_OUT.exists():
        raise FileNotFoundError(
            f"{ELEC_OUT} not found -- run build_and_save() first."
        )
    return pd.read_parquet(ELEC_OUT)


def load_eia_gasoline() -> pd.DataFrame:
    """Read the persisted PADD-month gasoline panel."""
    if not GAS_OUT.exists():
        raise FileNotFoundError(
            f"{GAS_OUT} not found -- run build_and_save() first."
        )
    return pd.read_parquet(GAS_OUT)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    elec_path, gas_path = build_and_save()
    print(f"wrote {elec_path}")
    print(load_eia_electricity().tail())
    print(f"wrote {gas_path}")
    print(load_eia_gasoline().tail())
