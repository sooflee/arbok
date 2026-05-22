"""DOE Alternative Fuels Data Center — EV chargers per ZIP.

NREL serves the full national EV-station list via the AFDC API. We use the free
``DEMO_KEY`` (rate-limited but a single full-pull is well within tolerance).
Output: per-ZIP count of public Level-2 + DC fast chargers + count of stations.

Per-zip features:
- ``afdc__station_count``: # of EV stations
- ``afdc__l2_ports``: total Level-2 charging ports
- ``afdc__dcfc_ports``: total DC fast-charge ports
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

AFDC_URL = "https://developer.nrel.gov/api/alt-fuel-stations/v1.json"
AFDC_RAW = RAW / "afdc"
AFDC_RAW.mkdir(parents=True, exist_ok=True)
AFDC_CACHE = AFDC_RAW / "ev_stations.json"


def fetch_ev_stations() -> pd.DataFrame:
    """One-shot pull of all US public EV stations. Cached after first call."""
    if AFDC_CACHE.exists():
        df_raw = pd.read_json(AFDC_CACHE)
    else:
        api_key = os.environ.get("NREL_API_KEY", "DEMO_KEY")
        params = {
            "api_key": api_key,
            "fuel_type": "ELEC",
            "status": "E",       # only operational stations
            "access": "public",
            "country": "US",
            "limit": "all",
        }
        r = requests.get(AFDC_URL, params=params, timeout=120)
        r.raise_for_status()
        payload = r.json()
        AFDC_CACHE.write_text(r.text)
        df_raw = pd.DataFrame(payload.get("fuel_stations", []))
    if df_raw.empty:
        raise RuntimeError("AFDC returned no stations")
    keep = ["zip", "ev_level2_evse_num", "ev_dc_fast_num", "state", "city"]
    keep = [k for k in keep if k in df_raw.columns]
    df = df_raw[keep].copy()
    df["zip"] = df["zip"].astype(str).str.extract(r"(\d{5})", expand=False).str.zfill(5)
    for c in ("ev_level2_evse_num", "ev_dc_fast_num"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def aggregate_per_zip(stations: pd.DataFrame) -> pd.DataFrame:
    g = stations.dropna(subset=["zip"]).groupby("zip")
    out = pd.DataFrame({
        "afdc__station_count": g.size(),
        "afdc__l2_ports": g["ev_level2_evse_num"].sum() if "ev_level2_evse_num" in stations else 0,
        "afdc__dcfc_ports": g["ev_dc_fast_num"].sum() if "ev_dc_fast_num" in stations else 0,
    }).reset_index()
    return out


def build_and_save() -> pd.DataFrame:
    stations = fetch_ev_stations()
    out = aggregate_per_zip(stations)
    path = PROCESSED / "afdc_ev_zip.parquet"
    out.to_parquet(path, index=False)
    print(f"Saved AFDC EV: {out.shape} -> {path}")
    return out


def load_afdc() -> pd.DataFrame:
    path = PROCESSED / "afdc_ev_zip.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_and_save() first.")
    return pd.read_parquet(path)
