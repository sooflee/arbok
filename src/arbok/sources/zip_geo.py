"""ZIP -> (city, state, dominant county) lookup.

Pulls the USPS city/state strings the HUD ZIP-COUNTY endpoint returns alongside
each row. The crosswalk loader proper strips these for modeling use, but they're
necessary for human-readable labels in the report / leaderboard.

Output: ``data/processed/zip_geo.parquet`` — one row per ZIP with the dominant
(highest-residential-share) county and the canonical USPS city name.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from arbok.config import PROCESSED

HUD_API = "https://www.huduser.gov/hudapi/public/usps"
ZIP_GEO_PATH = PROCESSED / "zip_geo.parquet"


def fetch_zip_geo() -> pd.DataFrame:
    """Pull ZIP -> city/state/county via the HUD ZIP_COUNTY API."""
    token = os.environ.get("HUD_USPS_TOKEN")
    if not token:
        raise RuntimeError("HUD_USPS_TOKEN not set in env / .env")
    r = requests.get(
        HUD_API,
        params={"type": 2, "query": "All"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=600,
    )
    r.raise_for_status()
    rows = r.json().get("data", {}).get("results", [])
    df = pd.DataFrame(rows)
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df["res_ratio"] = pd.to_numeric(df["res_ratio"], errors="coerce")
    # Dominant (zip, county) by residential ratio
    df = (
        df.sort_values(["zip", "res_ratio"], ascending=[True, False])
        .drop_duplicates(subset=["zip"], keep="first")
        .rename(columns={"geoid": "county_fips"})
    )
    df = df[["zip", "city", "state", "county_fips"]].copy()
    df["city"] = df["city"].astype(str).str.title()  # BOSTON -> Boston
    df["state"] = df["state"].astype(str).str.upper()
    return df.reset_index(drop=True)


def build_and_save() -> pd.DataFrame:
    df = fetch_zip_geo()
    df.to_parquet(ZIP_GEO_PATH, index=False)
    print(f"Saved zip_geo: {df.shape} -> {ZIP_GEO_PATH}")
    return df


def load_zip_geo() -> pd.DataFrame:
    if not ZIP_GEO_PATH.exists():
        return build_and_save()
    return pd.read_parquet(ZIP_GEO_PATH)
