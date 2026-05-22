"""HUD USPS crosswalks beyond ZIP-CBSA.

Extends :mod:`arbok.sources.hud_crosswalk` with the tract/county/state mappings
needed to aggregate or broadcast features between geographies.

All HUD USPS crosswalks live behind the same free-login portal:
    https://www.huduser.gov/portal/datasets/usps_crosswalk.html

`type` codes used by the HUD API endpoint
(https://www.huduser.gov/hudapi/public/usps):

    1 = ZIP-TRACT          5 = ZIP-COUSUB         8 = TRACT-ZIP
    2 = ZIP-COUNTY         6 = ZIP-CONGRESS DIST  9 = COUNTY-ZIP
    3 = ZIP-CBSA           7 = ZIP-CBSA DIV       10 = CBSA-ZIP

For ZIP-to-X mappings, each row has a `RES_RATIO`, `BUS_RATIO`, `OTH_RATIO`,
`TOT_RATIO` — the share of residential / business / other / total addresses
in the ZIP that fall into the target geography. Use RES_RATIO for residential
real-estate work; sum is 1.0 across the rows for a given ZIP.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from arbok.config import RAW

HUD_API = "https://www.huduser.gov/hudapi/public/usps"
HUD_RAW = RAW / "hud"
HUD_RAW.mkdir(parents=True, exist_ok=True)

_TYPE_CODES = {
    "zip_tract": 1,
    "zip_county": 2,
    "zip_cbsa": 3,
    "tract_zip": 8,
    "county_zip": 9,
    "cbsa_zip": 10,
}


_GEOID_TARGET = {
    "zip_tract": "tract",
    "zip_county": "county",
    "zip_cbsa": "cbsa",
    "tract_zip": "zip",
    "county_zip": "zip",
    "cbsa_zip": "zip",
}


def _api_pull(crosswalk: str, year: int | None = None, quarter: int | None = None) -> pd.DataFrame:
    """Pull a HUD USPS crosswalk via the API.

    The API rejects `query=All` combined with explicit year/quarter, so when
    no specific quarter is requested we omit them and accept whatever is the
    current "latest" release. To target a specific historical quarter, use
    the manual Excel workflow in :mod:`arbok.sources.hud_crosswalk`.
    """
    token = os.environ.get("HUD_USPS_TOKEN")
    if not token:
        raise RuntimeError(
            "HUD_USPS_TOKEN not set. Either set it (free signup at "
            "https://www.huduser.gov/portal/dataset/uspszip-api.html) "
            f"or manually download the {crosswalk} workbook into {HUD_RAW}/"
        )
    type_code = _TYPE_CODES[crosswalk]
    params: dict[str, str | int] = {"type": type_code, "query": "All"}
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(HUD_API, params=params, headers=headers, timeout=600)
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("data", {}).get("results", [])
    if not rows:
        raise RuntimeError(f"HUD API returned no rows for {crosswalk}")
    df = pd.DataFrame(rows)
    # API returns the non-zip geography under `geoid` regardless of type.
    target = _GEOID_TARGET[crosswalk]
    if "geoid" in df.columns and target not in df.columns:
        df = df.rename(columns={"geoid": target})
    return df


def _cache_or_pull(crosswalk: str, year: int | None, quarter: int | None) -> pd.DataFrame:
    tag = f"{year}Q{quarter}" if year and quarter else "latest"
    cache = HUD_RAW / f"{crosswalk}_{tag}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = _api_pull(crosswalk)
    df.to_parquet(cache, index=False)
    return df


def _normalize(df: pd.DataFrame, key_a: str, key_b: str) -> pd.DataFrame:
    """Lowercase column names, coerce keys to padded strings, keep ratio columns."""
    df = df.rename(columns=str.lower).copy()
    rename = {"zip": "zip", "tract": "tract", "county": "county", "cbsa": "cbsa"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "zip" in df.columns:
        df["zip"] = df["zip"].astype(str).str.zfill(5)
    if "tract" in df.columns:
        df["tract"] = df["tract"].astype(str).str.zfill(11)
    if "county" in df.columns:
        df["county"] = df["county"].astype(str).str.zfill(5)
    if "cbsa" in df.columns:
        df["cbsa"] = df["cbsa"].astype(str).str.zfill(5)
    keep = [c for c in (key_a, key_b, "res_ratio", "bus_ratio", "oth_ratio", "tot_ratio") if c in df.columns]
    return df[keep]


def load_zip_tract(year: int, quarter: int) -> pd.DataFrame:
    """Returns columns: zip, tract, res_ratio, bus_ratio, oth_ratio, tot_ratio.

    For a given ZIP, the res_ratios sum to 1.0 across constituent tracts.
    Use res_ratio as the weight when computing zip-level aggregates of tract data.
    """
    return _normalize(_cache_or_pull("zip_tract", year, quarter), "zip", "tract")


def load_zip_county(year: int, quarter: int) -> pd.DataFrame:
    return _normalize(_cache_or_pull("zip_county", year, quarter), "zip", "county")


def load_zip_state(year: int, quarter: int) -> pd.DataFrame:
    """Derived from zip_county: state FIPS = first 2 chars of 5-digit county FIPS."""
    zc = load_zip_county(year, quarter)
    zc = zc.copy()
    zc["state"] = zc["county"].str[:2]
    return (
        zc.groupby(["zip", "state"], as_index=False)[["res_ratio", "bus_ratio", "oth_ratio", "tot_ratio"]]
        .sum()
    )


def dominant_county_per_zip(zip_county: pd.DataFrame) -> pd.DataFrame:
    """One row per ZIP — the county with the largest residential share."""
    idx = zip_county.groupby("zip")["res_ratio"].idxmax()
    return zip_county.loc[idx, ["zip", "county"]].reset_index(drop=True)


def aggregate_tract_to_zip(
    tract_df: pd.DataFrame,
    zip_tract: pd.DataFrame,
    value_cols: list[str],
    weight_col: str = "res_ratio",
    tract_col: str = "tract",
) -> pd.DataFrame:
    """Weighted-mean aggregation of tract-level features up to zip level.

    `tract_df` must contain `tract_col` (11-digit GEOID, zero-padded) and `value_cols`.
    Returns one row per zip with weighted means of `value_cols`.
    """
    df = tract_df.copy()
    df[tract_col] = df[tract_col].astype(str).str.zfill(11)
    merged = zip_tract.merge(df, left_on="tract", right_on=tract_col, how="inner")
    out_frames = []
    for col in value_cols:
        # weighted mean, dropping rows where the value is NaN so weights renormalize
        valid = merged.dropna(subset=[col])
        if valid.empty:
            continue
        agg = (
            valid.groupby("zip")
            .apply(lambda g, c=col: (g[c] * g[weight_col]).sum() / g[weight_col].sum())
            .rename(col)
            .reset_index()
        )
        out_frames.append(agg)
    if not out_frames:
        return pd.DataFrame(columns=["zip", *value_cols])
    result = out_frames[0]
    for frame in out_frames[1:]:
        result = result.merge(frame, on="zip", how="outer")
    return result


def broadcast_county_to_zip(
    county_df: pd.DataFrame,
    zip_county: pd.DataFrame,
    value_cols: list[str],
    county_col: str = "county",
) -> pd.DataFrame:
    """Broadcast county-level features down to zip via the dominant-county-per-zip rule.

    `county_df` must contain `county_col` (5-digit FIPS) and `value_cols`.
    """
    df = county_df.copy()
    df[county_col] = df[county_col].astype(str).str.zfill(5)
    dom = dominant_county_per_zip(zip_county)
    return dom.merge(df, left_on="county", right_on=county_col, how="left")[["zip", *value_cols]]
