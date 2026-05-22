"""BEA Regional Economic Accounts loader — county-level annual personal income.

Pulls CAINC1 ("Personal Income Summary: Personal Income, Population, Per Capita
Personal Income") from the BEA Regional dataset JSON API. Per (county, year)
the table has three line codes:

    LineCode 1 — Personal income (thousands of dollars)
    LineCode 2 — Population (persons)
    LineCode 3 — Per capita personal income (dollars)

All three are fetched so callers get income, pop, and the derived per-capita
figure in one frame keyed by 5-digit FIPS.

Requires a free 36-character BEA UserID. Register once at
https://apps.bea.gov/API/signup/ and add ``BEA_API_KEY=<key>`` to .env.
No bulk-CSV fallback: BEA's bulk ZIPs ship the whole Regional dataset per
release; the API is the documented incremental path. Verified 2026-05-21:
``https://apps.bea.gov/api/data`` is live (parses params, JSON-errors on bad
key) and the signup URL is live.

Endpoint shape (one county-year sanity probe)::

    https://apps.bea.gov/api/data?UserID=<KEY>&method=GetData
        &DataSetName=Regional&TableName=CAINC1&LineCode=3
        &GeoFips=06037&Year=2022&ResultFormat=json

Response: ``BEAAPI.Results.Data`` is a list of
``{Code, GeoFips, GeoName, TimePeriod, CL_UNIT, UNIT_MULT, DataValue}``.
``DataValue`` is a string with thousands separators (suppression markers
``"(D)"``, ``"(NA)"``, ``"(L)"`` -> NaN).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

BEA_API_URL = "https://apps.bea.gov/api/data"
SIGNUP_URL = "https://apps.bea.gov/API/signup/"
CACHE_DIR = RAW / "bea_income"
OUT_PATH = PROCESSED / "bea_income_county_year.parquet"

REQUEST_TIMEOUT = 60

# CAINC1 line codes -> output column name.
LINE_CODES: dict[int, str] = {
    1: "total_personal_income_thousands",
    2: "population",
    3: "per_capita_income_usd",
}


def _require_api_key() -> str:
    """Return the BEA API key or raise a pointer to the signup page."""
    key = os.environ.get("BEA_API_KEY")
    if not key:
        raise RuntimeError(
            "BEA_API_KEY is not set. Register a free key at "
            f"{SIGNUP_URL} (takes ~1 minute, no approval), then add "
            "`BEA_API_KEY=<key>` to your .env."
        )
    return key


def _cache_path(year: int, line_code: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"cainc1_line{line_code}_{year}.json"


def _fetch_line_year(year: int, line_code: int, api_key: str) -> pd.DataFrame:
    """Pull one (LineCode, Year) slice across all counties.

    Cached under ``data/raw/bea_income/cainc1_line{code}_{year}.json`` so
    re-runs are offline-friendly. BEA's GeoFips=COUNTY returns ~3.1K rows
    (counties + a handful of state-equivalent aggregates we filter on FIPS).
    """
    cache = _cache_path(year, line_code)
    if cache.exists():
        payload = pd.read_json(cache, typ="series").to_dict()
    else:
        params = {
            "UserID": api_key,
            "method": "GetData",
            "DataSetName": "Regional",
            "TableName": "CAINC1",
            "LineCode": str(line_code),
            "GeoFips": "COUNTY",
            "Year": str(year),
            "ResultFormat": "json",
        }
        resp = requests.get(BEA_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        # Persist raw so we can re-parse without burning quota.
        pd.Series(payload).to_json(cache)

    results = payload.get("BEAAPI", {}).get("Results", {})
    # BEA reports invalid-request errors inside Results.Error rather than HTTP 4xx.
    if isinstance(results, dict) and "Error" in results:
        raise RuntimeError(f"BEA API error: {results['Error']}")
    data = results.get("Data") if isinstance(results, dict) else None
    if not data:
        raise RuntimeError(
            f"BEA returned no Data rows for LineCode={line_code} Year={year}; "
            f"payload keys: {list(results.keys()) if isinstance(results, dict) else type(results)}"
        )

    df = pd.DataFrame(data)
    df = df[["GeoFips", "TimePeriod", "DataValue"]].copy()
    df["area_fips"] = df["GeoFips"].astype(str).str.zfill(5)
    df["year"] = pd.to_numeric(df["TimePeriod"], errors="coerce").astype("Int64")
    # DataValue carries thousands separators and BEA suppression markers
    # ("(NA)", "(D)", "(L)") — strip commas, coerce the rest to NaN.
    cleaned = df["DataValue"].astype(str).str.replace(",", "", regex=False)
    df["value"] = pd.to_numeric(cleaned, errors="coerce")

    col = LINE_CODES[line_code]
    out = df[["area_fips", "year", "value"]].rename(columns={"value": col})
    # Drop non-county aggregates: real county FIPS have nonzero last 3 digits.
    # State totals end in "000" (e.g. "06000" = California). Keep only counties.
    return out[~out["area_fips"].str.endswith("000")].reset_index(drop=True)


def fetch_bea_county_income(years: list[int]) -> pd.DataFrame:
    """Fetch CAINC1 per-county income/population/per-capita for `years`.

    Columns: ``state_fips``, ``county_fips``, ``area_fips`` (5-digit string),
    ``year``, ``per_capita_income_usd``, ``total_personal_income_thousands``,
    ``population``. Suppressed cells (``"(D)"``, ``"(NA)"``) become NaN.
    """
    api_key = _require_api_key()
    per_year: list[pd.DataFrame] = []
    for year in years:
        wide = None
        for line_code in LINE_CODES:
            piece = _fetch_line_year(year, line_code, api_key)
            wide = piece if wide is None else wide.merge(
                piece, on=["area_fips", "year"], how="outer"
            )
        assert wide is not None
        per_year.append(wide)

    panel = pd.concat(per_year, ignore_index=True)
    panel["state_fips"] = panel["area_fips"].str[:2]
    panel["county_fips"] = panel["area_fips"].str[2:]
    return panel[
        [
            "state_fips",
            "county_fips",
            "area_fips",
            "year",
            "per_capita_income_usd",
            "total_personal_income_thousands",
            "population",
        ]
    ].sort_values(["year", "area_fips"]).reset_index(drop=True)


def build_and_save(years: list[int]) -> Path:
    """Build the multi-year county income panel and persist as parquet."""
    panel = fetch_bea_county_income(years)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"[bea_income] wrote {len(panel):,} rows for years={years} to {OUT_PATH}")
    return OUT_PATH


def load_bea_income() -> pd.DataFrame:
    """Read the persisted county-year BEA income panel."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} not found — run build_and_save(years=[...]) first."
        )
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    out = build_and_save([2020, 2021, 2022])
    print(f"wrote {out}")
    print(load_bea_income().head())
