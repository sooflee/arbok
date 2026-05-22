"""BLS Local Area Unemployment Statistics (LAUS) loader, county-month.

LAUS publishes monthly labor-force statistics for ~8,300 sub-state areas
(states, MSAs, counties, cities >=25K). For each county and month it gives
four core measures: unemployment rate (percent, 1 decimal), unemployment
count, employment count, and labor force (employed + unemployed).

County data start 1990-01 and update monthly with a ~1-month lag. The April
annual benchmark revision rewrites the prior ~5 years; cache by source file
date and re-fetch if you need the latest revision. Public landing:
https://www.bls.gov/lau/

We pull the monthly time-series flat file (TSV, ~335 MB, all counties, all
measures, 1990-present) once and cache it locally, then filter by year:

    https://download.bls.gov/pub/time.series/la/la.data.64.County

Schema (`la.txt`): 5 tab-separated columns — `series_id`, `year`, `period`,
`value`, `footnote_codes`. The 20-char `series_id` packs everything we need:

    [0:2)   survey  = "LA"
    [2:3)   seasonal = "U" (NSA) — county series are NSA-only
    [3:5)   area-type prefix "CN" (counties; la.area_type code F)
    [5:10)  5-digit area FIPS (state+county)
    [10:18) padding zeros
    [18:20) measure code: 03 rate, 04 unemp, 05 emp, 06 labor force

`period` is "M01".."M12" for months and "M13" for the annual average; we
drop M13. `value` is right-padded whitespace and stores floats; rates carry
one decimal, counts are integers. Missing cells carry footnote "N" and a
blank value — coerced to NA. Other footnotes per `la.footnote`: "P"
preliminary, "R"/"T" revision flags, "X" unavailable due to a federal
funding lapse, "G" 11-month annual mean (Oct-2025 shutdown).

BLS Akamai blocks requests without a contact email in the User-Agent; we
set one. See https://www.bls.gov/bls/pss.htm for the usage policy.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

LAUS_DIR = RAW / "bls_laus"
LAUS_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_DATA_URL = "https://download.bls.gov/pub/time.series/la/la.data.64.County"
COUNTY_DATA_FILE = LAUS_DIR / "la.data.64.County.tsv"

# BLS requires a contact-email User-Agent; bare browser UAs get a 403.
USER_AGENT = "arbok-research/0.1 (contact: bensonw.dev@gmail.com)"
REQUEST_TIMEOUT = 600

# measure_code -> output column name (see la.measure)
_MEASURE_COLS: dict[str, str] = {
    "03": "unemployment_rate",
    "04": "unemployed",
    "05": "employed",
    "06": "labor_force",
}
_OUT_PATH = PROCESSED / "laus_county_month.parquet"


def _download_county_file() -> Path:
    """Download the ~335 MB county TSV once; cached locally."""
    if COUNTY_DATA_FILE.exists() and COUNTY_DATA_FILE.stat().st_size > 0:
        return COUNTY_DATA_FILE
    print(f"[bls_laus] downloading {COUNTY_DATA_URL} -> {COUNTY_DATA_FILE}")
    headers = {"User-Agent": USER_AGENT}
    with requests.get(COUNTY_DATA_URL, headers=headers, stream=True, timeout=REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        with COUNTY_DATA_FILE.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    return COUNTY_DATA_FILE


def fetch_laus_county(years: list[int]) -> pd.DataFrame:
    """Return county-month LAUS observations for the requested years.

    Output (long-form, one row per county-month):

        state_fips, county_fips, area_fips, year_month,
        unemployment_rate, labor_force, employed, unemployed

    `year_month` is month-start Timestamp. Annual-average rows (period=M13)
    are dropped. Cells that LAUS suppressed (footnote "N" / blank value)
    surface as NaN so downstream YoY math does not divide by zero.
    """
    if not years:
        raise ValueError("years must be non-empty")
    year_set = {int(y) for y in years}

    path = _download_county_file()

    # Read once (~7M rows after dropna of M13) — keeping all measures in one
    # pass is faster than four scans. dtype tightened to keep memory ~150 MB.
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=1_000_000,
        na_filter=False,
    ):
        # Source columns AND values are whitespace-padded; normalise both.
        chunk.columns = chunk.columns.str.strip()
        chunk["series_id"] = chunk["series_id"].str.strip()
        chunk["period"] = chunk["period"].str.strip()
        chunk["year"] = chunk["year"].astype(int)
        # Filter early: requested years, monthly only, county-measure codes only.
        mask = chunk["year"].isin(year_set) & (chunk["period"] != "M13")
        if not mask.any():
            continue
        chunk = chunk.loc[mask].copy()
        measure = chunk["series_id"].str[18:20]
        chunk = chunk.loc[measure.isin(_MEASURE_COLS)].copy()
        if chunk.empty:
            continue
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError(f"No LAUS county observations found for years={sorted(year_set)}")
    df = pd.concat(chunks, ignore_index=True)

    # Decode series_id. County series are all NSA (seasonal char "U"); we
    # don't expose it since there's no choice.
    sid = df["series_id"]
    df["state_fips"] = sid.str[5:7]
    df["county_fips"] = sid.str[7:10]
    df["area_fips"] = sid.str[5:10]
    df["measure_code"] = sid.str[18:20]

    # year_month = month start. period is M01..M12 here (M13 dropped above).
    month = df["period"].str[1:].astype(int)
    df["year_month"] = pd.to_datetime(
        dict(year=df["year"], month=month, day=1)
    )

    # Value column has padded whitespace and may be blank when footnote="N".
    df["value"] = pd.to_numeric(df["value"].str.strip(), errors="coerce")

    wide = df.pivot_table(
        index=["state_fips", "county_fips", "area_fips", "year_month"],
        columns="measure_code",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns=_MEASURE_COLS)
    # Ensure all four measure columns exist even if a year is missing one.
    for col in _MEASURE_COLS.values():
        if col not in wide.columns:
            wide[col] = pd.NA

    wide.columns.name = None
    return wide[
        [
            "state_fips",
            "county_fips",
            "area_fips",
            "year_month",
            "unemployment_rate",
            "labor_force",
            "employed",
            "unemployed",
        ]
    ].sort_values(["area_fips", "year_month"]).reset_index(drop=True)


def build_and_save(years: list[int]) -> Path:
    """Build the county-month LAUS panel and persist as parquet."""
    panel = fetch_laus_county(years)
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(_OUT_PATH, index=False)
    print(f"[bls_laus] wrote {len(panel):,} rows for years={sorted(set(years))} to {_OUT_PATH}")
    return _OUT_PATH


def load_laus() -> pd.DataFrame:
    """Read the persisted county-month LAUS panel."""
    if not _OUT_PATH.exists():
        raise FileNotFoundError(
            f"{_OUT_PATH} not found. Run `bls_laus.build_and_save([...])` first."
        )
    return pd.read_parquet(_OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    out = build_and_save([2023])
    df = load_laus()
    print(df.head())
    print(f"rows={len(df):,} counties={df['area_fips'].nunique()}")
