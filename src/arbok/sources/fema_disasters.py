"""OpenFEMA Disaster Declarations Summaries loader.

One row per (disasterNumber, designated area). Free, no API key.
Pagination: $top capped at 1000; use $inlinecount=allpages for total, then
page with $skip=N and a stable $orderby=id.

Pitfalls: state-level declarations have fips_county_code="000" (dropped at
the county aggregation step). `designated_area` is free-text with "(County)",
"(Parish)", "(Borough)", "(Municipio)" suffixes — we ignore it for joins and
rely on FIPS codes. `disaster_number` repeats across designated areas; count
uniques, not rows.

Ref: https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

API_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"

# Subset of fields we keep. Keeps payload small.
SELECT_FIELDS = [
    "disasterNumber", "declarationDate", "declarationType", "incidentType",
    "state", "fipsStateCode", "fipsCountyCode", "designatedArea",
    "declarationTitle", "incidentBeginDate", "incidentEndDate",
]

FEMA_DIR = RAW / "openfema"
FEMA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = FEMA_DIR / "disasters.csv"
OUTPUT_PATH = PROCESSED / "fema_disasters_county_year.parquet"

PAGE_SIZE = 1000

# Incident-type buckets we explicitly break out as features. The catch-all
# "other" is everything else (Snow, Earthquake, Coastal Storm, Volcano, etc.).
INCIDENT_BUCKETS: dict[str, tuple[str, ...]] = {
    "flood": ("Flood",),
    "fire": ("Fire",),
    "hurricane": ("Hurricane",),
    "tornado": ("Tornado",),
    "severe_storm": ("Severe Storm", "Severe Ice Storm"),
}


def _bucket_for(incident_type: str) -> str:
    for bucket, names in INCIDENT_BUCKETS.items():
        if incident_type in names:
            return bucket
    return "other"


def fetch_disaster_summaries(start_year: int = 2000) -> pd.DataFrame:
    """Pull all FEMA disaster declarations with declarationDate >= start_year.

    Cached to `data/raw/openfema/disasters.csv`. Delete to refresh.
    """
    if CACHE_PATH.exists():
        return pd.read_csv(
            CACHE_PATH,
            dtype={"fips_state_code": "string", "fips_county_code": "string", "state": "string"},
            parse_dates=["declaration_date", "incident_begin_date", "incident_end_date"],
        )

    select = ",".join(SELECT_FIELDS)
    date_filter = f"declarationDate ge '{start_year}-01-01T00:00:00.000Z'"
    base = {"$select": select, "$filter": date_filter, "$format": "json"}

    head = requests.get(API_URL, params={**base, "$top": 1, "$inlinecount": "allpages"}, timeout=60)
    head.raise_for_status()
    total = head.json()["metadata"]["count"]
    print(f"[fema_disasters] fetching {total:,} records since {start_year}")

    rows: list[dict] = []
    for skip in range(0, total, PAGE_SIZE):
        resp = requests.get(
            API_URL,
            params={**base, "$top": PAGE_SIZE, "$skip": skip, "$orderby": "id"},
            timeout=120,
        )
        resp.raise_for_status()
        page = resp.json()["DisasterDeclarationsSummaries"]
        rows.extend(page)
        if (skip // PAGE_SIZE) % 10 == 0:
            print(f"[fema_disasters]   pulled {len(rows):,}/{total:,}")
        if not page:
            break

    df = pd.DataFrame(rows).rename(columns={
        "disasterNumber": "disaster_number", "declarationDate": "declaration_date",
        "declarationType": "declaration_type", "incidentType": "incident_type",
        "fipsStateCode": "fips_state_code", "fipsCountyCode": "fips_county_code",
        "designatedArea": "designated_area", "declarationTitle": "declaration_title",
        "incidentBeginDate": "incident_begin_date", "incidentEndDate": "incident_end_date",
    })
    df["declaration_date"] = pd.to_datetime(df["declaration_date"], errors="coerce", utc=True)
    df["incident_begin_date"] = pd.to_datetime(df["incident_begin_date"], errors="coerce", utc=True)
    df["incident_end_date"] = pd.to_datetime(df["incident_end_date"], errors="coerce", utc=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"[fema_disasters] cached {len(df):,} rows to {CACHE_PATH}")
    return df


def aggregate_per_county_year(df: pd.DataFrame) -> pd.DataFrame:
    """Per (state_fips, county_fips, year): unique-disaster count + bucket breakdown.

    Drops rows where fips_county_code is "000"/missing (state-level decls).
    """
    work = df.copy()
    work["fips_state_code"] = work["fips_state_code"].astype("string").str.zfill(2)
    work["fips_county_code"] = work["fips_county_code"].astype("string").str.zfill(3)
    work = work[work["fips_county_code"].notna() & (work["fips_county_code"] != "000")]
    work["year"] = pd.to_datetime(work["declaration_date"], utc=True, errors="coerce").dt.year
    work = work.dropna(subset=["year", "fips_state_code"])
    work["year"] = work["year"].astype(int)
    work["bucket"] = work["incident_type"].fillna("Unknown").map(_bucket_for)

    # Distinct disaster_number per county-year (a number can re-appear via amendments).
    keys = ["fips_state_code", "fips_county_code", "year"]
    totals = work.groupby(keys)["disaster_number"].nunique().rename("disasters_count").reset_index()
    bucket_counts = (
        work.groupby(keys + ["bucket"])["disaster_number"].nunique().unstack("bucket", fill_value=0)
    )
    bucket_counts.columns = [f"disasters_{c}" for c in bucket_counts.columns]
    bucket_counts = bucket_counts.reset_index()

    out = totals.merge(bucket_counts, on=keys, how="left").fillna(0)
    int_cols = [c for c in out.columns if c.startswith("disasters_")]
    out[int_cols] = out[int_cols].astype("Int64")
    return out


def compute_trailing_window(df: pd.DataFrame, window_years: int = 10) -> pd.DataFrame:
    """Per (state_fips, county_fips, year): trailing `window_years` disaster counts.

    Input: output of `aggregate_per_county_year`. Emits one row per (county,
    year) across the full observed range, with `disasters_{N}yr_*` columns
    counting events in the trailing window (current year inclusive).
    """
    count_cols = [c for c in df.columns if c.startswith("disasters_")]
    keys = ["fips_state_code", "fips_county_code"]

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    all_years = pd.DataFrame({"year": range(year_min, year_max + 1)})
    counties = df[keys].drop_duplicates()
    grid = counties.merge(all_years, how="cross")

    dense = grid.merge(df, on=keys + ["year"], how="left").fillna(0)
    dense[count_cols] = dense[count_cols].astype("int64")
    dense = dense.sort_values(keys + ["year"])

    rolled = (
        dense.groupby(keys, group_keys=False)[count_cols]
        .rolling(window=window_years, min_periods=1)
        .sum()
        .reset_index(drop=True)
    )
    rolled.columns = [c.replace("disasters_", f"disasters_{window_years}yr_") for c in count_cols]
    out = pd.concat([dense[keys + ["year"]].reset_index(drop=True), rolled], axis=1)
    int_cols = [c for c in out.columns if c.startswith("disasters_")]
    out[int_cols] = out[int_cols].astype("Int64")
    return out


def build_and_save(window_years: int = 10) -> Path:
    """End-to-end: fetch (cached) -> aggregate -> trailing window -> parquet."""
    raw = fetch_disaster_summaries()
    agg = aggregate_per_county_year(raw)
    panel = compute_trailing_window(agg, window_years=window_years)
    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"[fema_disasters] wrote {len(panel):,} county-year rows to {OUTPUT_PATH}")
    return OUTPUT_PATH


def load_fema_disasters() -> pd.DataFrame:
    """Convenience loader: build the parquet if missing, then read it."""
    if not OUTPUT_PATH.exists():
        build_and_save()
    return pd.read_parquet(OUTPUT_PATH)
