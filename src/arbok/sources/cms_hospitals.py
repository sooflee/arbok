"""CMS Hospital Compare — Hospital General Information loader.

Each Medicare-certified hospital receives an overall 1-5 star rating derived
from up to seven measure groups (mortality, safety of care, readmission,
patient experience / HCAHPS, timeliness of care, efficient use of imaging,
and a per-condition outcome bundle). The "General Information" file is the
public roll-up: one row per facility with the headline star plus per-group
"better / same / worse than national" counts and HCAHPS summary.

We aggregate to county-year to use as a "healthcare quality" feature. The
underlying intuition: counties whose hospitals consistently outperform the
national average tend to be richer / better-educated / better-staffed, and
that quality halo correlates with desirability and price growth at the metro
edges. We keep this as a county broadcast onto zip — hospital quality has
roughly metro-level autocorrelation, not zip-level.

Pitfalls:
- ~40% of facilities have ``Hospital overall rating == "Not Available"``
  (specialty hospitals — children's, psychiatric, long-term care, VA, DoD,
  critical-access, rural emergency — are not rated, plus some acute-cares
  with too few measures). They're kept in ``hospital_count`` but excluded
  from ``mean_star_rating`` and ``pct_with_4plus_stars``.
- The CSV ships county *names*, not FIPS. We join to Census's national county
  table to derive ``county_fips`` (5-digit). A handful of hospitals fall in
  territories (PR, GU, etc.) or have mis-spelled county strings (e.g.
  "DEKALB" vs "DE KALB") and end up with NaN ``county_fips``; they're
  dropped from the county rollup.
- No bed-count column in this file — CMS publishes beds in the separate
  "Provider of Services" file (different schema, requires extract). We
  surface ``n_beds`` as NaN here; ``total_beds`` in the county rollup is
  therefore also NaN until the POS loader lands.
- Single snapshot, refreshed monthly. ``Last Modified`` lives in the HTTP
  header; treat the parquet as point-in-time and re-fetch quarterly.

Ref: https://data.cms.gov/provider-data/dataset/xubh-q36u
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

HOSPITAL_GENERAL_URL = (
    "https://data.cms.gov/provider-data/sites/default/files/resources/"
    "893c372430d9d71a1c52737d01239d47_1777413958/Hospital_General_Information.csv"
)
COUNTY_FIPS_URL = (
    "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"
)

CACHE_DIR = RAW / "cms_hospitals"
OUT_PATH = PROCESSED / "cms_hospitals_county.parquet"

REQUEST_TIMEOUT = 180

# CMS Hospital Compare uses these column names verbatim.
_KEEP_COLS = {
    "Facility ID": "facility_id",
    "Facility Name": "name",
    "City/Town": "city",
    "State": "state",
    "ZIP Code": "zip",
    "County/Parish": "county_name",
    "Hospital Type": "hospital_type",
    "Hospital overall rating": "overall_star_rating",
}


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _download(url: str, cache_name: str) -> Path:
    cache = _cache_path(cache_name)
    if not cache.exists():
        print(f"[cms_hospitals] downloading {url}")
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
        print(f"[cms_hospitals] cached {len(resp.content) / 1e6:.2f} MB to {cache}")
    return cache


def _load_county_fips() -> pd.DataFrame:
    """Census national county FIPS table, normalized for name-based joining."""
    cache = _download(COUNTY_FIPS_URL, "national_county2020.txt")
    df = pd.read_csv(
        cache, sep="|",
        dtype={"STATE": "string", "STATEFP": "string", "COUNTYFP": "string"},
    )
    df["county_key"] = (
        df["COUNTYNAME"].str.upper()
        .str.replace(r"\s+(COUNTY|PARISH|BOROUGH|MUNICIPIO|CENSUS AREA|"
                     r"CITY AND BOROUGH|MUNICIPALITY|CITY)$", "", regex=True)
        .str.replace(r"[^\w\s]", "", regex=True).str.strip()
    )
    df["county_fips"] = df["STATEFP"].str.zfill(2) + df["COUNTYFP"].str.zfill(3)
    return df[["STATE", "county_key", "county_fips"]].rename(columns={"STATE": "state"})


def fetch_hospitals() -> pd.DataFrame:
    """Download and parse Hospital General Information.

    One row per Medicare-certified facility. ``overall_star_rating`` is an
    integer 1-5 when CMS publishes one and ``pd.NA`` for unrated facilities
    (the CSV uses the literal string "Not Available"). ``county_fips`` is
    derived via name-join to Census; missing for territory hospitals and a
    few spelling mismatches. ``n_beds`` is always NaN here (see module
    docstring).
    """
    cache = _download(HOSPITAL_GENERAL_URL, "Hospital_General_Information.csv")
    df = pd.read_csv(
        cache,
        usecols=list(_KEEP_COLS),
        dtype={"Facility ID": "string", "ZIP Code": "string", "State": "string"},
        low_memory=False,
    ).rename(columns=_KEEP_COLS)

    df["zip"] = df["zip"].str.zfill(5)
    df["overall_star_rating"] = pd.to_numeric(
        df["overall_star_rating"], errors="coerce"
    ).astype("Int8")
    df["n_beds"] = pd.Series([pd.NA] * len(df), dtype="Int32")

    fips = _load_county_fips()
    df["county_key"] = (
        df["county_name"].astype("string").str.upper()
        .str.replace(r"[^\w\s]", "", regex=True).str.strip()
    )
    df = df.merge(fips, on=["state", "county_key"], how="left")

    out_cols = [
        "facility_id", "name", "city", "state", "zip", "county_fips",
        "overall_star_rating", "n_beds", "hospital_type",
    ]
    return df[out_cols].reset_index(drop=True)


def aggregate_per_county(hospitals_df: pd.DataFrame) -> pd.DataFrame:
    """Per-county rollup with hospital count, mean star, total beds, 4+ share.

    Drops rows without ``county_fips`` (territories + name mismatches).
    ``mean_star_rating`` and ``pct_with_4plus_stars`` use only rated
    facilities; ``hospital_count`` includes all (rated + unrated).
    """
    df = hospitals_df.dropna(subset=["county_fips"]).copy()
    df["state_fips"] = df["county_fips"].str[:2]

    grouped = df.groupby(["state_fips", "county_fips"], as_index=False)
    counts = grouped.size().rename(columns={"size": "hospital_count"})
    beds = grouped["n_beds"].sum(min_count=1).rename(columns={"n_beds": "total_beds"})

    rated = df.dropna(subset=["overall_star_rating"]).copy()
    rated["overall_star_rating"] = rated["overall_star_rating"].astype(float)
    rated_grp = rated.groupby(["state_fips", "county_fips"], as_index=False)
    mean_star = rated_grp["overall_star_rating"].mean().rename(
        columns={"overall_star_rating": "mean_star_rating"}
    )
    pct4 = (
        rated.assign(_is4=(rated["overall_star_rating"] >= 4).astype(float))
        .groupby(["state_fips", "county_fips"], as_index=False)["_is4"]
        .mean()
        .rename(columns={"_is4": "pct_with_4plus_stars"})
    )

    out = counts.merge(beds, on=["state_fips", "county_fips"], how="left")
    out = out.merge(mean_star, on=["state_fips", "county_fips"], how="left")
    out = out.merge(pct4, on=["state_fips", "county_fips"], how="left")
    return out.sort_values("county_fips").reset_index(drop=True)


def build_and_save() -> Path:
    """Fetch, aggregate to county, write parquet."""
    hospitals = fetch_hospitals()
    panel = aggregate_per_county(hospitals)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"[cms_hospitals] wrote {len(panel):,} county rows to {OUT_PATH}")
    return OUT_PATH


def load_cms_hospitals() -> pd.DataFrame:
    """Convenience loader: read the county-level parquet (build first)."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} not found — run build_and_save() first."
        )
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    build_and_save()
    print(load_cms_hospitals().head())
