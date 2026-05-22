"""EPA Air Quality System (AQS) annual summary loader.

Pulls pre-generated annual concentration files (one row per monitor x parameter
x pollutant-standard) and aggregates to county-year. No API key needed — the
bulk CSVs at ``aqs.epa.gov/aqsweb/airdata/`` are public.

Parameters retained:
- ``88101`` PM2.5 - Local Conditions (annual arithmetic mean, ug/m3)
- ``44201`` Ozone (annual 4th-highest daily 8-hour max, ppm)

Pitfalls:
- A monitor has multiple rows per year, one per ``Pollutant Standard`` /
  ``Sample Duration``. We filter to a single canonical standard per parameter
  (PM25 Annual NAAQS for PM2.5, Ozone 8-hour 2015 NAAQS for Ozone) to avoid
  double-counting.
- POC (Parameter Occurrence Code) means one site can have 2-3 collocated
  instruments. We average across POCs at the site level before rolling up to
  county, so a site with three PM2.5 monitors does not outweigh a site with one.
- ``County Code == '000'`` rows are non-county (e.g. Canadian sites or unknown);
  dropped.
- EPA revises late data; re-download the year's zip to refresh (delete the cache).

Ref: https://aqs.epa.gov/aqsweb/airdata/download_files.html
"""
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

BULK_URL = "https://aqs.epa.gov/aqsweb/airdata/annual_conc_by_monitor_{year}.zip"
CACHE_DIR = RAW / "epa_aqs"
OUT_PATH = PROCESSED / "epa_aqs_county_year.parquet"

REQUEST_TIMEOUT = 180

# Parameter code -> (friendly name, canonical Pollutant Standard, value column,
# output column). For PM2.5 we take the annual arithmetic mean; for Ozone we
# take the 4th-highest daily 8-hour max (the basis of the NAAQS).
PARAM_SPEC: dict[str, dict[str, str]] = {
    "PM2.5": {
        "code": "88101",
        "pollutant_standard": "PM25 Annual 2012",
        "value_col": "Arithmetic Mean",
        "out_col": "pm25_annual_mean_ugm3",
    },
    "Ozone": {
        "code": "44201",
        "pollutant_standard": "Ozone 8-hour 2015",
        "value_col": "4th Max Value",
        "out_col": "ozone_annual_4thmax_ppm",
    },
}

# Columns we keep from the raw CSV — drops ~40 cols (addresses, percentiles,
# certification flags) we don't need for county rollups.
KEEP_COLS = [
    "State Code", "County Code", "Site Num", "Parameter Code", "POC",
    "Parameter Name", "Pollutant Standard", "Year",
    "Arithmetic Mean", "4th Max Value",
]


def _cache_path(year: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"annual_conc_by_monitor_{year}.zip"


def fetch_annual_summary(year: int) -> pd.DataFrame:
    """Download and parse the annual concentration file for ``year``.

    Cached at ``data/raw/epa_aqs/annual_conc_by_monitor_<year>.zip``.
    Returns the raw monitor-level frame filtered to PM2.5 + Ozone rows only,
    with state/county codes kept as zero-padded strings.
    """
    cache = _cache_path(year)
    if not cache.exists():
        url = BULK_URL.format(year=year)
        print(f"[epa_aqs] downloading {url}")
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
        print(f"[epa_aqs] cached {len(resp.content) / 1e6:.1f} MB to {cache}")

    keep_codes = {spec["code"] for spec in PARAM_SPEC.values()}
    with zipfile.ZipFile(cache) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            df = pd.read_csv(
                fh,
                usecols=KEEP_COLS,
                dtype={
                    "State Code": "string",
                    "County Code": "string",
                    "Site Num": "string",
                    "Parameter Code": "string",
                    "Pollutant Standard": "string",
                },
                low_memory=False,
            )
    df = df[df["Parameter Code"].isin(keep_codes)].copy()
    # State Code is sometimes alpha ("CC" = Canada). Drop non-numeric rows.
    df = df[df["State Code"].str.match(r"^\d+$", na=False)]
    df["State Code"] = df["State Code"].str.zfill(2)
    df["County Code"] = df["County Code"].str.zfill(3)
    df = df[df["County Code"] != "000"]
    return df.reset_index(drop=True)


def aggregate_per_county(
    df: pd.DataFrame, parameters: tuple[str, ...] = ("PM2.5", "Ozone")
) -> pd.DataFrame:
    """Roll monitor-level annual values up to county-year.

    Two-stage average: collocated POCs at the same site are averaged first, so
    a site with three PM2.5 instruments counts once; then site values are
    averaged within (state, county). Only the canonical Pollutant Standard per
    parameter is retained (see :data:`PARAM_SPEC`).
    """
    pieces: list[pd.DataFrame] = []
    for name in parameters:
        spec = PARAM_SPEC[name]
        sub = df[
            (df["Parameter Code"] == spec["code"])
            & (df["Pollutant Standard"] == spec["pollutant_standard"])
        ].copy()
        if sub.empty:
            continue
        sub["value"] = pd.to_numeric(sub[spec["value_col"]], errors="coerce")
        sub = sub.dropna(subset=["value"])

        site_keys = ["State Code", "County Code", "Site Num", "Year"]
        site_means = sub.groupby(site_keys, as_index=False)["value"].mean()

        county_keys = ["State Code", "County Code", "Year"]
        county_means = (
            site_means.groupby(county_keys, as_index=False)["value"]
            .mean()
            .rename(columns={"value": spec["out_col"]})
        )
        pieces.append(county_means)

    if not pieces:
        return pd.DataFrame(
            columns=["state_fips", "county_fips", "area_fips", "year"]
            + [PARAM_SPEC[p]["out_col"] for p in parameters]
        )

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=["State Code", "County Code", "Year"], how="outer")

    out = merged.rename(
        columns={"State Code": "state_fips", "County Code": "county_fips", "Year": "year"}
    )
    out["area_fips"] = out["state_fips"] + out["county_fips"]
    out["year"] = out["year"].astype(int)
    cols = ["state_fips", "county_fips", "area_fips", "year"] + [
        PARAM_SPEC[p]["out_col"] for p in parameters if PARAM_SPEC[p]["out_col"] in out.columns
    ]
    return out[cols].sort_values(["area_fips", "year"]).reset_index(drop=True)


def build_and_save(years: list[int]) -> Path:
    """Fetch each year, aggregate to county, concatenate, write parquet."""
    frames = [aggregate_per_county(fetch_annual_summary(y)) for y in years]
    panel = pd.concat(frames, ignore_index=True).sort_values(["area_fips", "year"])
    panel = panel.drop_duplicates(["area_fips", "year"], keep="last").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"[epa_aqs] wrote {len(panel):,} county-year rows to {OUT_PATH}")
    return OUT_PATH


def load_epa_aqs() -> pd.DataFrame:
    """Convenience loader: build the parquet if missing, then read it."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} not found — run build_and_save([years]) first."
        )
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    out = build_and_save([2023])
    print(load_epa_aqs().head())
