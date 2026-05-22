"""IRS SOI county-to-county migration data loader.

The IRS Statistics of Income (SOI) division publishes annual migration data
derived from year-over-year matched individual income-tax returns. For each
tax-filing year, two files are released at the county level:

  - `countyinflow{YY}{YY+1}.csv`  — one row per (destination, origin) pair.
        Counts returns that filed in `origin` last year and `destination`
        this year. Destination is the "focus" county.
  - `countyoutflow{YY}{YY+1}.csv` — one row per (origin, destination) pair.
        Counts returns that filed in `origin` last year and `destination`
        this year. Origin is the "focus" county.

The two file types describe the same flows but are pre-aggregated by which
county is held fixed, which makes per-county net migration trivial to compute.

Columns (identical schema across years, since ~2011):

    y1_statefips, y1_countyfips  — origin (prior-year) county FIPS
    y2_statefips, y2_countyfips  — destination (current-year) county FIPS
    y1_state                     — origin USPS state code (or aggregate label)
    y1_countyname                — origin county name (or aggregate label)
    n1                           — number of returns (tax filers ~= households)
    n2                           — number of exemptions (~= persons)
    agi                          — adjusted gross income, in thousands USD

Aggregate (non-county) rows have FIPS codes >= 57 / 96-98 and represent
state totals, foreign migration, "same county / different county" splits, etc.
We filter those out in `compute_county_net_migration`.

Public reference: https://www.irs.gov/statistics/soi-tax-stats-migration-data
URL convention: https://www.irs.gov/pub/irs-soi/county{direction}{YY}{YY+1}.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

IRS_SOI_DIR = RAW / "irs_soi"
IRS_SOI_DIR.mkdir(parents=True, exist_ok=True)

INFLOW_URL_TEMPLATE = "https://www.irs.gov/pub/irs-soi/countyinflow{yy1}{yy2}.csv"
OUTFLOW_URL_TEMPLATE = "https://www.irs.gov/pub/irs-soi/countyoutflow{yy1}{yy2}.csv"

# State FIPS > 56 are non-state aggregates (DC=11 and PR=72 are real, but the
# IRS only uses 57-58 for "US total / Foreign" and 96-98 for special rows).
_REAL_STATE_FIPS_MAX = 56


def _year_pair(year: int) -> tuple[str, str]:
    """For `year` = 2022 (the destination year), return ('21','22')."""
    yy1 = f"{(year - 1) % 100:02d}"
    yy2 = f"{year % 100:02d}"
    return yy1, yy2


def _url_for(year: int, direction: str) -> str:
    yy1, yy2 = _year_pair(year)
    if direction == "inflow":
        return INFLOW_URL_TEMPLATE.format(yy1=yy1, yy2=yy2)
    if direction == "outflow":
        return OUTFLOW_URL_TEMPLATE.format(yy1=yy1, yy2=yy2)
    raise ValueError(f"direction must be 'inflow' or 'outflow', got {direction!r}")


def _cache_path(year: int, direction: str) -> Path:
    yy1, yy2 = _year_pair(year)
    return IRS_SOI_DIR / f"county{direction}{yy1}{yy2}.csv"


def fetch_year(year: int, direction: str = "inflow") -> pd.DataFrame:
    """Download (if not cached) and parse one (year, direction) SOI file.

    `year` is the *destination* tax year, e.g. 2022 for the 2021->2022 release.
    Output columns (normalised; both inflow and outflow):

        dest_state_fips, dest_county_fips,
        origin_state_fips, origin_county_fips,
        n_returns, n_exemptions, agi_thousands, year, direction

    All aggregate rows (state totals, foreign migration, etc.) are kept here;
    `compute_county_net_migration` is responsible for filtering them out.
    """
    path = _cache_path(year, direction)
    if not path.exists():
        url = _url_for(year, direction)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)

    raw = pd.read_csv(path, encoding="latin-1",
                      dtype={"y1_statefips": "Int64", "y2_statefips": "Int64",
                             "y1_countyfips": "Int64", "y2_countyfips": "Int64"})

    # In both inflow and outflow files: y1 = origin (prior year), y2 = destination
    # (current year). The "focus" county differs (it's whichever side is held
    # fixed and pairs with many counterparts), but the column meaning is stable.
    out = pd.DataFrame(
        {
            "dest_state_fips": raw["y2_statefips"].astype("Int64"),
            "dest_county_fips": raw["y2_countyfips"].astype("Int64"),
            "origin_state_fips": raw["y1_statefips"].astype("Int64"),
            "origin_county_fips": raw["y1_countyfips"].astype("Int64"),
            "n_returns": pd.to_numeric(raw["n1"], errors="coerce").astype("Int64"),
            "n_exemptions": pd.to_numeric(raw["n2"], errors="coerce").astype("Int64"),
            "agi_thousands": pd.to_numeric(raw["agi"], errors="coerce").astype("Int64"),
            "year": year,
            "direction": direction,
        }
    )
    return out


def _is_real_county(state_fips: pd.Series, county_fips: pd.Series) -> pd.Series:
    return (
        state_fips.notna()
        & county_fips.notna()
        & (state_fips >= 1)
        & (state_fips <= _REAL_STATE_FIPS_MAX)
        & (county_fips >= 1)
    )


def compute_county_net_migration(year: int) -> pd.DataFrame:
    """Per-county net migration for the given destination tax year.

    Pulls the inflow and outflow files, restricts both sides of each pair to
    real counties (drops state-totals, foreign, and "non-migrants" rows), and
    returns net flows from the perspective of the focus county.

    Returned columns:
        state_fips, county_fips, year, net_returns, net_agi_thousands
    """
    inflow = fetch_year(year, "inflow")
    outflow = fetch_year(year, "outflow")

    # Focus county for inflow = destination; for outflow = origin.
    # Counterparty side is filtered to real counties so we don't double-count
    # state-total or foreign-origin/destination aggregate rows.
    in_real = inflow[
        _is_real_county(inflow["dest_state_fips"], inflow["dest_county_fips"])
        & _is_real_county(inflow["origin_state_fips"], inflow["origin_county_fips"])
    ]
    out_real = outflow[
        _is_real_county(outflow["origin_state_fips"], outflow["origin_county_fips"])
        & _is_real_county(outflow["dest_state_fips"], outflow["dest_county_fips"])
    ]

    in_sum = (
        in_real.groupby(["dest_state_fips", "dest_county_fips"], as_index=False)
        .agg(returns_in=("n_returns", "sum"), agi_in=("agi_thousands", "sum"))
        .rename(columns={"dest_state_fips": "state_fips", "dest_county_fips": "county_fips"})
    )
    out_sum = (
        out_real.groupby(["origin_state_fips", "origin_county_fips"], as_index=False)
        .agg(returns_out=("n_returns", "sum"), agi_out=("agi_thousands", "sum"))
        .rename(columns={"origin_state_fips": "state_fips", "origin_county_fips": "county_fips"})
    )

    merged = in_sum.merge(out_sum, on=["state_fips", "county_fips"], how="outer").fillna(0)
    merged["year"] = year
    merged["net_returns"] = merged["returns_in"] - merged["returns_out"]
    merged["net_agi_thousands"] = merged["agi_in"] - merged["agi_out"]

    return merged[
        ["state_fips", "county_fips", "year", "net_returns", "net_agi_thousands"]
    ].astype(
        {
            "state_fips": "Int64",
            "county_fips": "Int64",
            "year": "Int64",
            "net_returns": "Int64",
            "net_agi_thousands": "Int64",
        }
    )


def build_and_save(years: list[int]) -> Path:
    """Build net-migration panel for multiple years and save as parquet.

    Output path: `data/processed/irs_migration_county_year.parquet`.
    Returns the path written.
    """
    frames = [compute_county_net_migration(y) for y in years]
    panel = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED / "irs_migration_county_year.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"[irs_soi] wrote {len(panel):,} rows for years={years} to {out_path}")
    return out_path
