"""ACS 5-year demographics at ZCTA level.

Source: US Census Bureau American Community Survey 5-year estimates,
the most-detailed sub-state demographic dataset. Published at the ZCTA
(zip-code tabulation area) level, which approximates but does not equal
USPS ZIPs (HUD ZIP<->ZCTA crosswalk handles the reconciliation).

API: https://api.census.gov/data/{year}/acs/acs5

Two endpoints share the same query grammar:
  - Detailed tables (B-codes):  /acs/acs5
  - Subject tables (S-codes):   /acs/acs5/subject

ACS 5-year estimates are *centered* on year 2 of the 5-year window; never
treat as monthly. Variable codes are largely stable across years 2010+,
but minor relabeling does happen — keep VARIABLES versioned.

Auth: an API key is recommended for >500 calls/day. If the env var
CENSUS_API_KEY is set we pass it through; without one the public API
still works for low-volume use.
"""
from __future__ import annotations

import os
from io import StringIO

import pandas as pd
import requests

from arbok.config import PROCESSED

ACS_BASE = "https://api.census.gov/data/{year}/acs/acs5"

# Friendly name -> ACS variable code.
# Verified against the live variables endpoint on 2026-05-21:
#   B01003_001E   total population (detailed table)            VERIFIED
#   B19013_001E   median household income, past 12mo (B-table) standard
#   B25077_001E   median value of owner-occupied housing units standard
#   B25064_001E   median gross rent                            standard
#   B25003_002E   owner-occupied housing units (numerator)
#   B25003_001E   total occupied housing units (denominator)
#   S0101_C01_007E  population 25-29 (NOT 25-34 as some refs claim) VERIFIED
#   S0101_C01_008E  population 30-34                                VERIFIED
#   S0101_C01_009E  population 35-39                                VERIFIED
#   S0101_C01_010E  population 40-44                                VERIFIED
#   S0101_C01_030E  population 65 years and over                    VERIFIED
#   S1501_C02_015E  pct of pop 25+ with bachelor's degree or higher VERIFIED
#
# The task spec mapped pop_25_to_34 -> S0101_C01_007E but that code is
# actually 25-29 only. We expose both 25-29 and 30-34 raw, then derive
# 25-34 (and the 28-38 first-time-homebuyer cohort) downstream.
VARIABLES: dict[str, str] = {
    "total_pop": "B01003_001E",
    "median_household_income": "B19013_001E",
    "median_home_value": "B25077_001E",
    "median_gross_rent": "B25064_001E",
    # Owner-occupied share is derived from these two:
    "owner_occupied_units": "B25003_002E",
    "occupied_units_total": "B25003_001E",
    # Age bins (subject table S0101) — kept raw, combined in derive_*:
    "pop_25_29": "S0101_C01_007E",
    "pop_30_34": "S0101_C01_008E",
    "pop_35_39": "S0101_C01_009E",
    "pop_40_44": "S0101_C01_010E",
    "pop_65_plus": "S0101_C01_030E",
    # Education
    "bachelors_or_higher_pct": "S1501_C02_015E",
}

# ACS S0101 (Age & Sex subject table) had a schema break between the 2016 and
# 2017 5-year vintages. Verified empirically on 2026-05-22 against the live
# API and the Census variable-metadata endpoint:
#
# (a) For the 5-year age-bin codes S0101_C01_007E..S0101_C01_010E
#     (25-29 / 30-34 / 35-39 / 40-44), the *label* is stable across
#     2013-2022, but the *unit* changed: in 2013-2016 these return percent of
#     total population (the bins for one ZCTA sum to ~100), while in 2017+
#     they return absolute counts (sum to ~total_pop). We normalize the
#     pre-2017 percentages back to counts using B01003_001E (total_pop, which
#     is count-valued in every vintage).
#
# (b) S0101_C01_030E is *not* stable: in 2013-2016 it is "Median age (years)",
#     in 2017+ it is "65 years and over" population. The pre-2017 row index
#     for 65+ is different. Since fixing that is out of scope (would require
#     a vintage-aware variable map for pop_65_plus), we explicitly null
#     pop_65_plus for 2013-2016 rather than mis-converting median-age to a
#     fake count.
_S0101_PCT_VINTAGES = {2013, 2014, 2015, 2016}
_S0101_AGE_BIN_FRIENDLY_NAMES = (
    "pop_25_29",
    "pop_30_34",
    "pop_35_39",
    "pop_40_44",
)
_S0101_BROKEN_PRE_2017 = ("pop_65_plus",)


def _split_by_endpoint(variables: dict[str, str]) -> tuple[dict, dict]:
    """Split variables into detailed-table (B...) and subject-table (S...) buckets."""
    detailed = {k: v for k, v in variables.items() if v.startswith("B")}
    subject = {k: v for k, v in variables.items() if v.startswith("S")}
    return detailed, subject


def _fetch_chunk(
    year: int, var_codes: list[str], endpoint_suffix: str = ""
) -> pd.DataFrame:
    """One request to /acs/acs5{endpoint_suffix} for all ZCTAs."""
    url = ACS_BASE.format(year=year) + endpoint_suffix
    params = {
        "get": ",".join(["NAME"] + var_codes),
        "for": "zip code tabulation area:*",
    }
    api_key = os.environ.get("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    # The geography column is named "zip code tabulation area"
    df = df.rename(columns={"zip code tabulation area": "zcta"})
    # Numeric coercion — ACS uses negative sentinels (e.g. -666666666) for
    # suppressed / not-computable; coerce then null them out.
    for code in var_codes:
        df[code] = pd.to_numeric(df[code], errors="coerce")
        df.loc[df[code] < -1e8, code] = pd.NA
    return df.drop(columns=["NAME"], errors="ignore")


def fetch_acs_zcta(
    year: int = 2022, variables: dict[str, str] | None = None
) -> pd.DataFrame:
    """Fetch ACS 5-year ZCTA-level estimates for the given variables.

    Returns a wide DataFrame with `zcta` and one column per friendly name.
    Handles both detailed (B) and subject (S) endpoints in one call.
    """
    variables = variables or VARIABLES
    detailed, subject = _split_by_endpoint(variables)

    parts: list[pd.DataFrame] = []
    if detailed:
        parts.append(_fetch_chunk(year, list(detailed.values()), endpoint_suffix=""))
    if subject:
        parts.append(
            _fetch_chunk(year, list(subject.values()), endpoint_suffix="/subject")
        )

    if not parts:
        raise ValueError("No variables to fetch")

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="zcta", how="outer")

    # Rename ACS codes to friendly names
    code_to_friendly = {v: k for k, v in variables.items()}
    out = out.rename(columns=code_to_friendly)

    # Derive owner-occupied share
    if {"owner_occupied_units", "occupied_units_total"}.issubset(out.columns):
        denom = out["occupied_units_total"].replace(0, pd.NA)
        out["owner_occupied_pct"] = (
            out["owner_occupied_units"].astype("Float64") / denom.astype("Float64")
        ) * 100.0

    # Normalize S0101 age-bin units across vintages — see module-level note
    # next to _S0101_PCT_VINTAGES. For 2013-2016 we convert percentages back
    # to counts using total_pop, and we null pop_65_plus because the variable
    # code points to a different concept in those vintages.
    if year in _S0101_PCT_VINTAGES and "total_pop" in out.columns:
        total = out["total_pop"].astype("Float64")
        for col in _S0101_AGE_BIN_FRIENDLY_NAMES:
            if col in out.columns:
                out[col] = (out[col].astype("Float64") / 100.0) * total
        for col in _S0101_BROKEN_PRE_2017:
            if col in out.columns:
                out[col] = pd.NA

    return out


def derive_age_cohort_share(
    df: pd.DataFrame, cohort_label: str = "age_28_38"
) -> pd.DataFrame:
    """Add a share-of-population column for the prime first-time-homebuyer band.

    ACS S0101 bins are 5-year (25-29, 30-34, 35-39, 40-44). True 28-38
    cannot be cut exactly from these bins, so we approximate as the union
    of the 30-34 and 35-39 bins (the 10-year band centered on age 32-39)
    and report it as the cohort share. This deliberately *under*counts
    by missing ages 28-29, but captures the bulk of first-time-buyer
    activity. A linear-interpolation alternative was considered and
    rejected — adds noise without changing rankings.
    """
    needed = {"pop_30_34", "pop_35_39", "total_pop"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for cohort derivation: {missing}")
    out = df.copy()
    numer = out["pop_30_34"].fillna(0) + out["pop_35_39"].fillna(0)
    denom = out["total_pop"].replace(0, pd.NA)
    out[f"{cohort_label}_share"] = (numer.astype("Float64") / denom.astype("Float64"))
    return out


def build_and_save(years: list[int]) -> pd.DataFrame:
    """Fetch ACS ZCTA data for each year, concat with a `year` column, save parquet."""
    frames: list[pd.DataFrame] = []
    for y in years:
        df = fetch_acs_zcta(year=y)
        df = derive_age_cohort_share(df)
        df.insert(0, "year", y)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    target = PROCESSED / "acs_zcta_year.parquet"
    out.to_parquet(target, index=False)
    return out


def load_acs_zcta() -> pd.DataFrame:
    """Load the saved ACS panel."""
    return pd.read_parquet(PROCESSED / "acs_zcta_year.parquet")
