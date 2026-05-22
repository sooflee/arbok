"""Effective property tax rate per ZCTA per year.

Computed directly from the Census American Community Survey 5-year estimates
to sidestep scraping the Tax Foundation rankings (which themselves derive
from the same ACS variables).

Definition::

    effective_property_tax_rate
        = median real-estate taxes paid (ACS B25103_001E, USD/year)
        / median home value             (ACS B25077_001E, USD)

This is the standard "median taxes / median value" effective-rate proxy used
by the Tax Foundation, NAR, and most academic work. It's a ratio of two
*medians* (not a true ratio per household), so it's a stable structural
indicator rather than a household-level burden statistic.

Caveats:
- Numerator (B25103) is real-estate taxes paid on *owner-occupied* units
  with a mortgage; the denominator (B25077) is the median value across all
  owner-occupied units. The mismatch is small in practice and the ratio is
  what Tax Foundation publishes.
- ACS suppresses these variables in low-population ZCTAs (negative
  sentinels like -666666666); we coerce those to NA.
- 5-year ACS is centered on year 2 of the window — treat as slow-moving.
- ZCTA universe: B25103 has been published at the ZCTA level since the
  2011 5-year file, so any year from 2011 onward is feasible. We tested
  2020-2022; payload is ~33K ZCTAs * 2 vars per year, well under 5 MB.

Ref:
- https://api.census.gov/data/2022/acs/acs5/variables/B25103_001E.html
- https://api.census.gov/data/2022/acs/acs5/variables/B25077_001E.html
- https://taxfoundation.org/data/all/state/property-taxes-by-state/
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from arbok.config import PROCESSED

ACS_BASE = "https://api.census.gov/data/{year}/acs/acs5"
OUT_PATH = PROCESSED / "property_tax_zcta_year.parquet"

REQUEST_TIMEOUT = 120

# ACS detailed-table variable codes (verified 2026-05-21 against the live
# variables endpoint). Both are in the /acs/acs5 detailed endpoint — no
# subject-table split needed.
VAR_MEDIAN_TAX = "B25103_001E"   # Median real-estate taxes paid (USD)
VAR_MEDIAN_VALUE = "B25077_001E"  # Median value owner-occupied housing units (USD)


def _fetch_acs_zcta_chunk(year: int, var_codes: list[str]) -> pd.DataFrame:
    """One detailed-table request to /acs/acs5 for all ZCTAs."""
    url = ACS_BASE.format(year=year)
    params = {
        "get": ",".join(["NAME"] + var_codes),
        "for": "zip code tabulation area:*",
    }
    api_key = os.environ.get("CENSUS_API_KEY")
    if api_key:
        params["key"] = api_key
    r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={"zip code tabulation area": "zcta"})
    for code in var_codes:
        df[code] = pd.to_numeric(df[code], errors="coerce")
        # ACS suppression sentinels are large negatives (-666666666 etc).
        df.loc[df[code] < -1e8, code] = pd.NA
    return df.drop(columns=["NAME"], errors="ignore")


def fetch_property_tax_zcta(years: list[int]) -> pd.DataFrame:
    """Fetch B25103 + B25077 per ZCTA per year, compute effective rate.

    Returns a long frame with columns:
        zcta, year, median_real_estate_tax_paid, median_home_value,
        effective_property_tax_rate

    The rate is a unitless fraction (e.g. 0.0218 = 2.18%).
    """
    frames: list[pd.DataFrame] = []
    for y in years:
        print(f"[property_tax] fetching ACS B25103 + B25077 for {y}")
        df = _fetch_acs_zcta_chunk(y, [VAR_MEDIAN_TAX, VAR_MEDIAN_VALUE])
        df = df.rename(
            columns={
                VAR_MEDIAN_TAX: "median_real_estate_tax_paid",
                VAR_MEDIAN_VALUE: "median_home_value",
            }
        )
        df["year"] = y
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    # Compute rate as fraction. Guard against zero-value denominators.
    denom = out["median_home_value"].astype("Float64").replace(0, pd.NA)
    out["effective_property_tax_rate"] = (
        out["median_real_estate_tax_paid"].astype("Float64") / denom
    )
    return out[
        [
            "zcta",
            "year",
            "median_real_estate_tax_paid",
            "median_home_value",
            "effective_property_tax_rate",
        ]
    ].sort_values(["zcta", "year"]).reset_index(drop=True)


def build_and_save(years: list[int]) -> pd.DataFrame:
    """Fetch the requested years, write parquet, return the frame."""
    out = fetch_property_tax_zcta(years)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"[property_tax] wrote {len(out):,} zcta-year rows to {OUT_PATH}")
    return out


def load_property_tax() -> pd.DataFrame:
    """Convenience loader for the saved panel."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} not found - run build_and_save([years]) first."
        )
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    panel = build_and_save([2022])
    print(panel.head())
