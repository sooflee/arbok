"""Top-N US metros (CBSAs) by population, from Census Population Estimates Program.

Phase 0a. Produces `data/processed/top200_metros.csv` with columns:
    rank, cbsa, name, state, population, vintage

The Census PEP API path is `https://api.census.gov/data/{vintage}/pep/population`
and requires `CENSUS_API_KEY` in env (Census no longer accepts anonymous traffic).
If the API call fails, we fall back to a locally-staged CSV at
`data/raw/cbsa-est{vintage}.csv` (download manually from
https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-metro-and-micro-statistical-areas.html
and save the "Annual Estimates of the Resident Population" CSV).
"""
from __future__ import annotations

import os
import re
from io import StringIO

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW, TOP_N_METROS

API_URL = "https://api.census.gov/data/{vintage}/pep/population"
OUTPUT_CSV = PROCESSED / "top200_metros.csv"
METRO_SUFFIX = "Metro Area"  # CBSA NAME suffix for metropolitan statistical areas


def _split_name(name: str) -> tuple[str, str]:
    """Split 'New York-Newark-Jersey City, NY-NJ-PA Metro Area' into (core, states)."""
    cleaned = re.sub(r"\s+(Metro|Micro) Area$", "", name)
    if "," in cleaned:
        core, states = cleaned.rsplit(",", 1)
        return core.strip(), states.strip()
    return cleaned, ""


def fetch_cbsa_population(vintage: int = 2023) -> pd.DataFrame:
    """Pull CBSA-level population estimates from the Census PEP API.

    Falls back to `data/raw/cbsa-est{vintage}.csv` if the API call fails.
    Returned columns: cbsa, name, population, vintage.
    """
    params = {
        "get": f"NAME,POP_{vintage}",
        "for": "metropolitan statistical area/micropolitan statistical area:*",
    }
    key = os.getenv("CENSUS_API_KEY")
    if key:
        params["key"] = key

    url = API_URL.format(vintage=vintage)
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
    except (requests.RequestException, ValueError) as e:
        return _fallback_csv(vintage, reason=str(e))

    header, *data = rows
    df = pd.DataFrame(data, columns=header)
    df = df.rename(
        columns={
            f"POP_{vintage}": "population",
            "NAME": "name",
            "metropolitan statistical area/micropolitan statistical area": "cbsa",
        }
    )
    df["population"] = pd.to_numeric(df["population"])
    df["vintage"] = vintage
    return df[["cbsa", "name", "population", "vintage"]]


def _fallback_csv(vintage: int, reason: str) -> pd.DataFrame:
    """Read a locally-staged Census CSV when the API is unavailable.

    Expected file: `data/raw/cbsa-est{vintage}.csv` with at minimum columns
    CBSA, NAME, LSAD, POPESTIMATE{vintage} (the standard PEP release format).
    """
    path = RAW / f"cbsa-est{vintage}.csv"
    if not path.exists():
        raise RuntimeError(
            f"Census PEP API failed ({reason}) and no fallback CSV at {path}.\n"
            f"Either:\n"
            f"  1. Set CENSUS_API_KEY in env (get one at https://api.census.gov/data/key_signup.html), or\n"
            f"  2. Download cbsa-est{vintage}.csv from\n"
            f"     https://www.census.gov/data/tables/time-series/demo/popest/"
            f"2020s-total-metro-and-micro-statistical-areas.html\n"
            f"     and save it to {path}\n"
            f"TODO: once an API key is configured, the fallback is no longer needed."
        )
    raw = pd.read_csv(path, encoding="latin-1", dtype={"CBSA": str})
    pop_col = f"POPESTIMATE{vintage}"
    df = raw[["CBSA", "NAME", "LSAD", pop_col]].copy()
    df.columns = ["cbsa", "name", "lsad", "population"]
    # Re-append the " Metro Area" / " Micro Area" suffix so downstream filtering works.
    suffix_map = {"Metropolitan Statistical Area": " Metro Area",
                  "Micropolitan Statistical Area": " Micro Area"}
    df = df[df["lsad"].isin(suffix_map)].copy()
    df["name"] = df["name"] + df["lsad"].map(suffix_map)
    df["vintage"] = vintage
    return df[["cbsa", "name", "population", "vintage"]]


def top_n_metros(df: pd.DataFrame, n: int = TOP_N_METROS, metro_only: bool = True) -> pd.DataFrame:
    """Sort by population, optionally filter to metropolitan (drop micropolitan), take top N."""
    out = df.copy()
    if metro_only:
        out = out[out["name"].str.endswith(METRO_SUFFIX)]
    out = out.sort_values("population", ascending=False).head(n).reset_index(drop=True)
    out["rank"] = out.index + 1
    out[["core_name", "state"]] = out["name"].apply(lambda s: pd.Series(_split_name(s)))
    return out[["rank", "cbsa", "name", "core_name", "state", "population", "vintage"]]


def build_and_save(vintage: int = 2023, n: int = TOP_N_METROS) -> pd.DataFrame:
    """Fetch, filter to top N metros, save to CSV, print summary."""
    raw = fetch_cbsa_population(vintage=vintage)
    top = top_n_metros(raw, n=n, metro_only=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(top)} metros (vintage {vintage}) to {OUTPUT_CSV}")
    print(f"Total population covered: {top['population'].sum():,}")
    print("Top 5:")
    print(top.head(5)[["rank", "core_name", "state", "population"]].to_string(index=False))
    return top


def load_top200() -> pd.DataFrame:
    """Read the saved top-200 CSV."""
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(f"{OUTPUT_CSV} not found. Run build_and_save() first.")
    return pd.read_csv(OUTPUT_CSV, dtype={"cbsa": str})


if __name__ == "__main__":
    build_and_save()
