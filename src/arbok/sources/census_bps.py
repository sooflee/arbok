"""Census Building Permits Survey (BPS) loader at MSA / county granularity.

The BPS is the canonical monthly count of residential building permits issued
by US permit-issuing places. Census re-publishes it under
``https://www2.census.gov/econ/bps/`` in this directory layout:

  - ``Metro (ending 2023)/``        — pre-2024 metro panels (2017 OMB delineation)
  - ``CBSA (beginning Jan 2024)/``  — 2024+ metro panels (2023 OMB delineation)
  - ``County/``                     — county-level panels
  - ``State/``, ``Place/``          — not used here

File naming convention (confirmed 2026-05-21 against the directory listing):

  - Monthly metro (pre-2024):  ``ma{YYMM}c.txt`` (current) / ``ma{YYMM}y.txt`` (YTD)
  - Monthly metro (2024+):     ``cbsa{YYMM}c.txt`` / ``cbsa{YYMM}y.txt``
  - Monthly county:            ``co{YYMM}c.txt`` / ``co{YYMM}y.txt``
  - Annual:                    ``ma{YYYY}a.txt`` / ``cbsa{YYYY}a.txt`` / ``co{YYYY}a.txt``

Despite older references calling these "fixed-width", post-2000 files are
*comma-delimited* with a two-row banner header (category + sub-column row) and
one blank line before the data. Each unit class is exposed as three columns —
Bldgs, Units, Value — for both the raw and the "reported" (imputation-adjusted)
panels, yielding 28 columns at metro level and 29 at county level (extra
state-FIPS split). We keep *Units* (dwelling units) and the raw-panel valuation.

**Valuation units differ by geography**: metro/CBSA files report value in
*thousands of dollars*, county files in *whole dollars* (verified on 2024-01
samples). We normalise both to whole USD in ``valuation_total_usd``.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

BPS_DIR = RAW / "census_bps"
BPS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www2.census.gov/econ/bps"
METRO_OLD_DIR = "Metro%20(ending%202023)"
METRO_NEW_DIR = "CBSA%20(beginning%20Jan%202024)"
COUNTY_DIR = "County"

# Cutover from `ma`/2017-delineation to `cbsa`/2023-delineation files is
# calendar-driven (Jan 2024); we route by year rather than trying both URLs.
_CBSA_CUTOVER_YEAR = 2024

REQUEST_TIMEOUT = 60

# Column indices (0-based) for ma/cbsa monthly files (28 cols).
# Layout: 0=Date 1=CSA 2=CBSA 3=coverage 4=Name then 4x(Bldgs,Units,Value) raw
# at cols 5-16, then the same four classes again as "reported" (imputed) — skipped.
_METRO_COLS = {
    "year_month_raw": 0,
    "cbsa": 2,
    "cbsa_name": 4,
    "permits_1unit": 6,
    "permits_2units": 9,
    "permits_3to4units": 12,
    "permits_5plus_units": 15,
    "valuation_1unit": 7,
    "valuation_2units": 10,
    "valuation_3to4units": 13,
    "valuation_5plus_units": 16,
}

# County monthly files (29 cols): 0=Date 1=StateFIPS 2=CountyFIPS 3=Region
# 4=Division 5=Name, then 4x(Bldgs,Units,Value) raw at cols 6-17 (offset +1 vs metro).
_COUNTY_COLS = {
    "year_month_raw": 0,
    "state_fips": 1,
    "county_fips": 2,
    "county_name": 5,
    "permits_1unit": 7,
    "permits_2units": 10,
    "permits_3to4units": 13,
    "permits_5plus_units": 16,
    "valuation_1unit": 8,
    "valuation_2units": 11,
    "valuation_3to4units": 14,
    "valuation_5plus_units": 17,
}


def _file_name(geo: str, year: int, month: int | None) -> str:
    """Build the BPS file basename for the given geography + period."""
    if month is None:  # annual
        if geo == "msa":
            prefix = "cbsa" if year >= _CBSA_CUTOVER_YEAR else "ma"
            return f"{prefix}{year}a.txt"
        return f"co{year}a.txt"
    yymm = f"{year % 100:02d}{month:02d}"
    if geo == "msa":
        prefix = "cbsa" if year >= _CBSA_CUTOVER_YEAR else "ma"
        return f"{prefix}{yymm}c.txt"
    return f"co{yymm}c.txt"


def _file_url(geo: str, year: int, month: int | None) -> str:
    name = _file_name(geo, year, month)
    if geo == "msa":
        subdir = METRO_NEW_DIR if year >= _CBSA_CUTOVER_YEAR else METRO_OLD_DIR
    else:
        subdir = COUNTY_DIR
    return f"{BASE_URL}/{subdir}/{name}"


def _cache_path(geo: str, year: int, month: int | None) -> Path:
    return BPS_DIR / _file_name(geo, year, month)


def _download(geo: str, year: int, month: int | None) -> Path:
    path = _cache_path(geo, year, month)
    if path.exists() and path.stat().st_size > 0:
        return path
    url = _file_url(geo, year, month)
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _parse_bps_file(path: Path, geo: str) -> pd.DataFrame:
    """Parse one BPS monthly/annual file. Skips the 2-row banner + blank line."""
    text = path.read_text(encoding="latin-1")
    df = pd.read_csv(
        StringIO(text),
        header=None,
        skiprows=3,
        dtype=str,
        engine="python",
        skip_blank_lines=True,
        on_bad_lines="skip",
    )
    schema = _METRO_COLS if geo == "msa" else _COUNTY_COLS
    needed = max(schema.values()) + 1
    if df.shape[1] < needed:
        raise ValueError(
            f"{path.name}: parsed {df.shape[1]} columns, expected >= {needed}. "
            "Header layout may have changed."
        )

    def _num(idx: int) -> pd.Series:
        return pd.to_numeric(df[idx], errors="coerce").fillna(0).astype("Int64")

    if geo == "msa":
        out = pd.DataFrame(
            {
                "cbsa": df[schema["cbsa"]].astype(str).str.strip(),
                "cbsa_name": df[schema["cbsa_name"]].astype(str).str.strip(),
            }
        )
    else:
        state = df[schema["state_fips"]].astype(str).str.strip().str.zfill(2)
        county = df[schema["county_fips"]].astype(str).str.strip().str.zfill(3)
        out = pd.DataFrame(
            {
                "state_fips": state,
                "county_fips": county,
                "county_name": df[schema["county_name"]].astype(str).str.strip(),
            }
        )

    out["year_month"] = pd.to_datetime(
        df[schema["year_month_raw"]].astype(str).str.strip(),
        format="%Y%m",
        errors="coerce",
    )
    for unit in ("1unit", "2units", "3to4units", "5plus_units"):
        out[f"permits_{unit}"] = _num(schema[f"permits_{unit}"])
    out["permits_total"] = sum(
        out[f"permits_{u}"] for u in ("1unit", "2units", "3to4units", "5plus_units")
    )
    valuation_raw = sum(
        _num(schema[f"valuation_{u}"])
        for u in ("1unit", "2units", "3to4units", "5plus_units")
    )
    # Metro/CBSA files report valuation in $1000s; county files in $1s. Normalise.
    scale = 1000 if geo == "msa" else 1
    out["valuation_total_usd"] = (valuation_raw * scale).astype("Int64")

    out = out.dropna(subset=["year_month"]).reset_index(drop=True)
    return out


def _fetch_geo(year: int, geo: str, monthly: bool) -> pd.DataFrame:
    if not monthly:
        return _parse_bps_file(_download(geo, year, None), geo)
    frames = [_parse_bps_file(_download(geo, year, m), geo) for m in range(1, 13)]
    return pd.concat(frames, ignore_index=True)


def fetch_bps_msa(year: int, monthly: bool = True) -> pd.DataFrame:
    """Fetch BPS MSA-level permits for one year.

    Routes to ``ma*.txt`` (2017 OMB delineation) for years < 2024 and
    ``cbsa*.txt`` (2023 OMB delineation) thereafter. CBSA codes are stable
    across the cutover; CSA-level groupings and a handful of names changed.
    Output cols: cbsa, cbsa_name, year_month, permits_1unit, permits_2units,
    permits_3to4units, permits_5plus_units, permits_total, valuation_total_usd.
    """
    cols = [
        "cbsa", "cbsa_name", "year_month",
        "permits_1unit", "permits_2units", "permits_3to4units",
        "permits_5plus_units", "permits_total", "valuation_total_usd",
    ]
    return _fetch_geo(year, "msa", monthly)[cols]


def fetch_bps_county(year: int, monthly: bool = True) -> pd.DataFrame:
    """Fetch BPS county-level permits for one year (same schema, county keys)."""
    cols = [
        "state_fips", "county_fips", "county_name", "year_month",
        "permits_1unit", "permits_2units", "permits_3to4units",
        "permits_5plus_units", "permits_total", "valuation_total_usd",
    ]
    return _fetch_geo(year, "county", monthly)[cols]


def _processed_path(geo: str) -> Path:
    return PROCESSED / f"bps_{geo}_month.parquet"


def build_and_save(years: list[int], geo: str = "msa") -> Path:
    """Fetch monthly BPS for many years and persist to a single parquet."""
    if geo not in ("msa", "county"):
        raise ValueError(f"geo must be 'msa' or 'county', got {geo!r}")
    fetch = fetch_bps_msa if geo == "msa" else fetch_bps_county
    frames = [fetch(y, monthly=True) for y in years]
    panel = pd.concat(frames, ignore_index=True)
    out_path = _processed_path(geo)
    panel.to_parquet(out_path, index=False)
    print(f"[census_bps] wrote {len(panel):,} rows for years={years} to {out_path}")
    return out_path


def load_bps(geo: str = "msa") -> pd.DataFrame:
    """Read the persisted monthly BPS panel for the given geography."""
    path = _processed_path(geo)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run build_and_save(years, geo={geo!r}) first."
        )
    return pd.read_parquet(path)
