"""Zillow research-data loader (ZHVI + ZORI at zip level).

Source: https://www.zillow.com/research/data/

We use two series at zip-code geography:

* ZHVI — Single-Family Residence Time Series, Smoothed, Seasonally Adjusted
  ``Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv``
* ZORI — All Homes Plus Multifamily Time Series, Smoothed, Seasonally Adjusted
  ``Zip_zori_uc_sfrcondomfr_sm_sa_month.csv``

The CSVs are wide-format: one row per zip, one column per month
(``YYYY-MM-DD`` headers). We reshape to tidy long form with columns
``zip``, ``year_month`` (pandas Period[M]), ``value``.

Files are cached under ``data/raw/zillow/`` and re-downloaded only when
absent. Each file is ~10–125 MB.

Zip codes are 5-digit *strings* with leading zeros preserved (e.g.
``"01001"`` for Agawam, MA). Numeric coercion is forbidden.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from arbok.config import RAW

logger = logging.getLogger(__name__)

ZILLOW_RAW = RAW / "zillow"
ZILLOW_RAW.mkdir(parents=True, exist_ok=True)

ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv"
)
ZORI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Zip_zori_uc_sfrcondomfr_sm_sa_month.csv"
)

ZHVI_LOCAL = ZILLOW_RAW / Path(ZHVI_URL).name
ZORI_LOCAL = ZILLOW_RAW / Path(ZORI_URL).name


def _download(url: str, dest: Path, *, chunk: int = 1 << 20) -> Path:
    """Stream a URL to disk; skip if the file already exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Cached %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if block:
                    fh.write(block)
        tmp.replace(dest)
    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# Non-date metadata columns present in every Zillow zip-level CSV.
_META_COLS = {
    "RegionID",
    "SizeRank",
    "RegionName",
    "RegionType",
    "StateName",
    "State",
    "City",
    "Metro",
    "CountyName",
}


def _reshape(path: Path) -> pd.DataFrame:
    """Read a wide Zillow CSV and reshape to long ``zip``/``year_month``/``value``.

    ``RegionName`` is the 5-digit zip; we read it as string and zero-pad to
    guard against any source-side stripping.
    """
    # Read RegionName as string so leading zeros stick. dtype= on a wide CSV
    # avoids a full dtype dict at the cost of an inferred read for date cols.
    df = pd.read_csv(path, dtype={"RegionName": "string"})
    date_cols = [c for c in df.columns if c not in _META_COLS]
    if not date_cols:
        raise RuntimeError(f"No date columns found in {path.name}")

    long = df.melt(
        id_vars=["RegionName"],
        value_vars=date_cols,
        var_name="date",
        value_name="value",
    ).rename(columns={"RegionName": "zip"})

    long["zip"] = long["zip"].str.zfill(5)
    long["year_month"] = pd.to_datetime(long["date"], errors="coerce").dt.to_period("M")
    long = long.dropna(subset=["year_month", "value"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return long[["zip", "year_month", "value"]].reset_index(drop=True)


def fetch_zhvi() -> pd.DataFrame:
    """Download (if needed) and return tidy ZHVI: zip / year_month / value (USD)."""
    path = _download(ZHVI_URL, ZHVI_LOCAL)
    return _reshape(path)


def fetch_zori() -> pd.DataFrame:
    """Download (if needed) and return tidy ZORI: zip / year_month / value (USD/mo)."""
    path = _download(ZORI_URL, ZORI_LOCAL)
    return _reshape(path)
