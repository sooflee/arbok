"""Realtor.com Research monthly inventory loader (zip-code level).

Source: https://www.realtor.com/research/data/  ("Inventory Data" section,
"Real Estate Data — Monthly Inventory — by Zip Code — History").

The canonical historical CSV is published to a public S3 bucket and refreshed
monthly. One row per (postal_code, month_date_yyyymm). The file is ~760 MB
and covers 2016-07 onward.

Cached to ``data/raw/realtor/`` and re-downloaded only when absent.

Column mapping (source -> our canonical name):

* ``median_listing_price``                  -> ``median_listing_price``
* ``median_days_on_market``                 -> ``median_dom``
* ``active_listing_count``                  -> ``active_listing_count``
* ``new_listing_count``                     -> ``new_listing_count``
* ``price_reduced_count``                   -> ``price_reduced_count``
* ``pending_listing_count``                 -> ``pending_listing_count``
* ``median_listing_price_per_square_foot``  -> ``median_listing_price_per_sqft``

NOT exposed by Realtor in the zip-level Core Metrics file (left out / NaN):

* ``months_supply`` — never published at zip tier. Realtor only publishes
  months-of-supply at the metro tier (``..._Metro_History.csv``). For zip
  level we leave a NaN column so downstream features can fall back to
  ``active_listing_count / pending_listing_count`` or to broadcast metro
  months-supply via the HUD ZIP-CBSA crosswalk.

ZIPs (``postal_code``) are 5-digit *strings* with leading zeros preserved.
Numeric coercion is forbidden.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

logger = logging.getLogger(__name__)

REALTOR_RAW = RAW / "realtor"
REALTOR_RAW.mkdir(parents=True, exist_ok=True)

RDC_ZIP_INVENTORY_URL = (
    "https://econdata.s3-us-west-2.amazonaws.com/"
    "Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv"
)
RDC_ZIP_INVENTORY_LOCAL = REALTOR_RAW / Path(RDC_ZIP_INVENTORY_URL).name

PROCESSED_PATH = PROCESSED / "realtor_inventory_zip_month.parquet"

# Realtor source column -> our canonical name.
_RENAME: dict[str, str] = {
    "median_listing_price": "median_listing_price",
    "median_days_on_market": "median_dom",
    "active_listing_count": "active_listing_count",
    "new_listing_count": "new_listing_count",
    "price_reduced_count": "price_reduced_count",
    "pending_listing_count": "pending_listing_count",
    "median_listing_price_per_square_foot": "median_listing_price_per_sqft",
}

# Features the user requested but Realtor does not expose at zip tier.
_MISSING: tuple[str, ...] = ("months_supply",)

_FEATURE_COLS: tuple[str, ...] = (
    "months_supply",
    "median_dom",
    "median_listing_price",
    "active_listing_count",
    "new_listing_count",
    "price_reduced_count",
    "pending_listing_count",
    "median_listing_price_per_sqft",
)


def _download(url: str, dest: Path, *, chunk: int = 1 << 20) -> Path:
    """Stream a URL to disk; skip if already cached and non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Cached %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if block:
                    fh.write(block)
    tmp.replace(dest)
    logger.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def _reshape(path: Path) -> pd.DataFrame:
    """Read the Realtor zip CSV and return tidy long DataFrame.

    Realtor pads its monthly file with a final aggregate footer row whose
    ``postal_code`` is blank or non-numeric; we filter to 5-digit ZIPs.
    """
    usecols = ["month_date_yyyymm", "postal_code", *_RENAME.keys()]
    # postal_code as string so leading zeros stick; yyyymm as string so we can
    # parse cleanly to Period[M] without int->float coercion losing precision.
    raw = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"postal_code": "string", "month_date_yyyymm": "string"},
        low_memory=False,
    )

    df = raw.rename(columns=_RENAME).rename(columns={"postal_code": "zip"})

    # Drop footer / aggregate rows: keep only purely-numeric 5-digit-or-less ZIPs.
    df = df[df["zip"].notna() & df["zip"].str.fullmatch(r"\d{1,5}")]
    df["zip"] = df["zip"].str.zfill(5)

    # yyyymm (e.g. "202604") -> month-start Timestamp via Period[M].
    period = pd.PeriodIndex(
        pd.to_datetime(df["month_date_yyyymm"], format="%Y%m", errors="coerce"),
        freq="M",
    )
    df["year_month"] = period.to_timestamp()
    df = df.dropna(subset=["year_month"])

    # Fill in the columns Realtor does not expose at zip tier.
    for col in _MISSING:
        df[col] = pd.NA

    # Coerce numeric features (CSV may contain blanks for thin zips).
    for col in _FEATURE_COLS:
        if col in _MISSING:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = df[["zip", "year_month", *_FEATURE_COLS]].reset_index(drop=True)
    return out.sort_values(["zip", "year_month"], ignore_index=True)


def fetch_inventory() -> pd.DataFrame:
    """Download (if needed) and return tidy zip-month Realtor inventory.

    Returns one row per (``zip``, ``year_month``) with the canonical
    feature columns listed in the module docstring. ``months_supply`` is
    always NaN at zip tier (see module docstring).
    """
    path = _download(RDC_ZIP_INVENTORY_URL, RDC_ZIP_INVENTORY_LOCAL)
    return _reshape(path)


def build_and_save(dest: Path = PROCESSED_PATH) -> Path:
    """Build the tidy panel and write it to ``data/processed/`` as parquet."""
    df = fetch_inventory()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info("Wrote %s (%d rows, %.1f MB)", dest, len(df), dest.stat().st_size / 1e6)
    return dest


def load_realtor_inventory(*, rebuild: bool = False) -> pd.DataFrame:
    """Convenience loader: read processed parquet, building it if absent.

    Set ``rebuild=True`` to force a fresh build from the cached raw CSV.
    """
    if rebuild or not PROCESSED_PATH.exists():
        build_and_save(PROCESSED_PATH)
    df = pd.read_parquet(PROCESSED_PATH)
    # Parquet round-trip preserves dtypes, but be defensive about the ZIP.
    df["zip"] = df["zip"].astype("string").str.zfill(5)
    return df
