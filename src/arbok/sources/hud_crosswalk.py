"""HUD-USPS ZIP <-> CBSA crosswalk loader.

The HUD-USPS ZIP Code Crosswalk maps each 5-digit ZIP to its dominant
Census geography for a given quarter. We use the **ZIP-CBSA** flavor
(type=3 in the HUD API) and pick the dominant CBSA per zip by
residential-address ratio.

Access
------
HUD requires a free account + API token to download the files. Sign up
at https://www.huduser.gov/portal/dataset/uspszip-api.html and copy the
bearer token into the ``HUD_USPS_TOKEN`` environment variable.

Two ways to feed the loader:

1. **Manual Excel download (recommended; mirrors the canonical files).**
   Browse https://www.huduser.gov/portal/datasets/usps_crosswalk.html,
   pick a *quarter*, and download the **ZIP_CBSA** workbook. Save it to::

       data/raw/hud/ZIP_CBSA_{YYYY}{Q}.xlsx     # e.g. ZIP_CBSA_2024Q4.xlsx

   Then call :func:`load_zip_cbsa_crosswalk(year, quarter)`. The HUD
   workbook ships as ``.xlsx``; older releases are ``.xlsx`` too.

2. **API fallback** for CI / smoke tests:
   :func:`load_zip_cbsa_crosswalk_static` calls the JSON API at
   ``https://www.huduser.gov/hudapi/public/usps`` with
   ``Authorization: Bearer $HUD_USPS_TOKEN``. Only the most recent
   release is reliably available this way.

The returned DataFrame has columns: ``zip``, ``cbsa``,
``res_ratio``, ``bus_ratio``, ``oth_ratio``, ``tot_ratio``. We expose a
helper :func:`dominant_cbsa_per_zip` that collapses ties to the single
CBSA with the largest residential ratio per zip.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests

from arbok.config import RAW

logger = logging.getLogger(__name__)

HUD_RAW = RAW / "hud"
HUD_RAW.mkdir(parents=True, exist_ok=True)

HUD_API_URL = "https://www.huduser.gov/hudapi/public/usps"
HUD_PORTAL_URL = "https://www.huduser.gov/portal/datasets/usps_crosswalk.html"
# HUD type codes: 1=ZIP-Tract, 2=ZIP-County, 3=ZIP-CBSA, 4=ZIP-CBSA Div, ...
ZIP_CBSA_TYPE = 3

# Column rename so the manual Excel and API responses converge on one schema.
_RENAME = {
    "ZIP": "zip",
    "zip": "zip",
    "CBSA": "cbsa",
    "cbsa": "cbsa",
    "RES_RATIO": "res_ratio",
    "res_ratio": "res_ratio",
    "BUS_RATIO": "bus_ratio",
    "bus_ratio": "bus_ratio",
    "OTH_RATIO": "oth_ratio",
    "oth_ratio": "oth_ratio",
    "TOT_RATIO": "tot_ratio",
    "tot_ratio": "tot_ratio",
}


def _expected_xlsx(year: int, quarter: int) -> Path:
    return HUD_RAW / f"ZIP_CBSA_{year}Q{quarter}.xlsx"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the canonical schema with zero-padded zip / cbsa strings."""
    df = df.rename(columns=_RENAME)
    keep = [c for c in ("zip", "cbsa", "res_ratio", "bus_ratio", "oth_ratio", "tot_ratio")
            if c in df.columns]
    out = df[keep].copy()
    out["zip"] = out["zip"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(5)
    out["cbsa"] = out["cbsa"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(5)
    for col in ("res_ratio", "bus_ratio", "oth_ratio", "tot_ratio"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["zip", "cbsa"]).reset_index(drop=True)


def load_zip_cbsa_crosswalk(year: int, quarter: int) -> pd.DataFrame:
    """Load a manually-downloaded HUD ZIP-CBSA workbook for the given quarter.

    Looks for ``data/raw/hud/ZIP_CBSA_{year}Q{quarter}.xlsx``. If absent, raises
    a :class:`FileNotFoundError` that explains how to fetch it.
    """
    path = _expected_xlsx(year, quarter)
    if not path.exists():
        raise FileNotFoundError(
            f"HUD ZIP-CBSA crosswalk not found at {path}.\n"
            f"Manual steps:\n"
            f"  1. Create a free HUD account at {HUD_PORTAL_URL}\n"
            f"  2. From the same page download the ZIP-CBSA workbook for "
            f"{year} Q{quarter}\n"
            f"  3. Save it as {path.name} (lowercase extension)\n"
            f"Alternatively set HUD_USPS_TOKEN in env and call "
            f"load_zip_cbsa_crosswalk_static()."
        )
    logger.info("Reading HUD ZIP-CBSA crosswalk %s", path.name)
    df = pd.read_excel(path, dtype={"ZIP": str, "CBSA": str})
    return _normalize(df)


def load_zip_cbsa_crosswalk_static(quarter: str | None = None) -> pd.DataFrame:
    """Fallback: pull the latest ZIP-CBSA crosswalk via the HUD JSON API.

    Requires ``HUD_USPS_TOKEN`` in env. ``quarter`` follows HUD's ``YYYYQq``
    convention (e.g. ``"2024Q4"``); when omitted the API returns the most
    recent release.
    """
    token = os.getenv("HUD_USPS_TOKEN")
    if not token:
        raise RuntimeError(
            "HUD_USPS_TOKEN env var not set. Register at "
            f"{HUD_PORTAL_URL} and copy your API token into .env."
        )
    params: dict[str, str | int] = {"type": ZIP_CBSA_TYPE, "query": "All"}
    if quarter:
        params["year"] = quarter
    headers = {"Authorization": f"Bearer {token}"}
    logger.info("GET %s params=%s", HUD_API_URL, params)
    resp = requests.get(HUD_API_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    # HUD wraps the rows under {"data": {"results": [...]}} (verified empirically
    # against the live API; older docs show a flat list, so we handle both).
    if isinstance(payload, dict):
        results = payload.get("data", {}).get("results") or payload.get("results") or []
    else:
        results = payload
    if not results:
        raise RuntimeError(f"HUD API returned no rows: {payload!r:.200}")
    df = pd.DataFrame(results)
    # API uses `geoid` for the non-zip target geography (cbsa here).
    if "geoid" in df.columns and "cbsa" not in df.columns:
        df = df.rename(columns={"geoid": "cbsa"})
    return _normalize(df)


def dominant_cbsa_per_zip(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Collapse one-to-many ZIP->CBSA rows by picking the largest residential ratio.

    Returns a DataFrame with one row per zip and columns ``zip``, ``cbsa``,
    ``res_ratio``.
    """
    if crosswalk.empty:
        return crosswalk.assign(res_ratio=pd.Series(dtype=float))[["zip", "cbsa", "res_ratio"]]
    ranked = crosswalk.sort_values(
        ["zip", "res_ratio", "tot_ratio"], ascending=[True, False, False]
    )
    top = ranked.drop_duplicates(subset=["zip"], keep="first")
    return top[["zip", "cbsa", "res_ratio"]].reset_index(drop=True)
