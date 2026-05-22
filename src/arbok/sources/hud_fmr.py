"""HUD Fair Market Rent (FMR) loader.

FMR is the rent HUD uses to set Section-8 voucher payment standards. Two
flavours, annual, by bedroom count (efficiency / 1BR / 2BR / 3BR / 4BR,
USD/month):

* **Small Area FMR (SAFMR)** — per 5-digit ZIP, only inside the ~24
  metros HUD mandates SAFMR for (started FY2018; ZIP rows in API from
  ``year=2018`` onward).
* **FMR area** — per HUD Metro FMR Area and per non-metro county;
  universal, covers every US county plus territories.

Source: public JSON API at ``https://www.huduser.gov/hudapi/public/fmr/``
using the shared ``HUD_USPS_TOKEN`` bearer (see ``hud_crosswalk.py``).
Endpoints: ``listStates`` (56 codes), ``statedata/{state}?year=YYYY``
(metro + county FMR rows, with ``smallarea_status`` flag and 10-digit
``fips_code``), and ``data/{entity_id}?year=YYYY`` (SAFMR metros return
``basicdata`` as a list of ZIP rows; first row ``"MSA level"`` is
skipped). API year coverage 2017+; SAFMR populated 2018+.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

logger = logging.getLogger(__name__)

HUD_RAW = RAW / "hud"
HUD_RAW.mkdir(parents=True, exist_ok=True)
HUD_API_BASE = "https://www.huduser.gov/hudapi/public/fmr"
HUD_PORTAL_URL = "https://www.huduser.gov/portal/datasets/fmr.html"
# HUD API field -> our short bedroom column name.
_BR_RENAME = {"Efficiency": "fmr_efficiency", "One-Bedroom": "fmr_1br",
              "Two-Bedroom": "fmr_2br", "Three-Bedroom": "fmr_3br",
              "Four-Bedroom": "fmr_4br"}
_BR_COLS = list(_BR_RENAME.values())


def _get(path: str, params: dict | None = None, max_retries: int = 7) -> dict | list:
    """GET against the HUD API with exponential backoff on 429 / 5xx."""
    tok = os.getenv("HUD_USPS_TOKEN")
    if not tok:
        raise RuntimeError(
            f"HUD_USPS_TOKEN env var not set. Register at {HUD_PORTAL_URL} "
            "and copy your API token into .env."
        )
    headers = {"Authorization": f"Bearer {tok}"}
    url = f"{HUD_API_BASE}/{path}"
    delay = 2.0
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params or {}, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries - 1:
                resp.raise_for_status()
            logger.info("HUD API %s on %s, sleeping %.1fs", resp.status_code, path, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("unreachable")


def _list_states() -> list[str]:
    return [s["state_code"] for s in _get("listStates")]  # type: ignore[index]


def _to_numeric_br(row: dict) -> dict:
    return {col: pd.to_numeric(row.get(api), errors="coerce")
            for api, col in _BR_RENAME.items()}


def fetch_fmr_by_zip(year: int) -> pd.DataFrame:
    """Pull Small Area FMR rents for every SAFMR metro for ``year``.

    Walks ``listStates`` -> ``statedata`` for ``smallarea_status='1'``
    codes, then ``data/{code}`` for ZIP-level ``basicdata``. One row per
    (zip, year).
    """
    if year < 2018:
        raise ValueError(f"SAFMR ZIP data starts at 2018; got year={year}")
    rows: list[dict] = []
    for state in _list_states():
        try:
            payload = _get(f"statedata/{state}", {"year": year})
        except requests.HTTPError as exc:
            logger.warning("statedata %s %s failed: %s", state, year, exc)
            continue
        metros = payload.get("data", {}).get("metroareas", [])  # type: ignore[union-attr]
        for code in [m["code"] for m in metros if m.get("smallarea_status") == "1"]:
            try:
                area = _get(f"data/{code}", {"year": year})
            except requests.HTTPError as exc:
                logger.warning("data %s %s failed: %s", code, year, exc)
                continue
            inner = area.get("data", {})  # type: ignore[union-attr]
            basic = inner.get("basicdata", [])
            if not isinstance(basic, list):
                continue
            area_name = inner.get("area_name", "")
            for r in basic:
                z = str(r.get("zip_code", ""))
                if not z.isdigit():
                    continue  # skips "MSA level"
                rows.append({"zip": z.zfill(5), "year": year,
                             "fmr_area_code": code, "area_name": area_name,
                             **_to_numeric_br(r)})
            time.sleep(0.25)  # pace HUD's rate limiter
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["zip", "year"]).reset_index(drop=True)[
        ["zip", "year", "fmr_area_code", "area_name", *_BR_COLS]
    ]


def fetch_fmr_by_county(year: int) -> pd.DataFrame:
    """Pull county- and metro-area FMRs nationwide for ``year``.

    One row per (area_fips, year). ``area_fips`` is the 10-digit
    ``state(2)+county(3)+sub(5)`` HUD FIPS for counties, or the
    ``METRO...`` code for metro FMR areas; ``state_fips`` / ``county_fips``
    are sliced from ``fips_code`` (blank for metros).
    """
    if year < 2017:
        raise ValueError(f"FMR API year coverage starts at 2017; got year={year}")
    rows: list[dict] = []
    for state in _list_states():
        try:
            payload = _get(f"statedata/{state}", {"year": year})
        except requests.HTTPError as exc:
            logger.warning("statedata %s %s failed: %s", state, year, exc)
            continue
        time.sleep(0.2)
        data = payload.get("data", {})  # type: ignore[union-attr]
        for c in data.get("counties", []):
            fips = str(c.get("fips_code") or "").zfill(10)
            rows.append({"area_fips": fips, "state_fips": fips[:2],
                         "county_fips": fips[2:5],
                         "state_code": c.get("statecode") or state,
                         "area_name": c.get("county_name"),
                         "metro_name": c.get("metro_name"),
                         "kind": "county", "year": year, **_to_numeric_br(c)})
        for m in data.get("metroareas", []):
            rows.append({"area_fips": m.get("code"), "state_fips": "",
                         "county_fips": "",
                         "state_code": m.get("statecode") or state,
                         "area_name": m.get("metro_name"),
                         "metro_name": m.get("metro_name"),
                         "kind": "metro", "year": year, **_to_numeric_br(m)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["area_fips", "year"]).reset_index(drop=True)[
        ["area_fips", "state_fips", "county_fips", "state_code",
         "area_name", "metro_name", "kind", "year", *_BR_COLS]
    ]


def _parquet_path(by: str) -> Path:
    if by not in {"zip", "county"}:
        raise ValueError(f"by must be 'zip' or 'county'; got {by!r}")
    return PROCESSED / f"hud_fmr_{by}_year.parquet"


def build_and_save(years: list[int], by: str = "zip") -> Path:
    """Fetch FMR for every year in ``years`` and write to processed parquet."""
    fetch = fetch_fmr_by_zip if by == "zip" else fetch_fmr_by_county
    frames = []
    for y in years:
        logger.info("Fetching HUD FMR %s for year=%s", by, y)
        df = fetch(y)
        logger.info("  -> %d rows", len(df))
        frames.append(df)
    if not frames:
        raise RuntimeError("No years requested")
    full = pd.concat(frames, ignore_index=True)
    out = _parquet_path(by)
    full.to_parquet(out, index=False)
    logger.info("Wrote %s (%d rows)", out, len(full))
    return out


def load_hud_fmr(by: str = "zip") -> pd.DataFrame:
    """Load the cached HUD FMR parquet. Raises if missing."""
    path = _parquet_path(by)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Call build_and_save(years=[...], by={by!r}) first."
        )
    return pd.read_parquet(path)
