"""USAspending.gov federal contract + grant outlays, aggregated to recipient zip5.

USAspending publishes every federal award (contract, grant, loan, etc.) with
recipient location (zip, congressional district, county). We want annual
award $ flowing INTO each zip as a "sticky federal money" predictor.

Endpoint reality (verified live, 2026-05): the documented
`/api/v2/search/spending_by_geography/` endpoint only supports
`geo_layer` of {state, county, district, country} — zip5 is NOT exposed
("Field 'geo_layer' is outside valid values"). `/api/v2/recipient/duns/`
returns recipient totals but strips location.

Only path that yields true zip-level aggregation: the async bulk-download
endpoint `/api/v2/bulk_download/awards/`, which produces a zipped CSV of
prime transactions including `recipient_zip_4_code` and
`federal_action_obligation`. We POST a fiscal-year window, poll the
status endpoint, stream the CSV in chunks, and aggregate by recipient zip5.

Pitfalls:
- Contract CSVs use `recipient_zip_4_code` (a 9-digit string, ZIP+4 glued
  together, or a 5-digit ZIP, or blank, or a foreign postal code).
  Assistance CSVs use `recipient_zip_code` (5 digits). We normalize both
  to a 5-digit zip5 via leading-digit regex and drop blanks / non-US.
- Place-of-performance zip is also present but we deliberately use recipient
  zip (where the dollars *flow to*, not where the work happens).
- Each award appears multiple times (one row per transaction / modification).
  We sum `federal_action_obligation` (which is the delta per transaction)
  to get net obligated dollars, and count UNIQUE award keys for n_awards.
- Time filter is action_date, by federal fiscal year (Oct 1 – Sep 30).
- Download is async: 2 days of all-agencies prime transactions = ~140MB,
  ~5 min wait. A full FY of all award types is multi-GB and tens of minutes
  to hours. We cache the zip locally and stream-aggregate in chunks.
- USAspending search API only goes back to FY2008.

Ref: https://api.usaspending.gov/docs/endpoints
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

BULK_URL = "https://api.usaspending.gov/api/v2/bulk_download/awards/"
STATUS_URL = "https://api.usaspending.gov/api/v2/download/status"

USASP_DIR = RAW / "usaspending"
USASP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED / "usaspending_zip_year.parquet"

# Award-type buckets per USAspending data dictionary.
AWARD_TYPE_CODES: dict[str, list[str]] = {
    "contracts": ["A", "B", "C", "D"],
    "grants": ["02", "03", "04", "05"],
    "loans": ["07", "08"],
    "other": ["06", "09", "10", "11"],
    "all": ["A", "B", "C", "D", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"],
}

CHUNK_ROWS = 200_000
POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 60 * 30  # 30 min hard cap


def _fiscal_year_window(fiscal_year: int) -> tuple[str, str]:
    """US federal FY runs Oct 1 of prior calendar year through Sep 30."""
    return f"{fiscal_year - 1}-10-01", f"{fiscal_year}-09-30"


def _request_download(fiscal_year: int, award_type: str) -> dict:
    if award_type not in AWARD_TYPE_CODES:
        raise ValueError(f"award_type must be one of {list(AWARD_TYPE_CODES)}")
    start, end = _fiscal_year_window(fiscal_year)
    body = {
        "filters": {
            "prime_award_types": AWARD_TYPE_CODES[award_type],
            "date_type": "action_date",
            "date_range": {"start_date": start, "end_date": end},
        },
        "file_format": "csv",
    }
    r = requests.post(BULK_URL, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def _wait_for_file(file_name: str) -> str:
    """Poll status endpoint until 'finished'; return the file_url."""
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        r = requests.get(STATUS_URL, params={"file_name": file_name}, timeout=30)
        r.raise_for_status()
        status = r.json()
        if status["status"] == "finished":
            return status["file_url"]
        if status["status"] == "failed":
            raise RuntimeError(f"USAspending bulk download failed: {status.get('message')}")
        print(f"[usaspending]   status={status['status']} elapsed={status.get('seconds_elapsed')}s")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"USAspending download {file_name} did not finish in {POLL_TIMEOUT_S}s")


def _download_zip(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    print(f"[usaspending]   downloading {url} -> {dest.name}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def _read_csv_columns(fh) -> tuple[str, str]:
    """Sniff which zip + award-key column names this CSV uses.

    Contract CSVs:  recipient_zip_4_code  + contract_award_unique_key
    Assistance:     recipient_zip_code    + assistance_award_unique_key
    """
    header = pd.read_csv(fh, nrows=0).columns.tolist()
    fh.seek(0)
    zip_col = "recipient_zip_4_code" if "recipient_zip_4_code" in header else "recipient_zip_code"
    key_col = (
        "contract_award_unique_key" if "contract_award_unique_key" in header
        else "assistance_award_unique_key"
    )
    return zip_col, key_col


def _aggregate_csv_in_zip(zip_path: Path, fiscal_year: int) -> pd.DataFrame:
    """Stream every CSV in the archive, aggregating obligations by recipient zip5."""
    obl_totals: dict[str, float] = {}
    award_sets: dict[str, set] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in (n for n in zf.namelist() if n.lower().endswith(".csv")):
            print(f"[usaspending]   aggregating {name}")
            with zf.open(name) as sniff_fh:
                zip_col, key_col = _read_csv_columns(sniff_fh)
            with zf.open(name) as fh:
                reader = pd.read_csv(
                    fh,
                    usecols=[zip_col, "federal_action_obligation", key_col],
                    dtype={zip_col: "string", key_col: "string"},
                    chunksize=CHUNK_ROWS,
                    low_memory=False,
                )
                for chunk in reader:
                    zip5 = (
                        chunk[zip_col].astype("string").str.replace("-", "", regex=False)
                        .str.extract(r"^(\d{5})", expand=False)
                    )
                    obl = pd.to_numeric(chunk["federal_action_obligation"], errors="coerce").fillna(0.0)
                    sub = pd.DataFrame({"zip": zip5, "obl": obl, "k": chunk[key_col]}).dropna(subset=["zip"])
                    grouped = sub.groupby("zip", sort=False)
                    for z, grp in grouped:
                        obl_totals[z] = obl_totals.get(z, 0.0) + float(grp["obl"].sum())
                        award_sets.setdefault(z, set()).update(grp["k"].dropna().tolist())
    zips = list(obl_totals.keys())
    out = pd.DataFrame({
        "zip": zips,
        "fiscal_year": fiscal_year,
        "total_obligations_usd": [obl_totals[z] for z in zips],
        "n_awards": [len(award_sets[z]) for z in zips],
    })
    out["n_awards"] = out["n_awards"].astype("Int64")
    return out.sort_values("total_obligations_usd", ascending=False).reset_index(drop=True)


def fetch_zip_outlays(fiscal_year: int, award_type: str = "all") -> pd.DataFrame:
    """One row per recipient zip5 for the given FY.

    Columns: zip, fiscal_year, total_obligations_usd, n_awards.
    Caches the zipped CSV under data/raw/usaspending/.
    """
    cache_zip = USASP_DIR / f"fy{fiscal_year}_{award_type}.zip"
    if not cache_zip.exists():
        print(f"[usaspending] requesting FY{fiscal_year} {award_type} bulk download")
        meta = _request_download(fiscal_year, award_type)
        url = _wait_for_file(meta["file_name"])
        _download_zip(url, cache_zip)
    return _aggregate_csv_in_zip(cache_zip, fiscal_year)


def build_and_save(years: list[int], award_type: str = "all") -> Path:
    """Build and save the multi-year zip-level outlay panel."""
    frames = []
    for fy in years:
        frames.append(fetch_zip_outlays(fy, award_type=award_type))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["fiscal_year", "total_obligations_usd"], ascending=[True, False])
    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"[usaspending] wrote {len(panel):,} zip-year rows to {OUTPUT_PATH}")
    return OUTPUT_PATH


def load_usaspending() -> pd.DataFrame:
    """Convenience loader: build the parquet if missing, then read it."""
    if not OUTPUT_PATH.exists():
        build_and_save([pd.Timestamp.today().year - 1])
    return pd.read_parquet(OUTPUT_PATH)
