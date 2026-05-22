"""BLS Quarterly Census of Employment & Wages (QCEW) loader, county level.

QCEW is the closest thing to a full-coverage payroll census in the US:
~95% of jobs (every employer that pays state UI tax) reported quarter-by-
quarter, with hard counts of establishments, monthly employment, and wages.
Per `docs/PREDICTORS.md`, QCEW wage growth is the primary jobs-and-income
signal for the zip return model — broadcast county -> zip via crosswalk.

Public landing page:
    https://www.bls.gov/cew/downloadable-data-files.htm

The "singlefile" annual zip used here packs all four quarters of one calendar
year into a single ~2 GB CSV (~310 MB compressed). URL pattern (verified
2026-05; the BLS landing page lists this under "CSV Files By Industry / By
Area / Singlefile"):

    https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip

BLS does not publish per-quarter singlefiles. Sibling bundles in the same
directory: `*_qtrly_by_area.zip`, `*_qtrly_by_industry.zip` (same data
pre-split), and `{year}_annual_singlefile.zip` (annual averages, not
quarterly observations). We pull the qtrly singlefile and filter by `qtr`.

Key source columns (full schema is ~42 cols; we keep a subset):

    area_fips        5-char str. State(2)+County(3). Also non-county aggregates:
                     "US000", "{ST}000" (state), "C{xxxx}" (MSA), "{ST}999".
    own_code         Ownership: 0=total covered, 1=federal, 2=state, 3=local,
                     5=private. own_code=0 already sums 1+2+3+5 — never re-sum.
    industry_code    NAICS. "10" is BLS's "all industries" pseudo-code (not
                     a real NAICS sector); 2-6 digit NAICS at lower agglvls.
    agglvl_code      70 = County / total covered / all industries (own=0,
                     ind=10), one row per county per quarter. 71-78 = county
                     subdivisions; 40-48 MSA; 50-58 state; 10-18 national.
    disclosure_code  "N" = BLS-suppressed for confidentiality (small cell).
                     Numerics publish as 0 in suppressed rows; we coerce to NA.
                     Rare at agglvl 70, common in tiny counties / fine NAICS.
    avg_wkly_wage    total_qtrly_wages / (mean monthly emp * 13). Renamed
                     `avg_weekly_wage` on output.
    qtrly_estabs     Establishment count. Renamed `qtrly_estabs_count`.
    month{1,2,3}_emplvl, total_qtrly_wages — straight through.

Annual revisions overwrite prior-year values when a new vintage drops; cache
by year and re-fetch if you need the latest revision.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

QCEW_DIR = RAW / "bls_qcew"
QCEW_DIR.mkdir(parents=True, exist_ok=True)

SINGLEFILE_URL = "https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip"

# agglvl_code 70 = "County, total covered, all industries" — one row per
# county per quarter. This is the row we want for county-level wage growth.
COUNTY_TOTAL_AGGLVL = 70

# Output schema. Maps source CSV column -> exported column name.
_OUTPUT_COLUMN_MAP: dict[str, str] = {
    "area_fips": "area_fips",
    "industry_code": "industry_code",
    "own_code": "own_code",
    "avg_wkly_wage": "avg_weekly_wage",
    "month1_emplvl": "month1_emplvl",
    "month2_emplvl": "month2_emplvl",
    "month3_emplvl": "month3_emplvl",
    "total_qtrly_wages": "total_qtrly_wages",
    "qtrly_estabs": "qtrly_estabs_count",
}
_NUMERIC_OUTPUT_COLS = [
    "avg_weekly_wage",
    "month1_emplvl",
    "month2_emplvl",
    "month3_emplvl",
    "total_qtrly_wages",
    "qtrly_estabs_count",
]


def _singlefile_zip_path(year: int) -> Path:
    return QCEW_DIR / f"{year}_qtrly_singlefile.zip"


def _download_singlefile(year: int) -> Path:
    """Download the year's qtrly singlefile zip (~300 MB) if not cached."""
    path = _singlefile_zip_path(year)
    if path.exists() and path.stat().st_size > 0:
        return path
    url = SINGLEFILE_URL.format(year=year)
    print(f"[bls_qcew] downloading {url} -> {path}")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    return path


def fetch_qcew_quarterly(year: int, quarter: int) -> pd.DataFrame:
    """Return one quarter of county-aggregate QCEW rows (agglvl_code=70).

    Downloads the year's qtrly singlefile zip into `data/raw/bls_qcew/` (cached
    across calls), streams the inner ~2 GB CSV, and keeps only county-total
    rows for the requested quarter. ~3.2K counties per quarter.

    Returned columns:
        area_fips, state_fips, county_fips, year_quarter, industry_code,
        own_code, avg_weekly_wage, month1_emplvl, month2_emplvl, month3_emplvl,
        total_qtrly_wages, qtrly_estabs_count
    """
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"quarter must be 1-4, got {quarter}")

    zip_path = _download_singlefile(year)

    # Stream-filter the CSV: read in chunks and keep only rows for the target
    # quarter at agglvl 70 (one row per county). Cuts ~14.6M rows -> ~3.2K.
    chunks: list[pd.DataFrame] = []
    keep_cols = list(_OUTPUT_COLUMN_MAP.keys()) + ["agglvl_code", "year", "qtr", "disclosure_code"]
    with zipfile.ZipFile(zip_path) as zf:
        inner_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(inner_name) as fh:
            for chunk in pd.read_csv(
                fh,
                dtype={"area_fips": str, "industry_code": str, "disclosure_code": str},
                usecols=keep_cols,
                chunksize=500_000,
                low_memory=False,
            ):
                mask = (chunk["agglvl_code"] == COUNTY_TOTAL_AGGLVL) & (chunk["qtr"] == quarter)
                if mask.any():
                    chunks.append(chunk.loc[mask].copy())

    if not chunks:
        raise RuntimeError(f"No agglvl_code=70 rows found for {year}Q{quarter}")
    df = pd.concat(chunks, ignore_index=True)

    # Suppressed cells (disclosure_code="N") publish as zero; surface as NA so
    # downstream YoY math doesn't compute spurious 100% drops.
    suppressed = df["disclosure_code"].fillna("").str.upper() == "N"
    for src_col in _OUTPUT_COLUMN_MAP:
        if src_col in {"area_fips", "industry_code", "own_code"}:
            continue
        df.loc[suppressed, src_col] = pd.NA

    out = df.rename(columns=_OUTPUT_COLUMN_MAP)[list(_OUTPUT_COLUMN_MAP.values())].copy()
    out["area_fips"] = out["area_fips"].astype(str).str.strip().str.zfill(5)
    out["state_fips"] = out["area_fips"].str[:2]
    out["county_fips"] = out["area_fips"].str[2:]
    out["year_quarter"] = f"{year}Q{quarter}"
    for col in _NUMERIC_OUTPUT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out[
        [
            "area_fips",
            "state_fips",
            "county_fips",
            "year_quarter",
            "industry_code",
            "own_code",
            "avg_weekly_wage",
            "month1_emplvl",
            "month2_emplvl",
            "month3_emplvl",
            "total_qtrly_wages",
            "qtrly_estabs_count",
        ]
    ]


def aggregate_to_yoy_wage_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Compute YoY % change in `avg_weekly_wage` per (county, quarter).

    Input is the long-form panel from `fetch_qcew_quarterly` (one or more
    quarters concatenated). YoY is same-calendar-quarter, prior year, which
    strips QCEW's strong Q1/Q4 seasonality (bonuses, retail surge).

    Output columns:
        area_fips, state_fips, county_fips, year_quarter, year, quarter,
        avg_weekly_wage, avg_weekly_wage_yoy_pct
    """
    work = df[["area_fips", "state_fips", "county_fips", "year_quarter", "avg_weekly_wage"]].copy()
    work[["_year", "_q"]] = work["year_quarter"].str.split("Q", expand=True)
    work["year"] = work["_year"].astype(int)
    work["quarter"] = work["_q"].astype(int)
    work = work.drop(columns=["_year", "_q"])

    work = work.sort_values(["area_fips", "quarter", "year"]).reset_index(drop=True)
    work["prev_wage"] = work.groupby(["area_fips", "quarter"])["avg_weekly_wage"].shift(1)
    work["prev_year"] = work.groupby(["area_fips", "quarter"])["year"].shift(1)
    # Only flag as YoY if the previous record was exactly 1 year earlier.
    valid = work["prev_year"] == (work["year"] - 1)
    work["avg_weekly_wage_yoy_pct"] = pd.NA
    ratio = work.loc[valid, "avg_weekly_wage"] / work.loc[valid, "prev_wage"].replace(0, pd.NA)
    work.loc[valid, "avg_weekly_wage_yoy_pct"] = (ratio - 1) * 100

    return work[
        [
            "area_fips",
            "state_fips",
            "county_fips",
            "year_quarter",
            "year",
            "quarter",
            "avg_weekly_wage",
            "avg_weekly_wage_yoy_pct",
        ]
    ]


def build_and_save(years: list[int]) -> Path:
    """Build the county-quarter QCEW panel and persist as parquet.

    Pulls every quarter of every year in `years`, concatenates, attaches YoY
    wage growth, and writes `data/processed/qcew_county_quarter.parquet`.
    """
    frames: list[pd.DataFrame] = []
    for year in years:
        for quarter in (1, 2, 3, 4):
            frames.append(fetch_qcew_quarterly(year, quarter))
    panel = pd.concat(frames, ignore_index=True)
    yoy = aggregate_to_yoy_wage_growth(panel)
    full = panel.merge(
        yoy[["area_fips", "year_quarter", "avg_weekly_wage_yoy_pct"]],
        on=["area_fips", "year_quarter"],
        how="left",
    )
    out_path = PROCESSED / "qcew_county_quarter.parquet"
    full.to_parquet(out_path, index=False)
    print(f"[bls_qcew] wrote {len(full):,} rows for years={years} to {out_path}")
    return out_path


def load_qcew() -> pd.DataFrame:
    """Read the processed county-quarter panel. Raises if not yet built."""
    path = PROCESSED / "qcew_county_quarter.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `bls_qcew.build_and_save([...])` first."
        )
    return pd.read_parquet(path)
