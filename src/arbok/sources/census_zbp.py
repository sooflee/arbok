"""Census Zip Business Patterns — total establishments + employment per ZIP per year.

Free Census API, same key as ACS. ZBP publishes annual snapshots at ZIP-code level
(NOT ZCTA — actual USPS zips, less leading-zero pain).

Per-zip features:
- ``zbp__estab``: total establishments
- ``zbp__emp``: total paid employment
- ``zbp__payann_thousands``: annual payroll in $1k
- ``zbp__estab_per_capita``: derived later when joined with ACS pop

API note: ZBP availability halted after 2021 (replaced by County Business Patterns
expansion + Census Business Builder). Most recent zip-level vintage = 2018 in the
plain ``/zbp`` endpoint; from 2019 onward, ZIP-level data is in ``/cbp`` with a
``for=zipcode:*`` parameter. We try the modern endpoint first, fall back.
"""
from __future__ import annotations

import os

import pandas as pd
import requests

from arbok.config import PROCESSED


def fetch_zbp(year: int = 2018) -> pd.DataFrame:
    """Fetch total-industry ZBP rollup at ZIP level for a single year.

    ZIP-level ZBP is only published through vintage 2018 (later years are
    county-level only in CBP). For modeling we accept the most-recent zip-level
    snapshot as a stable structural feature.
    """
    if year > 2018:
        raise ValueError(f"ZIP-level ZBP not available past 2018; got {year}")
    api_key = os.environ.get("CENSUS_API_KEY")
    base = f"https://api.census.gov/data/{year}/zbp"
    params = {
        "get": "ESTAB,EMP",
        "for": "zipcode:*",
        "NAICS2017": "00",
    }
    if api_key:
        params["key"] = api_key
    r = requests.get(base, params=params, timeout=180)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={"zip code": "zip"})
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    for c in ("ESTAB", "EMP"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.rename(columns={"ESTAB": "zbp_estab", "EMP": "zbp_emp"})
    df["year"] = year
    return df[["zip", "year", "zbp_estab", "zbp_emp"]]


def build_and_save(years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        try:
            frames.append(fetch_zbp(y))
            print(f"  zbp {y}: ok")
        except Exception as e:
            print(f"  zbp {y}: FAILED ({type(e).__name__}: {e})")
    if not frames:
        raise RuntimeError("No ZBP years succeeded")
    out = pd.concat(frames, ignore_index=True)
    path = PROCESSED / "zbp_zip_year.parquet"
    out.to_parquet(path, index=False)
    print(f"Saved zbp: {out.shape} -> {path}")
    return out


def load_zbp() -> pd.DataFrame:
    path = PROCESSED / "zbp_zip_year.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_and_save([...years]) first.")
    return pd.read_parquet(path)
