"""NCES Common Core of Data (CCD) loader, district (LEA) level.

Public-school enrollment by district is the tract-mappable demographics
signal in `docs/PREDICTORS.md`. CCD is the annual federal census of every
US public K-12 LEA (~18.6K in SY 2022-23); SY 1986-87 to 2024-25 download.

Per district-year (`leaid` + `school_year_start`) we publish:
- total_enrollment       fall K-12 membership (F-33 V33, fallback MEMBERSCH)
- frpm_pct               % FRPM-eligible, aggregated from school-level
                         Lunch file (ccd_sch_033)
- teacher_fte_count      FTE teachers (derived "Teachers" row, ccd_lea_059)
- student_teacher_ratio  total_enrollment / teacher_fte_count
- dropout_count          grades 9-12 dropouts (ccd_lea_032)
- total_current_expend, per_pupil_spending  F-33 TCURELSC, /-by-V33
- county_fips, cbsa      F-33 CONUM/CBSA (geocoded LEA central office)

SEDA math+reading z-scores would augment this; as of 2026-05
edopportunity.org gates downloads behind email+DUA, so `fetch_seda_district`
is NotImplementedError.

District -> zip: F-33 already carries county FIPS per LEA; the EDGE file
`EDGE_GEOCODE_PUBLICLEA_<YY><YY+1>` adds CBSA + lat/lon + mailing zip.
Aggregate to county (enrollment-weighted rates, summed counts), then
broadcast to zip via the HUD zip<->county crosswalk — districts often span
many zips, so county is the safer intermediate. File discovery uses the
JSON API at /ccd/datatables/api/File/{fiscal}/{level}/{schoolYearId}/0/0/0.
"""
from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

CACHE_DIR = RAW / "nces_schools"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = PROCESSED / "nces_schools_district_year.parquet"

UA = "Mozilla/5.0 (arbok research; contact via project README)"
TIMEOUT = 600

CCD_FILE_API = "https://nces.ed.gov/ccd/datatables/api/File/{fiscal}/{level}/{syid}/0/0/0"
CCD_LOOKUP_API = "https://nces.ed.gov/ccd/datatables/api/Lookup/"
EDGE_GEOCODE_URL = "https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICLEA_{yy}{yy2}.zip"


@lru_cache(maxsize=1)
def _school_year_id_map() -> dict[int, int]:
    """Starting calendar year (e.g. 2022 for SY 2022-23) -> API schoolYearId."""
    r = requests.get(CCD_LOOKUP_API, headers={"User-Agent": UA}, timeout=60); r.raise_for_status()
    return {int(e["schoolYearShort"]): int(e["id"]) for e in r.json().get("schoolYears", [])
            if e.get("schoolYearShort") and e["schoolYearShort"].isdigit()}


def _discover_file_urls(year: int, fiscal: int, level: int) -> dict[str, str]:
    """{component_name: zip_url} for one (fiscal, level, school_year_start)."""
    syid = _school_year_id_map().get(year)
    if syid is None:
        raise ValueError(f"CCD has no school year starting {year}")
    r = requests.get(CCD_FILE_API.format(fiscal=fiscal, level=level, syid=syid),
                     headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    out: dict[str, str] = {}
    for sg in r.json().get("selectionGroupModels", []):
        for tg in sg.get("titleGroups", []):
            for eg in tg.get("elementGroups", []):
                if eg.get("elementType") != "Data File":
                    continue
                for cg in eg.get("componentGroups", []):
                    comp = cg.get("component") or tg.get("title", "")
                    for f in cg.get("files", []):
                        url = f.get("fileURL", "")
                        # F-33 ships multiple formats per component; keep the text variant.
                        if fiscal == 1 and "_txt" not in url.lower():
                            continue
                        out.setdefault(comp, url)
    return out


def _download(url: str) -> Path:
    """Cache `url` under data/raw/nces_schools/."""
    path = CACHE_DIR / url.rsplit("/", 1)[-1]
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"[nces_schools] downloading {url}")
    with requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        with path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    return path


def _read_zipped(zip_path: Path, sep: str = ",", **kwargs) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith((".csv", ".txt")))
        with zf.open(name) as fh:
            return pd.read_csv(fh, sep=sep, low_memory=False, **kwargs)


def _frpm_per_lea(year: int) -> pd.DataFrame:
    """Aggregate school-level lunch eligibility to LEA: sum(free+reduced)/sum(total)."""
    url = next((u for k, u in _discover_file_urls(year, 2, 7).items() if "Lunch" in k), None)
    if not url:
        return pd.DataFrame(columns=["leaid", "frpm_pct"])
    df = _read_zipped(_download(url), usecols=["LEAID", "LUNCH_PROGRAM", "STUDENT_COUNT",
                                                "TOTAL_INDICATOR", "DATA_GROUP"],
                      dtype={"LEAID": str})
    df = df[df["DATA_GROUP"] == "Free and Reduced-price Lunch Table"]
    df["STUDENT_COUNT"] = pd.to_numeric(df["STUDENT_COUNT"], errors="coerce").fillna(0)
    denom = df[df["TOTAL_INDICATOR"] == "Education Unit Total"].groupby("LEAID")["STUDENT_COUNT"].sum()
    num = df[df["LUNCH_PROGRAM"].isin(["Free lunch qualified", "Reduced-price lunch qualified"])] \
            .groupby("LEAID")["STUDENT_COUNT"].sum()
    out = denom.to_frame("denom").join(num.rename("num"), how="left").fillna({"num": 0}).reset_index()
    out["frpm_pct"] = (out["num"] / out["denom"].replace(0, pd.NA)) * 100
    return out.rename(columns={"LEAID": "leaid"})[["leaid", "frpm_pct"]]


def _fiscal_per_lea(year: int) -> pd.DataFrame:
    """F-33 fiscal -> enrollment, county FIPS, per-pupil $$. Sentinels (-1/-2/-3/-9) -> NaN."""
    url = next(iter(_discover_file_urls(year, 1, 5).values()), None)
    if not url:
        return pd.DataFrame()
    df = _read_zipped(_download(url), sep="\t",
                      usecols=["LEAID", "CONUM", "CBSA", "V33", "MEMBERSCH", "TOTALEXP", "TCURELSC"],
                      dtype={"LEAID": str, "CONUM": str, "CBSA": str}) \
        .rename(columns={"LEAID": "leaid", "CONUM": "county_fips", "CBSA": "cbsa"})
    for col in ("V33", "MEMBERSCH", "TOTALEXP", "TCURELSC"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = pd.NA
    df["total_enrollment"] = df["V33"].combine_first(df["MEMBERSCH"])
    df["per_pupil_spending"] = df["TCURELSC"] / df["total_enrollment"]
    return df.rename(columns={"TOTALEXP": "total_expenditure", "TCURELSC": "total_current_expend"})[
        ["leaid", "county_fips", "cbsa", "total_enrollment", "total_current_expend",
         "total_expenditure", "per_pupil_spending"]]


def fetch_ccd_district(year: int) -> pd.DataFrame:
    """Per-district panel for the school year starting `year`.

    Joins LEA Directory + F-33 fiscal + LEA Staff (derived "Teachers" row) +
    school-level FRPM aggregation + LEA Dropout. The full LEA Membership
    file (ccd_lea_052) is skipped — it unzips to 1.3 GB of grade x sex x race
    breakdowns but headline counts are already in F-33's V33/MEMBERSCH.
    """
    nonfiscal = _discover_file_urls(year, fiscal=2, level=5)
    if not nonfiscal.get("Directory"):
        raise RuntimeError(f"No CCD LEA Directory file for SY starting {year}")
    directory = _read_zipped(
        _download(nonfiscal["Directory"]),
        usecols=["LEAID", "LEA_NAME", "FIPST", "ST", "LEA_TYPE", "SY_STATUS"],
        dtype={"LEAID": str, "FIPST": str},
    ).rename(columns={"LEAID": "leaid", "LEA_NAME": "district_name",
                      "FIPST": "state_fips", "ST": "state_abbr"})
    teachers = pd.DataFrame(columns=["leaid", "teacher_fte_count"])
    if (u := nonfiscal.get("Staff")):
        d = _read_zipped(_download(u), usecols=["LEAID", "STAFF", "STAFF_COUNT", "TOTAL_INDICATOR"],
                         dtype={"LEAID": str})
        d = d[(d["STAFF"] == "Teachers") & d["TOTAL_INDICATOR"].str.startswith("Derived", na=False)]
        teachers = d.assign(teacher_fte_count=pd.to_numeric(d["STAFF_COUNT"], errors="coerce")) \
                    .rename(columns={"LEAID": "leaid"})[["leaid", "teacher_fte_count"]]
    dropouts = pd.DataFrame(columns=["leaid", "dropout_count"])
    if (u := next((v for k, v in nonfiscal.items() if "Dropout" in (k or "")), None)):
        d = _read_zipped(_download(u), usecols=["LEAID", "STUDENT_COUNT", "TOTAL_INDICATOR"],
                         dtype={"LEAID": str})
        d = d[d["TOTAL_INDICATOR"] == "Education Unit Total"]
        dropouts = d.assign(dropout_count=pd.to_numeric(d["STUDENT_COUNT"], errors="coerce")) \
                    .rename(columns={"LEAID": "leaid"})[["leaid", "dropout_count"]]
    panel = (directory.merge(_fiscal_per_lea(year), on="leaid", how="left")
                     .merge(teachers, on="leaid", how="left")
                     .merge(_frpm_per_lea(year), on="leaid", how="left")
                     .merge(dropouts, on="leaid", how="left"))
    panel["student_teacher_ratio"] = (
        panel["total_enrollment"] / panel["teacher_fte_count"].where(panel["teacher_fte_count"] > 0)
    )
    panel["school_year_start"] = year
    return panel


def fetch_seda_district() -> pd.DataFrame:
    """SEDA math+reading z-scores per district — not implemented (DUA-gated)."""
    raise NotImplementedError("SEDA requires DUA registration; fetch from edopportunity.org.")


_EDGE_RENAME = {"LEAID": "leaid", "NAME": "district_name", "STFIP": "state_fips",
                "CNTY": "county_fips", "NMCNTY": "county_name", "CBSA": "cbsa",
                "NMCBSA": "cbsa_name", "ZIP": "mailing_zip", "LAT": "lat", "LON": "lon"}


def fetch_edge_geocode(year: int) -> pd.DataFrame:
    """NCES EDGE geocodes: leaid -> county FIPS, CBSA, lat/lon, mailing zip."""
    yy, yy2 = f"{year % 100:02d}", f"{(year + 1) % 100:02d}"
    with zipfile.ZipFile(_download(EDGE_GEOCODE_URL.format(yy=yy, yy2=yy2))) as zf:
        xlsx_name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        with zf.open(xlsx_name) as fh:
            df = pd.read_excel(fh, usecols=list(_EDGE_RENAME),
                               dtype={k: str for k in _EDGE_RENAME if k not in ("LAT", "LON")})
    return df.rename(columns=_EDGE_RENAME)


def aggregate_to_county(
    districts_df: pd.DataFrame,
    district_to_county_xwalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Roll district panel to (county_fips, school_year_start): weighted rates, summed counts."""
    weighted = ("frpm_pct", "per_pupil_spending", "student_teacher_ratio")
    sums = ("total_enrollment", "teacher_fte_count", "dropout_count",
            "total_current_expend", "total_expenditure")
    if district_to_county_xwalk is not None:
        districts_df = districts_df.drop(columns=["county_fips"], errors="ignore").merge(
            district_to_county_xwalk[["leaid", "county_fips"]], on="leaid", how="left")
    df = districts_df.dropna(subset=["county_fips", "school_year_start"]).copy()
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    w = df["total_enrollment"].fillna(0)
    for c in weighted:
        df[f"_w_{c}"] = df[c] * w
    agg = {**{c: "sum" for c in sums}, "state_fips": "first",
           **{f"_w_{c}": "sum" for c in weighted}}
    out = df.groupby(["county_fips", "school_year_start"], as_index=False).agg(agg)
    for c in weighted:
        out[c] = out[f"_w_{c}"] / out["total_enrollment"].replace(0, pd.NA)
    return out.drop(columns=[f"_w_{c}" for c in weighted])


def build_and_save(years: list[int]) -> Path:
    """Build and persist the district-year panel as parquet."""
    panel = pd.concat([fetch_ccd_district(y) for y in years], ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"[nces_schools] wrote {len(panel):,} district-year rows to {OUT_PATH}")
    return OUT_PATH


def load_schools() -> pd.DataFrame:
    """Read the processed district-year panel. Raises if not yet built."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} not found. Run `nces_schools.build_and_save([...])` first."
        )
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":  # pragma: no cover
    build_and_save([2022]); print(load_schools().head())
