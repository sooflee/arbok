"""FCC BDC (Broadband Data Collection) availability loader, census-tract level.

Form 477's census-block deployment series was retired after H1-2022 and
replaced by the BDC, published twice a year via the National Broadband Map.
Differences vs. Form 477: units are Broadband Serviceable Locations (BSLs)
from the CostQuest Fabric (not households); three explicit speed tiers
(Served >=100/20, Underserved 25/3-100/20, Unserved); providers de-duplicated
by Holding Company (expect lower counts than 477); per-technology rows must
be filtered to the all-technology aggregate before per-tract roll-up.

Version-string convention: short labels like `Jun2024`, `Dec2023` (case-
insensitive) map to the FCC filename
`bdc_us_fixed_broadband_summary_by_geography_{as_of}_{revision}.zip`.
Aggregation levels in this family: National, State, County, Census Tract,
Place, CBSA, Congressional District, Tribal. We target Census Tract — the
smallest aggregate the FCC publishes without dropping to GB-scale per-block
files. Tract->ZIP is handled later via the HUD ZIP-TRACT crosswalk.

MANUAL DOWNLOAD REQUIRED. The portal requires a free FCC User Registration
account + a generated API token; direct file URLs are signed + short-lived.
Steps: (1) account at https://broadbandmap.fcc.gov (Sign In, top right);
(2) visit MANUAL_DOWNLOAD_URL, download the "Summary by Geography Type" ZIP
(typically 20-60 MB); (3) save as data/raw/fcc_bdc/{version}.zip
(e.g. Jun2024.zip — exact name). `_COLUMN_ALIASES` covers source-column
variants observed across Jun2022-Jun2024 vintages; extend on KeyError.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from arbok.config import PROCESSED, RAW

FCC_BDC_DIR = RAW / "fcc_bdc"
FCC_BDC_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_DOWNLOAD_URL = "https://broadbandmap.fcc.gov/data-download/nationwide-data"

# Canonical pattern per bdc-data-downloads-output.pdf. Concrete URLs are
# signed + short-lived; we keep the template for reference only.
BDC_SUMMARY_URL_TEMPLATE = (
    "https://broadbandmap.fcc.gov/nbm/map/api/national_map_process/"
    "nbm_get_nationwide_file/Fixed_Broadband/"
    "bdc_us_fixed_broadband_summary_by_geography_{as_of}_{revision}.zip"
)

# Half-year -> (month, day-of-month-end). H1 = Jun 30, H2 = Dec 31.
_VERSION_MONTH_END = {"jun": ("06", "30"), "dec": ("12", "31")}
_TRACT_CSV_PATTERNS = ("*tract*.csv", "*census_tract*.csv")

# Source-column variants observed across BDC Jun2022 / Dec2022 / Jun2023 /
# Dec2023 / Jun2024 releases (lowercased, stripped).
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "tract_geoid": ("geography_id", "geography_id_designation", "tract_id", "geoid", "geo_id"),
    "total_units": ("total_units", "total_locations", "total_bsl", "unit_count"),
    "providers_count_25_3": ("providers_25_3", "providers_count_25_3", "num_provider_25_3", "unique_provider_count_25_3"),
    "providers_count_100_20": ("providers_100_20", "providers_count_100_20", "num_provider_100_20", "unique_provider_count_100_20"),
    "providers_count_1000_100": ("providers_1000_100", "providers_count_1000_100", "num_provider_1000_100", "unique_provider_count_1000_100"),
    "pct_served_25_3": ("pct_25_3", "pct_served_25_3", "percent_25_3", "percent_served_25_3"),
    "pct_served_100_20": ("pct_100_20", "pct_served_100_20", "percent_100_20", "percent_served_100_20"),
    "pct_served_1000_100": ("pct_1000_100", "pct_served_1000_100", "percent_1000_100", "percent_served_1000_100"),
    "technology": ("technology", "technology_code", "tech_code"),
    "geography_type": ("geography_type", "geo_type", "geography_type_code"),
}


def _normalize_version(version: str) -> str:
    """Canonicalize 'jun2024' / 'Jun-2024' / 'JUN_2024' -> 'Jun2024'."""
    m = re.fullmatch(r"\s*([A-Za-z]{3})[\s_\-]*([0-9]{4})\s*", version)
    if not m or m.group(1).lower() not in _VERSION_MONTH_END:
        raise ValueError(f"Bad BDC version {version!r}; expected 'Jun2024' / 'Dec2023'")
    return f"{m.group(1).capitalize()}{m.group(2)}"


def _effective_date(version: str) -> str:
    v = _normalize_version(version)
    mm, dd = _VERSION_MONTH_END[v[:3].lower()]
    return f"{v[3:]}-{mm}-{dd}"


def _zip_path(version: str) -> Path:
    return FCC_BDC_DIR / f"{_normalize_version(version)}.zip"


def _resolve(lower_cols: dict[str, str], canonical: str) -> str | None:
    return next((lower_cols[a] for a in _COLUMN_ALIASES[canonical] if a in lower_cols), None)


def _extract_tract_csv(zip_path: Path) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for pat in _TRACT_CSV_PATTERNS:
            rx = re.compile(pat.replace("*", ".*"), re.IGNORECASE)
            for name in names:
                if rx.fullmatch(Path(name).name):
                    return zf.read(name)
    raise FileNotFoundError(f"No tract CSV in {zip_path.name}; saw {names[:8]}...")


def fetch_bdc_tract_summary(version: str = "Jun2024") -> pd.DataFrame:
    """Load BDC tract-level Summary by Geography Type for one release.

    Reads `data/raw/fcc_bdc/{version}.zip` (manual download — see module
    docstring), filters to the all-technology tract rows when present, and
    returns columns: tract_geoid (11-digit str), version, total_units,
    providers_count_25_3 / _100_20 / _1000_100,
    pct_served_25_3 / _100_20 / _1000_100.
    """
    version = _normalize_version(version)
    path = _zip_path(version)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing BDC summary ZIP: {path}\nManually download the 'Summary by "
            f"Geography Type' ZIP for {version} from {MANUAL_DOWNLOAD_URL} (free "
            f"FCC account + API token required) and save it there. "
            f"Canonical pattern: {BDC_SUMMARY_URL_TEMPLATE}"
        )

    raw = pd.read_csv(io.BytesIO(_extract_tract_csv(path)), dtype=str, low_memory=False)
    lower_map = {c.strip().lower(): c for c in raw.columns}

    def _filter(canonical: str, allowed: set[str]) -> None:
        nonlocal raw
        col = _resolve(lower_map, canonical)
        if col is None:
            return
        mask = raw[col].astype(str).str.strip().str.lower().isin(allowed)
        if mask.any():
            raw = raw[mask].copy()

    _filter("technology", {"all", "0", "any", "all_technology"})
    _filter("geography_type", {"tract", "census_tract", "census tract"})

    geoid_col = _resolve(lower_map, "tract_geoid")
    if geoid_col is None:
        raise KeyError(f"No tract-geoid column in {path.name}; cols={list(raw.columns)[:20]}")

    def _num(canonical: str) -> pd.Series:
        col = _resolve(lower_map, canonical)
        if col is None:
            return pd.Series([pd.NA] * len(raw), dtype="Float64")
        return pd.to_numeric(raw[col], errors="coerce").astype("Float64")

    out = pd.DataFrame({
        "tract_geoid": raw[geoid_col].astype(str).str.strip().str.zfill(11),
        "version": version,
        "total_units": _num("total_units"),
        "providers_count_25_3": _num("providers_count_25_3"),
        "providers_count_100_20": _num("providers_count_100_20"),
        "providers_count_1000_100": _num("providers_count_1000_100"),
        "pct_served_25_3": _num("pct_served_25_3"),
        "pct_served_100_20": _num("pct_served_100_20"),
        "pct_served_1000_100": _num("pct_served_1000_100"),
    })
    return out[out["tract_geoid"].str.fullmatch(r"\d{11}")].reset_index(drop=True)


def derive_starter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tract: broadband_avail_pct (>=100/20), gigabit_avail_pct (>=1000/100),
    provider_count (max across tiers; median understates since tiers nest)."""
    prov = ["providers_count_25_3", "providers_count_100_20", "providers_count_1000_100"]
    out = df[["tract_geoid", "version"]].copy()
    out["broadband_avail_pct"] = df["pct_served_100_20"]
    out["gigabit_avail_pct"] = df["pct_served_1000_100"]
    out["provider_count"] = df[prov].max(axis=1)
    return out


def build_and_save(versions: list[str]) -> Path:
    """Build long tract-level panel across versions to
    `data/processed/fcc_broadband_tract.parquet`. Columns: tract_geoid,
    version, effective_date, total_units, providers_count_* (3 tiers),
    pct_served_* (3 tiers), broadband_avail_pct, gigabit_avail_pct,
    provider_count.
    """
    frames: list[pd.DataFrame] = []
    for v in versions:
        df = fetch_bdc_tract_summary(v)
        merged = df.merge(derive_starter_features(df), on=["tract_geoid", "version"], how="left")
        merged["effective_date"] = _effective_date(v)
        frames.append(merged)
    panel = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED / "fcc_broadband_tract.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"[fcc_broadband] wrote {len(panel):,} rows for versions={versions} to {out_path}")
    return out_path


def load_fcc_broadband() -> pd.DataFrame:
    """Read the saved tract-level panel; raise if `build_and_save` hasn't run."""
    path = PROCESSED / "fcc_broadband_tract.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No processed FCC broadband panel at {path}. Run build_and_save([...]) first.")
    return pd.read_parquet(path)
