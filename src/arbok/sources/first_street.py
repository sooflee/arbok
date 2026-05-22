"""Natural-hazard risk scores at tract + county level (FEMA NRI substitution).

Phase 1's starter pack originally listed "First Street flood + fire score".
First Street's API is now paid (their CEJST partnership was discontinued in
2024), so this module substitutes the **FEMA National Risk Index (NRI) v1.20**
— a free, full-coverage dataset of 18 natural hazards at Census tract + county
level. The module name is kept (`first_street.py`) so the predictor catalog
and notebooks keep resolving. A paid First Street API path is left as a TODO
for future parcel/property-level granularity NRI's tracts can't match.

Source: FEMA's static `hazards.fema.gov/.../NRI_Table_*.zip` URLs were
deprecated when FEMA migrated to Drupal in 2025/2026 (they now return an
HTML splash page). The same NRI v1.20 data is republished as CSV by the
federal **resilience.climate.gov** ArcGIS Hub (FEMA org `XG15cJAlne2vxtgt`),
which serves real CSVs (~465 MB tracts, ~19 MB counties).

Hub label -> FEMA short code (per the NRI technical documentation):

    Inland Flooding - Hazard Type Risk Index Score   -> RFLD_RISKS (riverine)
    Coastal Flooding - Hazard Type Risk Index Score  -> CFLD_RISKS
    Wildfire - Hazard Type Risk Index Score          -> WFIR_RISKS
    Hurricane - Hazard Type Risk Index Score         -> HRCN_RISKS
    Heat Wave - Hazard Type Risk Index Score         -> HWAV_RISKS
    National Risk Index - Score - Composite          -> RISK_SCORE
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

NRI_DIR = RAW / "fema_nri"
NRI_DIR.mkdir(parents=True, exist_ok=True)

# resilience.climate.gov ArcGIS Hub item IDs for FEMA NRI v1.20 (Dec 2025).
NRI_TRACT_URL = (
    "https://resilience.climate.gov/api/download/v1/items/"
    "9da4eeb936544335a6db0cd7a8448a51/csv?layers=0"
)
NRI_COUNTY_URL = (
    "https://resilience.climate.gov/api/download/v1/items/"
    "39485e8035d446a5bff03259508ae355/csv?layers=0"
)

# Hub-label -> FEMA short code. Keep only the hazards we surface as features.
_HAZARD_RENAME: dict[str, str] = {
    "Inland Flooding - Hazard Type Risk Index Score": "RFLD_RISKS",
    "Coastal Flooding - Hazard Type Risk Index Score": "CFLD_RISKS",
    "Wildfire - Hazard Type Risk Index Score": "WFIR_RISKS",
    "Hurricane - Hazard Type Risk Index Score": "HRCN_RISKS",
    "Heat Wave - Hazard Type Risk Index Score": "HWAV_RISKS",
    "National Risk Index - Score - Composite": "RISK_SCORE",
}

# Download chunk size — NRI tract CSV is ~465 MB so stream to disk.
_CHUNK = 1 << 20  # 1 MiB


def _download(url: str, dest: Path) -> Path:
    """Stream-download `url` to `dest` (skip if file exists and is non-empty)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)
    return dest


def fetch_nri_tract() -> pd.DataFrame:
    """Download (cached) FEMA NRI v1.20 tract-level CSV and select risk columns.

    Returns a tract-level DataFrame with at least:

        tract_geoid (11-digit string), state_fips, county_fips, state, county,
        RFLD_RISKS, CFLD_RISKS, WFIR_RISKS, HRCN_RISKS, HWAV_RISKS, RISK_SCORE

    Missing values are preserved as `NaN` (NRI marks hazards as not-applicable
    for geographies where they cannot occur — e.g. coastal flood inland).
    """
    path = _download(NRI_TRACT_URL, NRI_DIR / "NRI_Table_CensusTracts.csv")
    keep = ["Census Tract FIPS Code", "State FIPS Code", "County FIPS Code",
            "State Name", "County Name", *_HAZARD_RENAME.keys()]
    raw = pd.read_csv(path, usecols=keep, dtype={"Census Tract FIPS Code": str,
                                                 "State FIPS Code": str,
                                                 "County FIPS Code": str})
    out = raw.rename(columns={
        "Census Tract FIPS Code": "tract_geoid",
        "State FIPS Code": "state_fips",
        "County FIPS Code": "county_fips",
        "State Name": "state",
        "County Name": "county",
        **_HAZARD_RENAME,
    })
    # NRI tract FIPS arrive without the leading state zero -> normalise to 11.
    out["tract_geoid"] = out["tract_geoid"].str.zfill(11)
    out["state_fips"] = out["state_fips"].str.zfill(2)
    out["county_fips"] = out["county_fips"].str.zfill(3)
    return out


def fetch_nri_county() -> pd.DataFrame:
    """Download (cached) FEMA NRI v1.20 county-level CSV and select risk columns.

    Returns a county-level DataFrame with at least:

        state_fips, county_fips, county_geoid (5-digit), state, county,
        RFLD_RISKS, CFLD_RISKS, WFIR_RISKS, HRCN_RISKS, HWAV_RISKS, RISK_SCORE
    """
    path = _download(NRI_COUNTY_URL, NRI_DIR / "NRI_Table_Counties.csv")
    keep = ["State FIPS Code", "County FIPS Code", "State-County FIPS Code",
            "State Name", "County Name", *_HAZARD_RENAME.keys()]
    raw = pd.read_csv(path, usecols=keep, dtype={"State FIPS Code": str,
                                                 "County FIPS Code": str,
                                                 "State-County FIPS Code": str})
    out = raw.rename(columns={
        "State FIPS Code": "state_fips",
        "County FIPS Code": "county_fips",
        "State-County FIPS Code": "county_geoid",
        "State Name": "state",
        "County Name": "county",
        **_HAZARD_RENAME,
    })
    out["state_fips"] = out["state_fips"].str.zfill(2)
    out["county_fips"] = out["county_fips"].str.zfill(3)
    out["county_geoid"] = out["county_geoid"].str.zfill(5)
    return out


def derive_starter_risk_features(tract_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-hazard tract frame to the three starter-pack features.

    Returns one row per tract:

        tract_geoid, flood_risk_combined, wildfire_risk, composite_risk

    `flood_risk_combined` = max of RFLD_RISKS (riverine) and CFLD_RISKS
    (coastal). NaN in either column is treated as 0 for the max, but a tract
    with both missing stays NaN — this avoids inflating inland tracts'
    flood score while still flagging coastal-only or river-only exposure.
    """
    if "tract_geoid" not in tract_df.columns:
        raise KeyError("tract_df must contain 'tract_geoid' (call fetch_nri_tract first)")
    rfld = pd.to_numeric(tract_df.get("RFLD_RISKS"), errors="coerce")
    cfld = pd.to_numeric(tract_df.get("CFLD_RISKS"), errors="coerce")
    both_missing = rfld.isna() & cfld.isna()
    flood_combined = pd.concat([rfld, cfld], axis=1).max(axis=1)
    flood_combined = flood_combined.where(~both_missing, other=pd.NA)
    return pd.DataFrame({
        "tract_geoid": tract_df["tract_geoid"].astype(str),
        "flood_risk_combined": flood_combined.astype("Float64"),
        "wildfire_risk": pd.to_numeric(tract_df.get("WFIR_RISKS"),
                                       errors="coerce").astype("Float64"),
        "composite_risk": pd.to_numeric(tract_df.get("RISK_SCORE"),
                                        errors="coerce").astype("Float64"),
    })


def build_and_save() -> tuple[Path, Path]:
    """Fetch both NRI levels, write parquet copies under data/processed/.

    Returns (tract_path, county_path). The tract parquet also keeps the
    raw per-hazard columns; if you only need the three starter-pack features
    derived in `derive_starter_risk_features`, pull from the parquet and apply
    it (cheap on a single ~85K-row frame).
    """
    tract = fetch_nri_tract()
    county = fetch_nri_county()
    tract_path = PROCESSED / "fema_nri_tract.parquet"
    county_path = PROCESSED / "fema_nri_county.parquet"
    tract.to_parquet(tract_path, index=False)
    county.to_parquet(county_path, index=False)
    print(f"[fema_nri] wrote {len(tract):,} tract rows -> {tract_path}")
    print(f"[fema_nri] wrote {len(county):,} county rows -> {county_path}")
    return tract_path, county_path


def load_fema_nri(level: str = "tract") -> pd.DataFrame:
    """Convenience loader for the processed parquet (level: 'tract' or 'county').

    Runs `build_and_save()` lazily on first call if the parquet is missing.
    """
    if level not in ("tract", "county"):
        raise ValueError(f"level must be 'tract' or 'county', got {level!r}")
    path = PROCESSED / f"fema_nri_{level}.parquet"
    if not path.exists():
        build_and_save()
    return pd.read_parquet(path)
