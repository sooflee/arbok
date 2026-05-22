"""NOAA VIIRS Day/Night Band nighttime-lights loader (zip-level zonal stats).

Source: Colorado School of Mines Earth Observation Group (EOG), "VIIRS
Nighttime Lights" (VNL) — https://eogdata.mines.edu/products/vnl/

We use the **VNL V2.1 annual global composites**, the standard cleaned product
(stray-light corrected, cloud-masked, biomass-burning-filtered, with an
"average_masked" layer that zeroes background and ephemeral lights). Files are
single-band, 15 arc-second (~500m at equator), unsigned-int32 GeoTIFFs in
EPSG:4326 (geographic lat/lon, WGS84). Per-year file layout::

    https://eogdata.mines.edu/nighttime_light/annual/v21/{year}/
        VNL_v21_npp_{year}_global_vcmslcfg_c<stamp>.average_masked.dat.tif.gz

The `<stamp>` is a per-release timestamp that the EOG mints when a year's
composite is published, so we have to list the directory rather than guess.

Auth: EOG OAuth2 password grant. Register (free) at
https://eogdata.mines.edu/products/register/ then set::

    EOG_USER_TOKEN              # a pre-fetched bearer token, OR
    EOG_USER + EOG_PASS         # plus EOG_CLIENT_ID / EOG_CLIENT_SECRET

Tokens expire after ~5 minutes — for any real backfill, run this once
interactively, cache the GeoTIFFs locally, then re-process from disk.

Workflow:
    1. fetch_viirs_annual(year)     -> GeoTIFF on disk
    2. fetch_zcta_shapefile(vintage) -> Census TIGER ZCTA polygons
    3. zonal_mean_radiance(tif, shp) -> per-ZCTA mean DNB radiance (nW/cm^2/sr)
    4. build_year / build_and_save   -> parquet of (zcta, year, mean_radiance)
    5. derive_yoy_delta              -> the actual model feature

The raster is unprojected lat/lon; ZCTA shapefile is also EPSG:4326 — no
reprojection needed for zonal stats, but areas (sr-equivalent pixels) are
slightly biased high at low latitudes. For YoY *deltas* (the feature we
actually feed the model) this bias cancels, so we do NOT area-weight.
"""
from __future__ import annotations

import gzip
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

VIIRS_DIR = RAW / "noaa_viirs"
ZCTA_DIR = RAW / "census_zcta"
VIIRS_DIR.mkdir(parents=True, exist_ok=True)
ZCTA_DIR.mkdir(parents=True, exist_ok=True)

EOG_ANNUAL_BASE = "https://eogdata.mines.edu/nighttime_light/annual/v21"
VIIRS_ANNUAL_URL_TEMPLATE = (
    EOG_ANNUAL_BASE + "/{year}/VNL_v21_npp_{year}_global_vcmslcfg_c{stamp}.average_masked.dat.tif.gz"
)
EOG_TOKEN_URL = "https://eogauth-new.mines.edu/realms/eog/protocol/openid-connect/token"
EOG_REGISTER_URL = "https://eogdata.mines.edu/products/register/"

CENSUS_ZCTA_URL_TEMPLATE = (
    "https://www2.census.gov/geo/tiger/TIGER{vintage}/ZCTA520/tl_{vintage}_us_zcta520.zip"
)


def _eog_token() -> str:
    """Return a bearer token: prefer EOG_USER_TOKEN, else password-grant."""
    tok = os.environ.get("EOG_USER_TOKEN")
    if tok:
        return tok
    user = os.environ.get("EOG_USER")
    pw = os.environ.get("EOG_PASS")
    cid = os.environ.get("EOG_CLIENT_ID", "eogdata_oidc")
    csec = os.environ.get("EOG_CLIENT_SECRET", "")
    if not (user and pw):
        raise RuntimeError(
            "EOG auth missing. Set EOG_USER_TOKEN, or EOG_USER + EOG_PASS "
            f"(+ EOG_CLIENT_ID / EOG_CLIENT_SECRET). Register: {EOG_REGISTER_URL}"
        )
    r = requests.post(
        EOG_TOKEN_URL,
        data={
            "client_id": cid,
            "client_secret": csec,
            "username": user,
            "password": pw,
            "grant_type": "password",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _resolve_annual_url(year: int, token: str) -> str:
    """Find the dated filename for `year` by listing the year directory."""
    dir_url = f"{EOG_ANNUAL_BASE}/{year}/"
    r = requests.get(dir_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    pat = re.compile(rf"VNL_v21_npp_{year}_global_vcmslcfg_c(\d+)\.average_masked\.dat\.tif\.gz")
    stamps = sorted(set(pat.findall(r.text)))
    if not stamps:
        raise RuntimeError(f"No average_masked GeoTIFF found in {dir_url}")
    return VIIRS_ANNUAL_URL_TEMPLATE.format(year=year, stamp=stamps[-1])


def fetch_viirs_annual(year: int) -> Path:
    """Download the VNL V2.1 annual composite GeoTIFF for `year`, cached.

    Requires `EOG_USER_TOKEN` (or EOG_USER/EOG_PASS) — register free at
    https://eogdata.mines.edu/products/register/. Returns the path to the
    decompressed .tif on disk (the .gz is ~1.5 GB, the .tif ~3 GB).
    """
    tif_path = VIIRS_DIR / f"VNL_v21_npp_{year}_average_masked.tif"
    if tif_path.exists():
        return tif_path
    token = _eog_token()
    url = _resolve_annual_url(year, token)
    gz_path = VIIRS_DIR / f"VNL_v21_npp_{year}_average_masked.tif.gz"
    with requests.get(
        url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=600
    ) as r:
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with gzip.open(gz_path, "rb") as src, open(tif_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink(missing_ok=True)
    return tif_path


def fetch_zcta_shapefile(vintage: int = 2023) -> Path:
    """Download + unzip the Census TIGER/Line ZCTA520 shapefile (~500MB)."""
    url = CENSUS_ZCTA_URL_TEMPLATE.format(vintage=vintage)
    zip_path = ZCTA_DIR / f"tl_{vintage}_us_zcta520.zip"
    shp_path = ZCTA_DIR / f"tl_{vintage}_us_zcta520" / f"tl_{vintage}_us_zcta520.shp"
    if shp_path.exists():
        return shp_path
    if not zip_path.exists():
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    import zipfile

    extract_dir = shp_path.parent
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return shp_path


def zonal_mean_radiance(raster_path: Path, zcta_shapefile_path: Path) -> pd.DataFrame:
    """Mean DNB radiance per ZCTA polygon.

    Output: DataFrame with `zcta` (5-digit str) and `mean_radiance`
    (nW/cm^2/sr). Polygons with no valid pixels are dropped. Requires
    `rasterio`, `rasterstats`, and `geopandas` — install with::

        uv add rasterio rasterstats
    """
    try:
        import geopandas as gpd
        from rasterstats import zonal_stats
    except ImportError as e:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "zonal_mean_radiance requires rasterio, rasterstats, and geopandas. "
            "Install with:  uv add rasterio rasterstats   "
            "(geopandas is already in the [geo] extras: uv sync --extra geo)"
        ) from e

    zctas = gpd.read_file(zcta_shapefile_path)
    zcta_col = next((c for c in ("ZCTA5CE20", "ZCTA5CE10", "GEOID20") if c in zctas.columns), None)
    if zcta_col is None:
        raise KeyError(f"No ZCTA id column in shapefile; saw {list(zctas.columns)}")

    # Raster + ZCTA shapefile are both EPSG:4326 — no reprojection needed.
    stats = zonal_stats(
        vectors=zctas.geometry,
        raster=str(raster_path),
        stats=["mean"],
        nodata=-999.0,
        all_touched=False,
        geojson_out=False,
    )
    out = pd.DataFrame(
        {
            "zcta": zctas[zcta_col].astype(str).str.zfill(5),
            "mean_radiance": [s.get("mean") if s else None for s in stats],
        }
    )
    return out.dropna(subset=["mean_radiance"]).reset_index(drop=True)


def build_year(year: int) -> pd.DataFrame:
    """Full per-year pipeline: download raster + ZCTAs, compute zonal mean."""
    tif = fetch_viirs_annual(year)
    shp = fetch_zcta_shapefile()
    df = zonal_mean_radiance(tif, shp)
    df["year"] = year
    return df[["zcta", "year", "mean_radiance"]]


def build_and_save(years: list[int]) -> Path:
    """Build a multi-year ZCTA-level radiance panel and write to parquet."""
    frames = [build_year(y) for y in years]
    panel = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED / "viirs_zcta_year.parquet"
    panel.to_parquet(out_path, index=False)
    return out_path


def derive_yoy_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ZCTA year-over-year delta in mean radiance (the model feature).

    Adds `mean_radiance_prev`, `radiance_yoy_abs`, and `radiance_yoy_pct`.
    Pct is computed against a floor (max(prev, 0.1)) to keep dim rural ZCTAs
    from blowing up to ±inf — VNL radiance is in nW/cm^2/sr and dim ZCTAs
    routinely sit near 0.
    """
    df = df.sort_values(["zcta", "year"]).copy()
    df["mean_radiance_prev"] = df.groupby("zcta")["mean_radiance"].shift(1)
    df["radiance_yoy_abs"] = df["mean_radiance"] - df["mean_radiance_prev"]
    floor = df["mean_radiance_prev"].clip(lower=0.1)
    df["radiance_yoy_pct"] = df["radiance_yoy_abs"] / floor
    return df.dropna(subset=["mean_radiance_prev"]).reset_index(drop=True)


def load_viirs_nightlights() -> pd.DataFrame:
    """Load the processed ZCTA-year radiance panel with YoY deltas.

    Convenience reader: returns the parquet built by `build_and_save`,
    enriched with `derive_yoy_delta`. Raises FileNotFoundError with a hint
    if the panel hasn't been built yet.
    """
    path = PROCESSED / "viirs_zcta_year.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run noaa_viirs.build_and_save([...years]) first "
            "(requires EOG_USER_TOKEN and rasterio/rasterstats installed)."
        )
    return derive_yoy_delta(pd.read_parquet(path))
