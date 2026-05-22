"""Foursquare Open Source Places loader (snapshot, global POI parquet).

Foursquare publishes ~100M global POIs under Apache-2.0. We count amenity
anchors per US ZIP: Whole Foods / Trader Joe's / Costco / Wegmans (the
"Whole Foods class" predictor) plus third-wave coffee and breweries
(speculative density signals — PREDICTORS.md §5).

SNAPSHOT, NOT A TIME SERIES. The OS release is a single dated snapshot
(e.g. dt=2026-05-14). Temporal Δ ("WF count YoY") is NOT directly derivable
from one file. Future enhancements: (1) cache successive monthly snapshots
and diff `fsq_place_id` sets, exploiting `date_closed`/`date_refreshed`;
(2) cross-reference OSM history for opening dates.

MANUAL DOWNLOAD REQUIRED (~204 GB full; HF gates behind a TOS click-through):
  1. Accept terms at https://huggingface.co/datasets/foursquare/fsq-os-places
  2. Pick the latest `release/dt=YYYY-MM-DD/places/parquet/*.parquet`. For a
     US-only subset use `huggingface_hub.snapshot_download` or pyarrow.dataset
     country=='US' pushdown (~5-10 GB). Mirror: `s3://fsq-os-places/`.
  3. Save *.parquet files to `data/raw/foursquare/` (any count; concatenated).

ZIP: `postcode` is 5-digit US ZIP for US rows; read as string to preserve
leading zeros (e.g. `02118`). ZIP+4 suffix is stripped.

QUALITY: national chains (WF, Costco, Starbucks) are near-complete; small
independents (third-wave coffee, neighbourhood breweries) under-count in
less-mapped metros — treat as noisy *relative* signals. Third-wave is
approximated as coffee_shop minus a hand-picked chain allowlist; a proper
specialty-brand allowlist (Blue Bottle, Intelligentsia, Stumptown, ...) is
a future enhancement.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from arbok.config import PROCESSED, RAW

FOURSQUARE_DIR = RAW / "foursquare"
FOURSQUARE_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_DOWNLOAD_URL = "https://huggingface.co/datasets/foursquare/fsq-os-places"
S3_MIRROR = "s3://fsq-os-places/"

# Brand name aliases — case-insensitive substring match on `name`.
BRANDS: dict[str, list[str]] = {
    "whole_foods": ["whole foods"],
    "trader_joes": ["trader joe"],  # matches "Trader Joe's" with/without apostrophe
    "costco": ["costco"],
    "wegmans": ["wegmans"],
}

# Foursquare v2025 category IDs (numeric strings, as in the OS taxonomy).
# Verify against the live taxonomy CSV at
# https://docs.foursquare.com/data-products/docs/categories after each release.
#   13035 = Coffee Shop (Dining & Drinking > Cafes, Coffee, and Tea Houses)
#   13029 = Brewery     (Dining & Drinking > Bar)
CATEGORIES: dict[str, list[str]] = {
    "coffee_shop": ["13035"],
    "brewery": ["13029"],
}

# Chains to subtract when approximating "third-wave coffee" density.
COFFEE_CHAIN_EXCLUDES: list[str] = [
    "starbucks",
    "dunkin",
    "peet",
    "tim hortons",
    "caribou coffee",
    "dutch bros",
    "the coffee bean",
    "scooter's coffee",
    "biggby coffee",
]

_KEEP_COLS = [
    "fsq_id",
    "name",
    "latitude",
    "longitude",
    "address",
    "locality",
    "region",
    "postcode",
    "category",
    "category_ids",
]


def _discover_parquet_files(path: Path | None) -> list[Path]:
    if path is not None and path.is_file():
        return [path]
    base = path if path is not None else FOURSQUARE_DIR
    files = sorted(base.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files in {base}. Download a release from "
            f"{MANUAL_DOWNLOAD_URL} (or {S3_MIRROR}) and save *.parquet to {FOURSQUARE_DIR}/."
        )
    return files


def load_places(path: Path | None = None) -> pd.DataFrame:
    """Load Foursquare OS Places parquet(s) and return a normalised DataFrame.

    Reads `data/raw/foursquare/*.parquet` by default (or the file/dir at
    `path`). Filters to US rows (country == 'US') and returns the columns
    listed in `_KEEP_COLS`. `postcode` is preserved as a 5-char zero-padded
    string. `category` is the first / most-granular `fsq_category_ids` entry.
    """
    files = _discover_parquet_files(path)
    # Foursquare schema: `fsq_place_id` + `fsq_category_ids` (array).
    source_cols = [
        "fsq_place_id",
        "name",
        "latitude",
        "longitude",
        "address",
        "locality",
        "region",
        "postcode",
        "country",
        "fsq_category_ids",
    ]
    frames: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_parquet(fp, columns=source_cols, dtype_backend="pyarrow")
        df["postcode"] = df["postcode"].astype("string")
        df = df[df["country"].astype("string").str.upper() == "US"].copy()
        frames.append(df)
    places = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # Strip ZIP+4 suffix ("02118-1234") and keep the leading 5 digits.
    places["postcode"] = (
        places["postcode"].astype("string").str.extract(r"^(\d{5})", expand=False)
    )

    def _first(x: object) -> str | None:
        if x is None:
            return None
        try:
            if len(x) == 0:  # type: ignore[arg-type]
                return None
            return str(x[0])  # type: ignore[index]
        except TypeError:
            return None

    places["category"] = places["fsq_category_ids"].map(_first).astype("string")
    places["category_ids"] = places["fsq_category_ids"]
    places = places.rename(columns={"fsq_place_id": "fsq_id"})
    return places[_KEEP_COLS]


def _name_matches_any(names: pd.Series, needles: list[str]) -> pd.Series:
    """Vectorised case-insensitive substring OR-match across a list of needles."""
    n = names.astype("string").str.lower().fillna("")
    mask = pd.Series(False, index=n.index)
    for needle in needles:
        mask = mask | n.str.contains(needle.lower(), regex=False, na=False)
    return mask


def _group_count(hits: pd.DataFrame) -> pd.DataFrame:
    hits = hits.dropna(subset=["postcode"])
    return (
        hits.groupby(hits["postcode"].astype("string"))
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"postcode": "zip"})
    )


def count_brand_per_zip(places: pd.DataFrame, brand_key: str) -> pd.DataFrame:
    """Count rows whose `name` matches any alias for `brand_key`, per ZIP.
    Returns columns: `zip`, `count`."""
    if brand_key not in BRANDS:
        raise KeyError(f"Unknown brand {brand_key!r}. Known: {sorted(BRANDS)}")
    return _group_count(places[_name_matches_any(places["name"], BRANDS[brand_key])])


def count_category_per_zip(
    places: pd.DataFrame,
    category_key: str,
    exclude_chains: bool = False,
) -> pd.DataFrame:
    """Count POIs whose primary category OR any `category_ids` entry is in
    CATEGORIES[key]. If `exclude_chains`, drop rows whose `name` matches any
    `COFFEE_CHAIN_EXCLUDES` entry. Returns columns: `zip`, `count`."""
    if category_key not in CATEGORIES:
        raise KeyError(f"Unknown category {category_key!r}. Known: {sorted(CATEGORIES)}")
    wanted = set(CATEGORIES[category_key])
    primary = places["category"].astype("string").isin(wanted)

    def _any_match(x: object) -> bool:
        if x is None:
            return False
        try:
            return any(str(c) in wanted for c in x)  # type: ignore[union-attr]
        except TypeError:
            return False

    secondary = places["category_ids"].map(_any_match)
    hits = places[primary | secondary]
    if exclude_chains:
        hits = hits[~_name_matches_any(hits["name"], COFFEE_CHAIN_EXCLUDES)]
    return _group_count(hits)


def derive_starter_features() -> pd.DataFrame:
    """Build the Phase-1 Foursquare feature row per ZIP and write to processed/.

    Columns: zip, wf_count, tj_count, costco_count, wegmans_count,
             third_wave_coffee_count, brewery_count.
    Writes `data/processed/foursquare_pois_zip.parquet`.
    """
    places = load_places()

    parts: dict[str, pd.DataFrame] = {
        "wf_count": count_brand_per_zip(places, "whole_foods"),
        "tj_count": count_brand_per_zip(places, "trader_joes"),
        "costco_count": count_brand_per_zip(places, "costco"),
        "wegmans_count": count_brand_per_zip(places, "wegmans"),
        "third_wave_coffee_count": count_category_per_zip(
            places, "coffee_shop", exclude_chains=True
        ),
        "brewery_count": count_category_per_zip(places, "brewery"),
    }

    out: pd.DataFrame | None = None
    for col, df in parts.items():
        df = df.rename(columns={"count": col})
        out = df if out is None else out.merge(df, on="zip", how="outer")
    assert out is not None
    feature_cols = list(parts.keys())
    out[feature_cols] = out[feature_cols].fillna(0).astype("int64")
    out = out.sort_values("zip").reset_index(drop=True)

    dest = PROCESSED / "foursquare_pois_zip.parquet"
    out.to_parquet(dest, index=False)
    return out


def load_foursquare_pois() -> pd.DataFrame:
    """Convenience: load the processed per-zip feature table, building it if absent."""
    dest = PROCESSED / "foursquare_pois_zip.parquet"
    if not dest.exists():
        return derive_starter_features()
    return pd.read_parquet(dest)
