"""Build an interactive Folium choropleth of ZIP-level decile rank for the
post-2012 LightGBM fwd_3y model. Output: data/processed/zip_decile_map.html.

Wave-3 task #9 — taking over from the stalled subagent.
"""
from __future__ import annotations

import time
from pathlib import Path

import folium
import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap

from arbok.config import PROCESSED, RAW

POST_2012_LGBM = PROCESSED / "phase2_artifacts" / "fwd_3y__post_2012__lgbm.txt"
POST_2012_FEATS = PROCESSED / "phase2_artifacts" / "fwd_3y__post_2012__lgbm.features.txt"
TIGER_SHP = RAW / "census_zcta" / "tl_2020_us_zcta520.shp"
OUT_HTML = PROCESSED / "zip_decile_map.html"


def load_model() -> tuple[lgb.Booster, list[str]]:
    booster = lgb.Booster(model_file=str(POST_2012_LGBM))
    feats = POST_2012_FEATS.read_text().strip().split("\n")
    return booster, feats


def score_most_recent_month(booster: lgb.Booster, features: list[str]) -> pd.DataFrame:
    """Score every zip at the latest month where every required feature has data.

    Mirrors score_zips.py: load the feature store, pick the latest month with
    >=80% coverage on the modeling features, predict.
    """
    fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
    # Find the latest month with adequate coverage on the modeling features
    have = [c for c in features if c in fs.columns]
    if len(have) < len(features):
        missing = set(features) - set(have)
        print(f"  [warn] {len(missing)} model features missing from feature store: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    # Find latest month with high coverage
    month_col = "year_month"
    months = sorted(fs[month_col].unique())
    chosen = None
    for m in reversed(months):
        slice_ = fs[fs[month_col] == m]
        if slice_.empty:
            continue
        cov = slice_[have].notna().mean().mean()
        if cov >= 0.6:
            chosen = m
            break
    if chosen is None:
        raise RuntimeError("No month with adequate feature coverage")
    print(f"  scoring at month: {chosen}")
    snap = fs[fs[month_col] == chosen].copy()

    # Top-100 metros filter
    top_metros = pd.read_csv(PROCESSED / "top_metros.csv")
    keep_cbsa = set(top_metros["cbsa"].astype(str))
    snap["cbsa"] = snap["cbsa"].astype(str)
    snap = snap[snap["cbsa"].isin(keep_cbsa)].copy()
    print(f"  {len(snap)} zips in {len(keep_cbsa)} top metros")

    X = snap[have].astype(float).fillna(snap[have].median(numeric_only=True))
    # Pad missing model features with median imputation (zeros if no median)
    for f in features:
        if f not in X.columns:
            X[f] = 0.0
    X = X[features]
    pred = booster.predict(X)
    snap["pred"] = pred
    snap["decile"] = pd.qcut(snap["pred"].rank(method="first"), q=10, labels=False)
    return snap[["zip", "cbsa", "zhvi", "pred", "decile"]].reset_index(drop=True)


def load_simplified_shapes(zip_set: set[str]) -> gpd.GeoDataFrame:
    """Load ZCTA shapefile filtered to the zips we care about, simplified for size."""
    print(f"  reading shapefile (this is the big step)...")
    t0 = time.time()
    # Read only the columns we need; pyogrio is fast.
    gdf = gpd.read_file(TIGER_SHP, columns=["ZCTA5CE20", "geometry"])
    print(f"  shapefile loaded ({len(gdf)} polygons) in {time.time()-t0:.1f}s")
    gdf = gdf.rename(columns={"ZCTA5CE20": "zip"})
    gdf["zip"] = gdf["zip"].astype(str).str.zfill(5)
    gdf = gdf[gdf["zip"].isin(zip_set)].copy()
    print(f"  filtered to {len(gdf)} matching polygons")
    # Simplify to keep HTML size down
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.005, preserve_topology=True)
    return gdf


def build_map(scored: pd.DataFrame, shapes: gpd.GeoDataFrame) -> folium.Map:
    """Render a Folium choropleth keyed by zip decile, with a tooltip per zip."""
    merged = shapes.merge(scored, on="zip", how="inner")
    print(f"  merged scored + shapes: {len(merged)} zips with geometry")

    # Add metro display name for tooltips
    top_metros = pd.read_csv(PROCESSED / "top_metros.csv")
    top_metros["cbsa"] = top_metros["cbsa"].astype(str)
    if "core_name" in top_metros.columns:
        top_metros["metro_name"] = top_metros["core_name"]
    else:
        top_metros["metro_name"] = top_metros["cbsa"]
    merged = merged.merge(top_metros[["cbsa", "metro_name"]], on="cbsa", how="left")

    # Center on CONUS
    m = folium.Map(location=[39.5, -98.5], zoom_start=4, tiles="CartoDB positron")

    # Diverging colormap: red (bottom decile) → grey (mid) → green (top decile)
    cmap = LinearColormap(
        colors=["#a50026", "#f46d43", "#fdae61", "#fee08b", "#ffffbf",
                "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"],
        vmin=0, vmax=9, caption="Predicted fwd_3y return decile (9 = top 10%)",
    )
    cmap.add_to(m)

    def style_fn(feat):
        decile = feat["properties"].get("decile")
        if decile is None or pd.isna(decile):
            return {"fillColor": "#cccccc", "color": "#666", "weight": 0.2, "fillOpacity": 0.5}
        return {
            "fillColor": cmap(int(decile)),
            "color": "#666",
            "weight": 0.1,
            "fillOpacity": 0.65,
        }

    def make_props(row):
        zhvi = f"${row.zhvi:,.0f}" if pd.notna(row.zhvi) else "—"
        pred = f"{row.pred*100:+.2f}%"
        metro = row.metro_name if pd.notna(row.metro_name) else "—"
        return {
            "zip": row.zip,
            "metro": metro,
            "zhvi": zhvi,
            "pred": pred,
            "decile": int(row.decile) if pd.notna(row.decile) else None,
        }

    # Convert to a GeoJSON-ish structure that Folium can ingest.
    # Use folium.GeoJson with a tooltip; keep properties light.
    merged_for_geo = merged.copy()
    merged_for_geo["pred_pct"] = (merged_for_geo["pred"] * 100).round(2)
    merged_for_geo["zhvi_fmt"] = merged_for_geo["zhvi"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )
    folium.GeoJson(
        merged_for_geo[["zip", "metro_name", "zhvi_fmt", "pred_pct", "decile", "geometry"]],
        name="ZIP decile",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["zip", "metro_name", "zhvi_fmt", "pred_pct", "decile"],
            aliases=["ZIP:", "Metro:", "ZHVI:", "Pred fwd_3y ann.:", "Decile (0=bot, 9=top):"],
            localize=True,
            sticky=False,
            labels=True,
        ),
    ).add_to(m)
    return m


def main() -> None:
    t0 = time.time()
    print("Loading post-2012 fwd_3y LightGBM booster...")
    booster, features = load_model()
    print(f"  {len(features)} features expected by model")

    print("Scoring zips...")
    scored = score_most_recent_month(booster, features)
    print(f"  scored {len(scored)} zips; decile bins built")

    zip_set = set(scored["zip"].astype(str))
    print("Loading + simplifying ZCTA shapes...")
    shapes = load_simplified_shapes(zip_set)

    print("Building Folium map...")
    m = build_map(scored, shapes)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_HTML))
    sz_mb = OUT_HTML.stat().st_size / (1024 * 1024)
    print(f"Saved {OUT_HTML}  ({sz_mb:.1f} MB)")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
