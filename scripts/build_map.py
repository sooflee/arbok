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
import matplotlib
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from branca.element import Element

from arbok.config import PROCESSED, RAW

POST_2012_LGBM = PROCESSED / "phase2_artifacts" / "fwd_3y__post_2012__lgbm.txt"
POST_2012_FEATS = PROCESSED / "phase2_artifacts" / "fwd_3y__post_2012__lgbm.features.txt"
TIGER_SHP = RAW / "census_zcta" / "tl_2020_us_zcta520.shp"
ZIP_LEADERBOARD = PROCESSED / "zip_leaderboard.parquet"
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


def _cividis_hex(n: int = 10) -> list[str]:
    """Return n discrete hex colors from matplotlib's cividis colormap.

    Cividis is perceptually uniform AND specifically designed to be readable by
    viewers with common forms of color-vision deficiency (CVD). Unlike a
    red->green diverging palette, the gradient is monotonic in luminance so
    the ordering is unambiguous even in grayscale or for color-blind users.
    """
    cmap = matplotlib.colormaps["cividis"]
    return [matplotlib.colors.rgb2hex(cmap(i / (n - 1))) for i in range(n)]


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

    # Optional: join 80% prediction interval (p10/p90) from the top-50
    # zip_leaderboard.parquet. Only those zips will have the band; the rest
    # show a dash.
    if ZIP_LEADERBOARD.exists():
        lb = pd.read_parquet(ZIP_LEADERBOARD)
        lb["zip"] = lb["zip"].astype(str).str.zfill(5)
        keep_cols = ["zip"] + [c for c in ("pred_p10", "pred_p90") if c in lb.columns]
        if len(keep_cols) >= 3:
            merged = merged.merge(lb[keep_cols].drop_duplicates("zip"), on="zip", how="left")
        else:
            merged["pred_p10"] = np.nan
            merged["pred_p90"] = np.nan
    else:
        merged["pred_p10"] = np.nan
        merged["pred_p90"] = np.nan

    # Start the map; we'll snap the initial view to the data extent below
    # (better for mobile than a fixed CONUS zoom).
    m = folium.Map(location=[39.5, -98.5], zoom_start=5, tiles="CartoDB positron")

    # Colorblind-safe perceptually uniform palette (cividis, 10 discrete bins).
    # Replaces the prior red->green diverging palette which is unreadable for
    # ~8% of men with red-green color-vision deficiency.
    cmap = LinearColormap(
        colors=_cividis_hex(10),
        vmin=0, vmax=9,
        caption="Predicted fwd_3y return decile rank (0 = bottom 10% → 9 = top 10%)",
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
            "fillOpacity": 0.75,
        }

    # Build the per-feature props the tooltip will use.
    merged_for_geo = merged.copy()
    merged_for_geo["pred_pct"] = (merged_for_geo["pred"] * 100).round(2)
    merged_for_geo["zhvi_fmt"] = merged_for_geo["zhvi"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )

    # Human-readable decile label, e.g. "9 (top 10%)" or "0 (bottom 10%)".
    def _decile_label(d) -> str:
        if pd.isna(d):
            return "—"
        di = int(d)
        if di == 9:
            return f"{di} of 9 (top 10%)"
        if di == 0:
            return f"{di} of 9 (bottom 10%)"
        return f"{di} of 9"

    merged_for_geo["decile_label"] = merged_for_geo["decile"].apply(_decile_label)

    # 80% prediction band; only populated for top-50 leaderboard zips.
    def _band(row) -> str:
        p10, p90 = row.get("pred_p10"), row.get("pred_p90")
        if pd.isna(p10) or pd.isna(p90):
            return "—"
        return f"{p10*100:+.2f}% to {p90*100:+.2f}%"

    merged_for_geo["band_fmt"] = merged_for_geo.apply(_band, axis=1)

    folium.GeoJson(
        merged_for_geo[[
            "zip", "metro_name", "zhvi_fmt", "pred_pct",
            "decile", "decile_label", "band_fmt", "geometry",
        ]],
        name="ZIP decile",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["zip", "metro_name", "zhvi_fmt", "pred_pct", "decile_label", "band_fmt"],
            aliases=[
                "ZIP:",
                "Metro:",
                "ZHVI:",
                "Pred fwd_3y ann.:",
                "Decile rank (0=worst → 9=best):",
                "80% pred. band (top-50 only):",
            ],
            localize=True,
            sticky=False,
            labels=True,
        ),
    ).add_to(m)

    # Frame the initial view to the actual data extent so mobile users
    # don't open to a tiny CONUS view.
    try:
        minx, miny, maxx, maxy = merged.total_bounds
        if np.isfinite([minx, miny, maxx, maxy]).all():
            m.fit_bounds([[miny, minx], [maxy, maxx]])
    except Exception as e:  # pragma: no cover - best effort
        print(f"  [warn] fit_bounds failed: {e}")

    # Nudge the colormap legend to the bottom of the viewport. branca's
    # LinearColormap renders absolutely-positioned SVG at the top of the page
    # by default; on small screens that overlaps page content. Force it to
    # bottom-right via CSS.
    legend_css = """
    <style>
      svg.legend { top: auto !important; bottom: 14px !important;
                   right: 14px !important; left: auto !important;
                   max-width: 92vw !important; background: rgba(255,255,255,0.85);
                   padding: 4px 6px; border-radius: 4px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
      @media (max-width: 600px) {
        svg.legend { transform: scale(0.85); transform-origin: bottom right; }
      }
    </style>
    """
    m.get_root().header.add_child(Element(legend_css))

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
