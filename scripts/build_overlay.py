"""Phase 4 personal overlay shortlists — Owner-Occupier and Investor.

Builds two parallel ZIP-level shortlists by scoring the latest snapshot with the
trained Phase 2 LightGBM models and then layering personal-preference filters
on top. The two profiles weight DIFFERENT objectives:

  - Owner-Occupier: predict PRICE return (fwd_3y), then filter for low climate
    risk, low recent disaster frequency, low pupil-teacher ratio, low premature
    death rate (life-expectancy proxy), and (optionally) majority owner-occupied.
  - Investor: predict TOTAL return (fwd_3y_total = price + rent carry), then
    filter for >=6% gross rent yield, ZHVI in a capital-efficient range
    ($80K-$400K), and an existing rental market (acs__owner_occupied_pct <= 0.7).

Both shortlists are nationwide (top-100-metro restricted, same as the
leaderboards) and price-unrestricted on the owner-occupier side.

Models are NOT refit — we just score the latest snapshot. Median thresholds
are computed over the same top-100-metro zip universe (after dropping rows
missing the relevant feature).

Filter features like school student/teacher ratio, premature death, and ACS
owner-occupied share are publication-lagged and stale at the very latest month
in the feature store. For these, we fall back to each zip's most recent
non-null value across the panel. This is appropriate because they are slow-
moving structural features (the school ratio in a given ZIP doesn't shift
month-to-month).

NIBRS county-year crime: not present in the feature store yet. The crime
filter is documented as skipped in the HTML fragment.

Outputs:
  - data/processed/overlay_owner_occupier.parquet
  - data/processed/overlay_investor.parquet
  - data/processed/overlay.html  (fragment, two stacked tables)
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from html import escape

import lightgbm as lgb
import numpy as np
import pandas as pd

from arbok.config import PROCESSED, TOP_N_METROS
from arbok.sources.census_metros import load_top_metros
from arbok.sources.zip_geo import load_zip_geo

PRICE_ARTIFACTS = PROCESSED / "phase2_artifacts"
TOTAL_ARTIFACTS = PROCESSED / "phase2_artifacts_total"

PRICE_MODEL = PRICE_ARTIFACTS / "fwd_3y__post_2012__lgbm.txt"
PRICE_FEATS = PRICE_ARTIFACTS / "fwd_3y__post_2012__lgbm.features.txt"
PRICE_Q10 = PRICE_ARTIFACTS / "fwd_3y__post_2012__lgbm_q10.txt"
PRICE_Q10_FEATS = PRICE_ARTIFACTS / "fwd_3y__post_2012__lgbm_q10.features.txt"
PRICE_Q50 = PRICE_ARTIFACTS / "fwd_3y__post_2012__lgbm_q50.txt"
PRICE_Q50_FEATS = PRICE_ARTIFACTS / "fwd_3y__post_2012__lgbm_q50.features.txt"
PRICE_Q90 = PRICE_ARTIFACTS / "fwd_3y__post_2012__lgbm_q90.txt"
PRICE_Q90_FEATS = PRICE_ARTIFACTS / "fwd_3y__post_2012__lgbm_q90.features.txt"

TOTAL_MODEL = TOTAL_ARTIFACTS / "fwd_3y_total__post_2012__lgbm.txt"
TOTAL_FEATS = TOTAL_ARTIFACTS / "fwd_3y_total__post_2012__lgbm.features.txt"

OUT_OWNER = PROCESSED / "overlay_owner_occupier.parquet"
OUT_INVESTOR = PROCESSED / "overlay_investor.parquet"
OUT_HTML = PROCESSED / "overlay.html"

TOP_N = 30


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def _load_booster(model_path, feats_path):
    booster = lgb.Booster(model_file=str(model_path))
    features = feats_path.read_text().strip().split("\n")
    return booster, features


def _predict(booster, features: list[str], df: pd.DataFrame) -> np.ndarray:
    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    return booster.predict(X)


def _pick_scoring_month(fs: pd.DataFrame, features: list[str]) -> pd.Timestamp:
    """Latest month with >=1000 zips at >=80% feature coverage."""
    counts = (
        fs.groupby("year_month")
        .apply(lambda g: ((g[features].notna().sum(axis=1) >= len(features) * 0.8).sum()))
        .sort_index()
    )
    ok = counts[counts >= 1000]
    return ok.index.max() if not ok.empty else counts.idxmax()


def _latest_value_per_zip(fs: pd.DataFrame, col: str) -> pd.Series:
    """For each zip, get the most recent non-null value of `col`. Slow-moving
    structural features (school ratios, ACS demographics, NRI risk scores,
    health outcomes) are stale at the very latest month but are essentially
    constant over a 1–2 year window, so the last observed value is fine."""
    if col not in fs.columns:
        return pd.Series(dtype=float, name=col)
    sub = fs[["zip", "year_month", col]].dropna(subset=[col])
    if sub.empty:
        return pd.Series(dtype=float, name=col)
    sub = sub.sort_values("year_month").drop_duplicates(subset=["zip"], keep="last")
    return sub.set_index("zip")[col]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading feature store…")
    fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
    print(f"  fs shape: {fs.shape}")

    top_metros = load_top_metros().head(TOP_N_METROS)
    zip_geo = load_zip_geo()[["zip", "city", "state"]]

    # Metro display name = principal (largest-pop) city per CBSA.
    tm = top_metros.copy()
    tm["cbsa"] = tm["cbsa"].astype(str).str.zfill(5)
    tm["principal_city"] = tm["core_name"].str.split("-").str[0]
    tm["principal_state"] = tm["state"].str.split("-").str[0]
    tm["metro_name"] = tm["principal_city"] + ", " + tm["principal_state"]
    cbsa_to_metro = tm.set_index("cbsa")["metro_name"].to_dict()

    # Restrict feature store to top-100-metro zips (matches score_zips.py).
    fs = fs[fs["cbsa"].astype(str).isin(set(tm["cbsa"]))].copy()
    print(f"  after top-{TOP_N_METROS} metro filter: {fs.shape}")

    # ───── Load models ─────
    print("\nLoading models…")
    price_booster, price_feats = _load_booster(PRICE_MODEL, PRICE_FEATS)
    print(f"  price model: {len(price_feats)} features")
    total_booster, total_feats = _load_booster(TOTAL_MODEL, TOTAL_FEATS)
    print(f"  total-return model: {len(total_feats)} features")

    # Quantile boosters for the price model (used for p10/p90 on owner-occupier).
    price_q_models = {}
    price_q_feats = {}
    for col, mp, fp in [
        ("pred_p10", PRICE_Q10, PRICE_Q10_FEATS),
        ("pred_p50", PRICE_Q50, PRICE_Q50_FEATS),
        ("pred_p90", PRICE_Q90, PRICE_Q90_FEATS),
    ]:
        if mp.exists() and fp.exists():
            price_q_models[col] = lgb.Booster(model_file=str(mp))
            price_q_feats[col] = fp.read_text().strip().split("\n")
    print(f"  loaded {len(price_q_models)} price quantile boosters: {list(price_q_models)}")

    # ───── Pick scoring month ─────
    # We use the price-model's feature list to pick the scoring month — it has
    # broader coverage (no rent_yield dependence), so it picks the truly latest
    # month. Both models then score that snapshot. (rent_yield_t is sparser
    # but the latest-value-per-zip fallback below covers stale rows.)
    latest = _pick_scoring_month(fs, price_feats)
    print(f"\nScoring as of: {latest}")
    snap = fs[fs["year_month"] == latest].copy()
    print(f"  snapshot rows: {len(snap):,}")

    # ───── Score both models on snapshot ─────
    print("Scoring price model…")
    snap["pred_fwd_3y"] = _predict(price_booster, price_feats, snap)

    if price_q_models:
        ordered = ["pred_p10", "pred_p50", "pred_p90"]
        ordered = [c for c in ordered if c in price_q_models]
        arr_cols = []
        for c in ordered:
            arr_cols.append(_predict(price_q_models[c], price_q_feats[c], snap))
        arr = np.column_stack(arr_cols)
        # Enforce monotone quantiles (cumulative max).
        arr = np.maximum.accumulate(arr, axis=1)
        for i, c in enumerate(ordered):
            snap[c] = arr[:, i]
        # Match score_zips.py convention: point pred = p50 if available.
        if "pred_p50" in snap.columns:
            snap["pred_fwd_3y"] = snap["pred_p50"].values

    print("Scoring total-return model…")
    snap["pred_fwd_3y_total"] = _predict(total_booster, total_feats, snap)
    # No quantile boosters were trained for the total-return target (runtime
    # budget — see scripts/run_phase2_total.py). We expose only the point pred.

    # ───── Filter features: forward-fill from latest non-null per zip ─────
    # These are slow-moving structural features and the latest panel month
    # often hasn't been refreshed yet (publication lag).
    print("\nForward-filling stale filter features per zip…")
    filter_feats = [
        "nri__wildfire_risk",
        "nri__flood_risk_combined",
        "fema__disasters_10yr_count",
        "school__student_teacher_ratio",
        "health__premature_death_yppl",
        "acs__owner_occupied_pct",
        "rent_yield_t",
    ]
    for col in filter_feats:
        ff = _latest_value_per_zip(fs, col)
        # Attach as `<col>_ff` so we don't clobber the snapshot's same-month value
        # (which may be NaN). Then prefer the ff column when populated.
        snap = snap.merge(
            ff.rename(col + "_ff"), left_on="zip", right_index=True, how="left",
        )
        snap[col + "_use"] = snap[col].where(snap[col].notna(), snap[col + "_ff"])
        cov = snap[col + "_use"].notna().mean() * 100
        print(f"  {col}: {cov:.1f}% covered (after ffill)")

    # acs__owner_occupied_pct is on a 0-100 scale in the feature store; the
    # spec uses 0-1 thresholds (>= 0.5 for owner-occupier, [0.4, 0.7] for
    # investor). Normalize to 0-1 here so the rest of the script reads
    # naturally and matches the spec.
    if "acs__owner_occupied_pct_use" in snap.columns and snap["acs__owner_occupied_pct_use"].max() > 1.5:
        snap["acs__owner_occupied_pct_use"] = snap["acs__owner_occupied_pct_use"] / 100.0
        print("  (normalized acs__owner_occupied_pct from 0-100 to 0-1)")

    # ───── Compute medians across top-100-metro zips for thresholds ─────
    # The medians are taken over the SAME snapshot universe (top-100 metros,
    # latest scoring month, with ffilled features), dropping zips missing the
    # feature. We never include the cross-metro median over the full panel —
    # that would let pre-2008 zip-months distort the threshold.
    def _median(col_use: str) -> float | None:
        s = snap[col_use].dropna()
        if s.empty:
            return None
        return float(s.median())

    medians = {
        "wildfire": _median("nri__wildfire_risk_use"),
        "flood": _median("nri__flood_risk_combined_use"),
        "school": _median("school__student_teacher_ratio_use"),
        # Premature death: LOWER is better (proxy for higher life expectancy).
        # Spec asks "life_expectancy >= median" → use "premature_death_yppl <= median".
        "premature_death": _median("health__premature_death_yppl_use"),
    }
    print(f"\nMedians (across top-{TOP_N_METROS}-metro snapshot zips):")
    for k, v in medians.items():
        print(f"  {k}: {v:.3f}" if v is not None else f"  {k}: (no coverage — filter skipped)")

    # ═════════════════════════════════════════════════════════════════════════
    # Owner-occupier shortlist
    # ═════════════════════════════════════════════════════════════════════════
    print("\n═══ Owner-Occupier shortlist ═══")
    oo = snap.copy()
    skipped_filters_oo: list[str] = []
    # Positive predicted return.
    oo = oo[oo["pred_fwd_3y"] > 0]
    print(f"  after pred_fwd_3y > 0: {len(oo):,}")

    # Climate risk filters.
    if medians["wildfire"] is not None:
        oo = oo[oo["nri__wildfire_risk_use"] <= medians["wildfire"]]
        print(f"  after wildfire <= median ({medians['wildfire']:.2f}): {len(oo):,}")
    else:
        skipped_filters_oo.append("nri__wildfire_risk (no coverage)")
    if medians["flood"] is not None:
        oo = oo[oo["nri__flood_risk_combined_use"] <= medians["flood"]]
        print(f"  after flood <= median ({medians['flood']:.2f}): {len(oo):,}")
    else:
        skipped_filters_oo.append("nri__flood_risk_combined (no coverage)")

    # Disaster frequency: hard cap at 5.
    if "fema__disasters_10yr_count_use" in oo.columns and oo["fema__disasters_10yr_count_use"].notna().any():
        oo = oo[oo["fema__disasters_10yr_count_use"] <= 5]
        print(f"  after fema disasters_10yr <= 5: {len(oo):,}")
    else:
        skipped_filters_oo.append("fema__disasters_10yr_count (no coverage)")

    # School quality (lower student-teacher = better).
    if medians["school"] is not None:
        oo = oo[oo["school__student_teacher_ratio_use"] <= medians["school"]]
        print(f"  after student/teacher <= median ({medians['school']:.2f}): {len(oo):,}")
    else:
        skipped_filters_oo.append("school__student_teacher_ratio (no coverage)")

    # Crime: NIBRS not yet in feature store. Document and skip.
    skipped_filters_oo.append(
        "crime__*_per_100k (NIBRS not in feature store — needs FBI CDE_API_KEY)"
    )

    # Health: lower premature_death = higher life expectancy.
    if medians["premature_death"] is not None:
        oo = oo[oo["health__premature_death_yppl_use"] <= medians["premature_death"]]
        print(f"  after premature death <= median ({medians['premature_death']:.0f}): {len(oo):,}")
    else:
        skipped_filters_oo.append("health__premature_death_yppl (no coverage)")

    # Optional ownership filter.
    if oo["acs__owner_occupied_pct_use"].notna().any():
        oo = oo[oo["acs__owner_occupied_pct_use"] >= 0.5]
        print(f"  after acs owner_occupied >= 0.5: {len(oo):,}")
    else:
        skipped_filters_oo.append("acs__owner_occupied_pct (no coverage, optional filter skipped)")

    # Rank by predicted price return, take top 30.
    oo_top = (
        oo.sort_values("pred_fwd_3y", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )

    # Attach metro + city/state names.
    oo_top["cbsa"] = oo_top["cbsa"].astype(str).str.zfill(5)
    oo_top["metro_name"] = oo_top["cbsa"].map(cbsa_to_metro).fillna("—")
    oo_top = oo_top.merge(zip_geo, on="zip", how="left")

    oo_cols = [
        "zip", "city", "state", "cbsa", "metro_name", "zhvi",
        "pred_fwd_3y", "pred_p10", "pred_p90",
        "nri__wildfire_risk_use", "nri__flood_risk_combined_use",
        "fema__disasters_10yr_count_use", "school__student_teacher_ratio_use",
        "health__premature_death_yppl_use", "acs__owner_occupied_pct_use",
    ]
    # Some quantile cols may not exist if quantile boosters weren't found.
    oo_cols = [c for c in oo_cols if c in oo_top.columns]
    oo_out = oo_top[oo_cols].rename(columns={
        "nri__wildfire_risk_use": "nri_wildfire_risk",
        "nri__flood_risk_combined_use": "nri_flood_risk",
        "fema__disasters_10yr_count_use": "fema_disasters_10yr",
        "school__student_teacher_ratio_use": "student_teacher_ratio",
        "health__premature_death_yppl_use": "premature_death_yppl",
        "acs__owner_occupied_pct_use": "acs_owner_occupied_pct",
    })
    oo_out.to_parquet(OUT_OWNER, index=False)
    print(f"  saved → {OUT_OWNER} ({len(oo_out)} rows)")

    # ═════════════════════════════════════════════════════════════════════════
    # Investor shortlist
    # ═════════════════════════════════════════════════════════════════════════
    print("\n═══ Investor shortlist ═══")
    inv = snap.copy()
    skipped_filters_inv: list[str] = []

    # Top quartile by predicted TOTAL return across the snapshot.
    q75_total = float(inv["pred_fwd_3y_total"].quantile(0.75))
    inv = inv[inv["pred_fwd_3y_total"] >= q75_total]
    print(f"  after pred_fwd_3y_total >= Q75 ({q75_total:+.3f}): {len(inv):,}")

    # Rent yield >= 6%.
    if inv["rent_yield_t_use"].notna().any():
        inv = inv[inv["rent_yield_t_use"] >= 0.06]
        print(f"  after rent_yield_t >= 0.06: {len(inv):,}")
    else:
        skipped_filters_inv.append("rent_yield_t (no coverage)")

    # ZHVI window: $80K–$400K (capital-efficient range for cash-flow investors).
    inv = inv[(inv["zhvi"] >= 80_000) & (inv["zhvi"] <= 400_000)]
    print(f"  after zhvi in [$80K, $400K]: {len(inv):,}")

    # Owner-occupancy share: 0.4 ≤ x ≤ 0.7.
    if inv["acs__owner_occupied_pct_use"].notna().any():
        inv = inv[
            (inv["acs__owner_occupied_pct_use"] <= 0.7)
            & (inv["acs__owner_occupied_pct_use"] >= 0.4)
        ]
        print(f"  after acs owner_occ in [0.4, 0.7]: {len(inv):,}")
    else:
        skipped_filters_inv.append("acs__owner_occupied_pct (no coverage)")

    # Rank by predicted total return, take top 30.
    inv_top = (
        inv.sort_values("pred_fwd_3y_total", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    inv_top["cbsa"] = inv_top["cbsa"].astype(str).str.zfill(5)
    inv_top["metro_name"] = inv_top["cbsa"].map(cbsa_to_metro).fillna("—")
    inv_top = inv_top.merge(zip_geo, on="zip", how="left")

    inv_cols = [
        "zip", "city", "state", "cbsa", "metro_name", "zhvi", "zori",
        "rent_yield_t_use", "pred_fwd_3y_total", "pred_fwd_3y",
        "acs__owner_occupied_pct_use",
    ]
    inv_cols = [c for c in inv_cols if c in inv_top.columns]
    inv_out = inv_top[inv_cols].rename(columns={
        "rent_yield_t_use": "rent_yield_t",
        "acs__owner_occupied_pct_use": "acs_owner_occupied_pct",
    })
    inv_out.to_parquet(OUT_INVESTOR, index=False)
    print(f"  saved → {OUT_INVESTOR} ({len(inv_out)} rows)")

    # ═════════════════════════════════════════════════════════════════════════
    # Reporting + HTML fragment
    # ═════════════════════════════════════════════════════════════════════════
    overlap_zips = set(oo_out["zip"]).intersection(set(inv_out["zip"]))
    print(f"\nOverlap between owner-occupier and investor shortlists: {len(overlap_zips)} zip(s)")
    if overlap_zips:
        print(f"  {sorted(overlap_zips)}")

    # Build the HTML fragment.
    html = _render_html(
        oo_out=oo_out,
        inv_out=inv_out,
        cbsa_to_metro=cbsa_to_metro,
        latest=latest,
        medians=medians,
        skipped_filters_oo=skipped_filters_oo,
        skipped_filters_inv=skipped_filters_inv,
        overlap_zips=overlap_zips,
    )
    OUT_HTML.write_text(html)
    print(f"\nSaved HTML fragment → {OUT_HTML} ({OUT_HTML.stat().st_size/1024:.1f} KB)")

    # ───── Final terminal report ─────
    print("\n" + "=" * 70)
    print("PHASE 4 OVERLAY SUMMARY")
    print("=" * 70)
    print(f"Scored as of: {latest}")
    print(f"Owner-occupier shortlist: {len(oo_out)} zips")
    if len(oo_out) < TOP_N:
        print(f"  (sub-{TOP_N}: filters were strict — {TOP_N - len(oo_out)} fewer rows)")
    if not oo_out.empty:
        dom_metros_oo = oo_out["metro_name"].value_counts().head(3)
        print(f"  dominant metros: {dict(dom_metros_oo)}")
        print(f"  median ZHVI: ${oo_out['zhvi'].median():,.0f}")
        print(f"  median pred fwd_3y: {oo_out['pred_fwd_3y'].median()*100:+.2f}%")

    print(f"\nInvestor shortlist: {len(inv_out)} zips")
    if len(inv_out) < TOP_N:
        print(f"  (sub-{TOP_N}: filters were strict — {TOP_N - len(inv_out)} fewer rows)")
    if not inv_out.empty:
        dom_metros_inv = inv_out["metro_name"].value_counts().head(3)
        print(f"  dominant metros: {dict(dom_metros_inv)}")
        print(f"  median ZHVI: ${inv_out['zhvi'].median():,.0f}")
        print(f"  median rent yield: {inv_out['rent_yield_t'].median()*100:.2f}%")
        print(f"  median pred fwd_3y_total: {inv_out['pred_fwd_3y_total'].median()*100:+.2f}%")

    print(f"\nOverlap: {len(overlap_zips)} zip(s) in both shortlists")


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:+.2f}%"


def _fmt_money(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.0f}"


def _fmt_num(x: float | None, dp: int = 1) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.{dp}f}"


def _owner_table_html(df: pd.DataFrame) -> str:
    """Owner-occupier display table — dedupe by CBSA so it isn't all-one-metro."""
    if df.empty:
        return "<p><i>(No zips passed the owner-occupier filters.)</i></p>"
    disp = df.drop_duplicates(subset=["cbsa"], keep="first").reset_index(drop=True)
    has_p10 = "pred_p10" in disp.columns
    has_p90 = "pred_p90" in disp.columns
    rows = []
    for _, r in disp.iterrows():
        loc = f"{r.get('city', '')}, {r.get('state', '')}".strip(", ")
        cells = [
            f"<td>{escape(str(r['zip']))}</td>",
            f"<td>{escape(loc)}</td>",
            f"<td>{escape(str(r.get('metro_name', '—')))}</td>",
            f"<td>{_fmt_money(r.get('zhvi'))}</td>",
            f"<td>{_fmt_pct(r.get('pred_fwd_3y'))}</td>",
        ]
        if has_p10:
            cells.append(f"<td>{_fmt_pct(r.get('pred_p10'))}</td>")
        if has_p90:
            cells.append(f"<td>{_fmt_pct(r.get('pred_p90'))}</td>")
        cells.extend([
            f"<td>{_fmt_num(r.get('nri_wildfire_risk'))}</td>",
            f"<td>{_fmt_num(r.get('nri_flood_risk'))}</td>",
            f"<td>{int(r['fema_disasters_10yr']) if pd.notna(r.get('fema_disasters_10yr')) else '—'}</td>",
            f"<td>{_fmt_num(r.get('student_teacher_ratio'))}</td>",
            f"<td>{int(r['premature_death_yppl']) if pd.notna(r.get('premature_death_yppl')) else '—'}</td>",
            f"<td>{_fmt_num(r.get('acs_owner_occupied_pct'), dp=2)}</td>",
        ])
        rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = ["ZIP", "City, State", "Metro", "ZHVI", "Pred fwd_3y"]
    if has_p10:
        headers.append("p10")
    if has_p90:
        headers.append("p90")
    headers.extend([
        "NRI wildfire", "NRI flood", "FEMA 10y",
        "Stud/Tchr", "Prem. death (yppl)", "ACS own%",
    ])
    head = "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>"
    return f"<table class='leaderboard'>{head}{''.join(rows)}</table>"


def _investor_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><i>(No zips passed the investor filters.)</i></p>"
    disp = df.drop_duplicates(subset=["cbsa"], keep="first").reset_index(drop=True)
    rows = []
    for _, r in disp.iterrows():
        loc = f"{r.get('city', '')}, {r.get('state', '')}".strip(", ")
        rows.append(
            "<tr>"
            f"<td>{escape(str(r['zip']))}</td>"
            f"<td>{escape(loc)}</td>"
            f"<td>{escape(str(r.get('metro_name', '—')))}</td>"
            f"<td>{_fmt_money(r.get('zhvi'))}</td>"
            f"<td>{_fmt_money(r.get('zori'))}</td>"
            f"<td>{_fmt_pct(r.get('rent_yield_t'))}</td>"
            f"<td>{_fmt_pct(r.get('pred_fwd_3y_total'))}</td>"
            f"<td>{_fmt_pct(r.get('pred_fwd_3y'))}</td>"
            f"<td>{_fmt_num(r.get('acs_owner_occupied_pct'), dp=2)}</td>"
            "</tr>"
        )
    headers = [
        "ZIP", "City, State", "Metro", "ZHVI", "ZORI",
        "Rent yield", "Pred fwd_3y total", "Pred fwd_3y (price)", "ACS own%",
    ]
    head = "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>"
    return f"<table class='leaderboard'>{head}{''.join(rows)}</table>"


def _render_html(
    *,
    oo_out: pd.DataFrame,
    inv_out: pd.DataFrame,
    cbsa_to_metro: dict,
    latest: pd.Timestamp,
    medians: dict,
    skipped_filters_oo: list[str],
    skipped_filters_inv: list[str],
    overlap_zips: set,
) -> str:
    """Self-contained HTML fragment (no <html>/<head>) so it can be embedded
    in the main report. Uses the same `.leaderboard` table class as
    generate_report.py — the parent report's CSS will style it consistently.
    """

    def _summary(df: pd.DataFrame, kind: str) -> str:
        if df.empty:
            return f"<p><i>No zips passed {kind} filters.</i></p>"
        top_metros = df["metro_name"].value_counts().head(3)
        if kind == "owner-occupier":
            med_pred = df["pred_fwd_3y"].median() * 100
            return (
                f"<p><b>{len(df)} zips</b>; dominant metros: "
                + ", ".join(f"{m} ({n})" for m, n in top_metros.items())
                + f". Median ZHVI <b>${df['zhvi'].median():,.0f}</b>; "
                + f"median predicted fwd_3y price return <b>{med_pred:+.2f}%</b>.</p>"
            )
        else:
            med_yield = df["rent_yield_t"].median() * 100
            med_total = df["pred_fwd_3y_total"].median() * 100
            return (
                f"<p><b>{len(df)} zips</b>; dominant metros: "
                + ", ".join(f"{m} ({n})" for m, n in top_metros.items())
                + f". Median ZHVI <b>${df['zhvi'].median():,.0f}</b>; "
                + f"median rent yield <b>{med_yield:.2f}%</b>; "
                + f"median predicted fwd_3y total return <b>{med_total:+.2f}%</b>.</p>"
            )

    overlap_note = ""
    if overlap_zips:
        overlap_note = (
            f"<p><b>{len(overlap_zips)} ZIP(s) appear in both shortlists:</b> "
            + ", ".join(f"<code>{escape(z)}</code>" for z in sorted(overlap_zips))
            + ". These zips combine low climate / school risk with a positive "
            "rental yield carry — both profiles agree.</p>"
        )
    else:
        overlap_note = (
            "<p><b>The two shortlists do not overlap on any ZIP.</b> "
            "The owner-occupier profile up-weights low-risk, school-strong, low-disaster "
            "areas while the investor profile favors lower-priced, higher-yield areas; "
            "they end up pointing at different zips by construction.</p>"
        )

    med_lines = []
    for k, v in medians.items():
        if v is None:
            med_lines.append(f"<li><code>{k}</code>: <i>no coverage — filter skipped</i></li>")
        else:
            med_lines.append(f"<li><code>{k}</code>: median = <b>{v:.3f}</b></li>")

    skipped_oo_html = ""
    if skipped_filters_oo:
        skipped_oo_html = (
            "<p class='caveat'><b>Owner-occupier filters skipped:</b><ul>"
            + "".join(f"<li>{escape(s)}</li>" for s in skipped_filters_oo)
            + "</ul></p>"
        )
    skipped_inv_html = ""
    if skipped_filters_inv:
        skipped_inv_html = (
            "<p class='caveat'><b>Investor filters skipped:</b><ul>"
            + "".join(f"<li>{escape(s)}</li>" for s in skipped_filters_inv)
            + "</ul></p>"
        )

    return f"""
  <h2>6 · Personal-overlay shortlists (Phase 4)</h2>
  <p>Two parallel ZIP-level shortlists, both nationwide and both built on the
     latest snapshot (<b>{latest.strftime('%Y-%m')}</b>), one tuned for an
     owner-occupier and one for a cash-flow investor. Filter thresholds use the
     <b>median across top-{TOP_N_METROS}-metro zips</b> on the snapshot
     (after forward-filling slow-moving structural features like school
     student-teacher ratio and ACS demographics from the most recent non-null
     panel month — these features lag the price target by 1–2 years and don't
     change month-to-month). The underlying parquets keep all {TOP_N} rows; the
     display tables below dedupe by metro (CBSA) so you don't see the same city
     ten times.</p>

  <h3>6a · Owner-occupier (price-return focus)</h3>
  <p>Scored on the price-only fwd_3y model. Filters keep zips with positive
     expected price return AND low climate risk (wildfire + flood ≤ median),
     low recent disaster frequency (≤ 5 FEMA declarations in 10 years), strong
     schools (student-teacher ratio ≤ median), good health outcomes
     (premature death rate ≤ median, as a life-expectancy proxy), and majority
     owner-occupied (acs__owner_occupied_pct ≥ 0.5).</p>
  {_summary(oo_out, "owner-occupier")}
  {_owner_table_html(oo_out)}
  {skipped_oo_html}

  <h3>6b · Investor (total-return + rent-yield focus)</h3>
  <p>Scored on the total-return fwd_3y model (price appreciation + ZORI rent
     carry, trained in the Phase 2 total-return run). Filters keep zips in the
     top quartile of predicted total return AND with a gross rent yield ≥ 6%,
     a ZHVI in the cash-flow-friendly band ($80K–$400K), and ACS
     owner-occupancy share in [0.4, 0.7] so there's an active rental market
     without signalling deep distress.</p>
  {_summary(inv_out, "investor")}
  {_investor_table_html(inv_out)}
  {skipped_inv_html}

  <h3>6c · Overlap and threshold transparency</h3>
  {overlap_note}
  <p><b>Median thresholds used (over top-{TOP_N_METROS}-metro snapshot zips):</b></p>
  <ul>
    {''.join(med_lines)}
  </ul>
  <div class='caveat'>
    <b>Caveats specific to this overlay.</b>
    <ul style='margin-top:.4em;margin-bottom:0;'>
      <li><b>NIBRS crime not yet in the feature store.</b> The owner-occupier
          spec called for a crime filter but the FBI NIBRS county-year crime
          aggregate is gated on <code>CDE_API_KEY</code> and not yet loaded.
          The filter is skipped, documented above.</li>
      <li><b>Total-return model has no quantile boosters.</b> The investor
          table shows only point predictions for total return — the Phase 2
          total-return run was point-only by design (runtime budget).</li>
      <li><b>Slow-moving features are forward-filled.</b> School ratios, ACS
          demographics, NRI scores, FEMA disaster counts, and premature
          mortality use each ZIP's most-recent non-null value across the
          panel (not the scoring month, which is often unrefreshed). For
          stable structural features this is a fair approximation; we're not
          claiming a real-time read on these.</li>
      <li><b>This is not investment advice.</b> The model is unaware of
          zoning, specific listings, walkability, your personal commute or
          family constraints — it's a quantitative sort, not a buy
          recommendation.</li>
    </ul>
  </div>
"""


if __name__ == "__main__":
    main()
