"""Render two self-contained HTML pages from Phase 2 results + leaderboard.

Outputs:
  * data/processed/index.html   — landing page (decision aid: where to buy, overlays,
                                  macro context, alt ranking, headline results).
  * data/processed/methods.html — methodology + deep dive (CV robustness, backtest,
                                  decile spread chart, SHAP, price stratification).

Both pages share the same <head> (favicon, OG, Twitter meta, inlined Plotly), CSS
block, sticky TOC, accordions, sortable tables, and reading-time indicator. The
landing page lives at the root of the GitHub Pages deploy.
"""
from dotenv import load_dotenv
load_dotenv()

import datetime
from html import escape

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shap

from arbok.config import PROCESSED
from arbok.labels import label as feat_label

# Build date used to cache-bust the iframe + static asset URLs on GitHub Pages.
BUILD_DATE = datetime.date.today().isoformat()
# GitHub Pages site URL (used for og:url meta tag). Hard-coded from `git remote`.
SITE_URL = "https://sooflee.github.io/arbok/"

ARTIFACTS = PROCESSED / "phase2_artifacts"
RESULTS = pd.read_parquet(PROCESSED / "phase2_results.parquet")
LEADERBOARD = pd.read_parquet(PROCESSED / "zip_leaderboard.parquet")
CITY_LEADERBOARD = pd.read_parquet(PROCESSED / "city_leaderboard.parquet") if (PROCESSED / "city_leaderboard.parquet").exists() else None
WALKFORWARD = pd.read_parquet(PROCESSED / "phase2_walkforward.parquet") if (PROCESSED / "phase2_walkforward.parquet").exists() else None
SPATIALCV = pd.read_parquet(PROCESSED / "phase2_spatialcv.parquet") if (PROCESSED / "phase2_spatialcv.parquet").exists() else None
BACKTEST_PRICE = pd.read_parquet(PROCESSED / "backtest_topdecile.parquet") if (PROCESSED / "backtest_topdecile.parquet").exists() else None
BACKTEST_TOTAL = pd.read_parquet(PROCESSED / "backtest_topdecile_total.parquet") if (PROCESSED / "backtest_topdecile_total.parquet").exists() else None
ZIP_LB_TOTAL = pd.read_parquet(PROCESSED / "zip_leaderboard_total.parquet") if (PROCESSED / "zip_leaderboard_total.parquet").exists() else None
OVERLAY_HTML = (PROCESSED / "overlay.html").read_text() if (PROCESSED / "overlay.html").exists() else None
OVERLAY_OWNER_OCC = pd.read_parquet(PROCESSED / "overlay_owner_occupier.parquet") if (PROCESSED / "overlay_owner_occupier.parquet").exists() else None


def owner_occupier_concentration_stat() -> str:
    """Return text like '22 of 30 zips fall in the Washington, DC metro' computed live."""
    if OVERLAY_OWNER_OCC is None or OVERLAY_OWNER_OCC.empty:
        return "the overlay's filter chain"
    df = OVERLAY_OWNER_OCC.copy()
    name_col = "metro_name" if "metro_name" in df.columns else ("cbsa" if "cbsa" in df.columns else None)
    if name_col is None:
        return f"{len(df)} zips"
    counts = df[name_col].value_counts()
    top_metro = counts.index[0]
    top_n = int(counts.iloc[0])
    return f"{top_n} of {len(df)} zips fall in the {top_metro} metro"
FS = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
ZIP_GEO = pd.read_parquet(PROCESSED / "zip_geo.parquet") if (PROCESSED / "zip_geo.parquet").exists() else None
TOP_METROS = pd.read_csv(PROCESSED / "top_metros.csv") if (PROCESSED / "top_metros.csv").exists() else None
if TOP_METROS is not None:
    TOP_METROS["cbsa"] = TOP_METROS["cbsa"].astype(str).str.zfill(5)
    # Pick the largest-population city per CBSA as the metro's display name.
    # core_name like 'Pittsburgh' or 'New York-Newark-Jersey City'; first token is the
    # principal (largest-pop) city by Census convention.
    TOP_METROS["principal_city"] = TOP_METROS["core_name"].str.split("-").str[0]
    TOP_METROS["principal_state"] = TOP_METROS["state"].str.split("-").str[0]

HORIZONS = ["fwd_1y", "fwd_3y", "fwd_5y"]
HORIZON_CLASS = {"fwd_1y": "short (1y)", "fwd_3y": "medium (3y)", "fwd_5y": "long (5y)"}
SPLIT_LABEL = {"pre_2008": "pre-2008 (GFC stress)", "post_2012": "post-2012 (main)"}


def _div(fig: go.Figure, include_js: bool, aria_label: str | None = None) -> str:
    """Render a Plotly figure as an HTML <div>.

    Plotly is inlined exactly once per report (the first chart sets
    ``include_js=True``; everything else passes ``False`` and reuses the bundle).
    This makes the report self-contained and immune to CDN flakiness on
    GitHub Pages at the cost of ~3 MB.

    Wrapping the figure in a ``role='img'`` / ``aria-label`` div gives screen
    readers something to announce — Plotly's own SVG doesn't expose this.
    """
    html = pio.to_html(fig, full_html=False, include_plotlyjs=(True if include_js else False))
    if aria_label:
        return f"<div role='img' aria-label=\"{escape(aria_label)}\">{html}</div>"
    return html


def _wrap_table(table_html: str) -> str:
    """Wrap a table in a horizontal-scroll container so wide tables don't
    overflow narrow mobile viewports."""
    return f"<div style='overflow-x:auto'>{table_html}</div>"


def chart_decile_spread() -> go.Figure:
    df = RESULTS.copy()
    df["horizon_label"] = df["horizon"].map(HORIZON_CLASS)
    df["model_split"] = df["model"] + " (" + df["split"] + ")"
    fig = px.bar(
        df,
        x="horizon_label",
        y="decile_spread",
        color="model",
        barmode="group",
        facet_col="split",
        title="Out-of-sample decile spread by horizon × split × model",
        labels={"decile_spread": "decile spread (top - bottom realized return)", "horizon_label": "horizon"},
        category_orders={"horizon_label": list(HORIZON_CLASS.values()),
                         "model": ["overall_mean", "metro_mean", "hedonic_acs", "elasticnet", "lightgbm"]},
        height=420,
    )
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    return fig


def chart_shap_for(horizon: str, split: str = "post_2012", k: int = 15) -> go.Figure:
    booster = lgb.Booster(model_file=str(ARTIFACTS / f"{horizon}__{split}__lgbm.txt"))
    features = (ARTIFACTS / f"{horizon}__{split}__lgbm.features.txt").read_text().strip().split("\n")
    mask = FS[f"is_train_{split}"].fillna(False) & FS[horizon].notna()
    X = FS.loc[mask, features]
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    sample = X.sample(min(50_000, len(X)), random_state=0) if len(X) > 50_000 else X
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[0]
    mean_abs = np.abs(sv).mean(axis=0)
    top = (
        pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=True)
        .tail(k)
        .reset_index(drop=True)
    )
    # Map raw column names -> (display, description) for hover tooltips.
    top["display"] = top["feature"].apply(lambda f: feat_label(f)[0])
    top["description"] = top["feature"].apply(lambda f: feat_label(f)[1])

    fig = go.Figure(
        go.Bar(
            x=top["mean_abs_shap"],
            y=top["display"],
            orientation="h",
            marker_color="#3a5da9",
            customdata=top[["feature", "description"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "<i>%{customdata[0]}</i><br>"
                "mean |SHAP|: %{x:.4f}<br><br>"
                "%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Top {k} SHAP features — {horizon} @ {split}  ·  hover for description",
        xaxis_title="mean |SHAP| (avg per-zip prediction contribution)", yaxis_title="",
        height=460, margin=dict(l=240, r=20, t=60, b=40),
    )
    return fig


def _humanize_drivers(s: str) -> str:
    """Turn 'fred.real_rate_10y(-0.040), nri.wildfire_risk(+0.009)' into '10Y real rate (-0.040), NRI wildfire risk (+0.009)'.

    The wrapping ``<span class='tt'>`` element is both ``tabindex='0'`` (so
    keyboard users can focus it) and CSS-styled to reveal a sibling
    ``.tt-body`` on hover, focus, OR focus-within — meaning the description
    surfaces on tap (touch fires focus on the tabindex'd span) without needing
    a single line of JavaScript. The native ``title`` attr is kept too so the
    browser's own tooltip still works for mouse users.
    """
    if not isinstance(s, str):
        return ""
    out_parts = []
    for part in s.split(", "):
        # parse 'fred.real_rate_10y(-0.040)' -> ('fred.real_rate_10y', '-0.040')
        if "(" in part:
            name, num = part.rsplit("(", 1)
            num = "(" + num
        else:
            name, num = part, ""
        raw = name.replace(".", "__")
        display, desc = feat_label(raw)
        out_parts.append(
            f"<span class='tt' tabindex='0' title='{escape(desc)}'>"
            f"{escape(display)}"
            f"<span class='tt-body'>{escape(desc)}</span>"
            f"</span> <code>{escape(num)}</code>"
        )
    return ", ".join(out_parts)


def chart_city_leaderboard_top_n(n: int = 30) -> str:
    """One row per CBSA (metro), sorted by best-zip predicted return.

    The raw city_leaderboard.parquet rolled up by (city_name, state), which split
    a single metro into many ZIP-level hamlets (e.g. 10+ tiny PA places all in
    CBSA 38300 = Pittsburgh). Here we dedupe by CBSA, keep the highest-predicted
    zip's row, and relabel the display city as the largest-population city in
    that CBSA (the principal city from top_metros.csv).
    """
    if CITY_LEADERBOARD is None:
        return "<p><i>City leaderboard not built — run scripts/score_zips.py.</i></p>"
    df = CITY_LEADERBOARD.copy()
    df["cbsa"] = df["cbsa"].astype(str).str.zfill(5)

    # Dedupe by CBSA — keep the row with the highest predicted return per metro.
    df = (
        df.sort_values("predicted_fwd_3y_annualized", ascending=False)
          .drop_duplicates(subset=["cbsa"], keep="first")
          .reset_index(drop=True)
    )

    # Attach principal city/state from top_metros (population-largest in CBSA).
    if TOP_METROS is not None:
        df = df.merge(
            TOP_METROS[["cbsa", "principal_city", "principal_state", "population"]],
            on="cbsa", how="left",
        )
        df["display_city"] = df["principal_city"].fillna(df["city"])
        df["display_state"] = df["principal_state"].fillna(df["state"])
    else:
        df["display_city"] = df["city"]
        df["display_state"] = df["state"]

    df = df.head(n).copy()

    df["location"] = df["display_city"].astype(str) + ", " + df["display_state"].astype(str)
    df["best_zhvi"] = df["zhvi"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    df["pred_best"] = df["predicted_fwd_3y_annualized"].apply(lambda x: f"{x*100:+.2f}%")
    df["pred_top5"] = df["mean_pred_top5"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—")
    df["pred_p10_fmt"] = df["pred_p10"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—") if "pred_p10" in df.columns else "—"
    df["pred_p90_fmt"] = df["pred_p90"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—") if "pred_p90" in df.columns else "—"
    df["strong"] = df["strong_zip_count"].astype(str)
    df["drivers_html"] = df["drivers"].apply(_humanize_drivers)
    # Column order matches the zip leaderboard so the shared CSS rule
    # (.leaderboard td:nth-child(6) bold/blue) lights up the predicted-return column.
    df = df[["location", "cbsa", "best_zip", "best_zhvi", "pred_top5", "pred_best", "pred_p10_fmt", "pred_p90_fmt", "strong", "drivers_html"]]
    df.columns = [
        "Metro (principal city)", "CBSA", "Best ZIP", "Best ZIP price",
        "Mean pred (top 5 zips)", "Pred for best ZIP",
        "p10 (80% PI lo)", "p90 (80% PI hi)",
        "Zips in top 10%", "Top SHAP drivers for best ZIP (hover)",
    ]
    rows = []
    for row in df.values:
        cells = []
        for col_idx, v in enumerate(row):
            if col_idx == len(row) - 1:
                cells.append(f"<td>{v}</td>")
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in df.columns) + "</tr>"
    return _wrap_table(f"<table class='leaderboard'>{head}{''.join(rows)}</table>")


def chart_leaderboard_top_n(n: int = 20) -> str:
    """Return an HTML <table> string with top-n zips, including city/state lookup + human drivers."""
    df = LEADERBOARD.head(n).copy()
    # Attach city/state from the HUD zip-geo lookup
    if ZIP_GEO is not None:
        df = df.merge(ZIP_GEO[["zip", "city", "state"]], on="zip", how="left")
    else:
        df["city"] = ""
        df["state"] = ""
    df["location"] = df.apply(
        lambda r: f"{r['city']}, {r['state']}" if pd.notna(r.get("city")) and r["city"] else "—",
        axis=1,
    )
    df["zhvi_fmt"] = df["zhvi"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    df["zori_fmt"] = df["zori"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    df["predicted"] = df["predicted_fwd_3y_annualized"].apply(lambda x: f"{x*100:+.2f}%")
    df["pred_p10_fmt"] = df["pred_p10"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—") if "pred_p10" in df.columns else "—"
    df["pred_p90_fmt"] = df["pred_p90"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "—") if "pred_p90" in df.columns else "—"
    df["drivers_html"] = df["drivers"].apply(_humanize_drivers)
    df = df[["zip", "location", "cbsa", "zhvi_fmt", "zori_fmt", "predicted", "pred_p10_fmt", "pred_p90_fmt", "drivers_html"]]
    df.columns = [
        "ZIP", "City, State", "CBSA", "ZHVI", "ZORI", "Pred fwd_3y ann.",
        "p10 (80% PI lo)", "p90 (80% PI hi)",
        "Top SHAP drivers (hover for feature description)",
    ]
    rows = []
    for row in df.values:
        cells = []
        for col_idx, v in enumerate(row):
            # drivers column already has HTML; everything else escapes
            if col_idx == len(row) - 1:
                cells.append(f"<td>{v}</td>")
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in df.columns) + "</tr>"
    return _wrap_table(f"<table class='leaderboard'>{head}{''.join(rows)}</table>")


def price_stratified_decile_spread(
    horizon: str = "fwd_3y",
    split: str = "post_2012",
    tiers: tuple = (("<$200K", 0, 200_000), ("$200K–$500K", 200_000, 500_000), ("$500K+", 500_000, float("inf"))),
) -> pd.DataFrame:
    """For each ZHVI price tier, compute the post-2012 LightGBM decile-spread on
    the test window. Useful for telling apart 'the model is sorting on price
    mean-reversion' from 'the model has signal at every price point'."""
    booster = lgb.Booster(model_file=str(ARTIFACTS / f"{horizon}__{split}__lgbm.txt"))
    features = (ARTIFACTS / f"{horizon}__{split}__lgbm.features.txt").read_text().strip().split("\n")
    mask = FS[f"is_test_{split}"].fillna(False) & FS[horizon].notna() & FS["zhvi"].notna()
    X = FS.loc[mask, features].copy()
    X = X.fillna(X.median(numeric_only=True))
    realized = FS.loc[mask, horizon].to_numpy()
    zhvi = FS.loc[mask, "zhvi"].to_numpy()
    pred = booster.predict(X)

    out = []
    for label, lo, hi in tiers:
        m = (zhvi >= lo) & (zhvi < hi)
        if m.sum() < 1000:
            out.append({"tier": label, "n": int(m.sum()), "top_decile": np.nan,
                        "bot_decile": np.nan, "spread": np.nan, "median_zhvi": np.nan})
            continue
        p = pd.Series(pred[m])
        r = pd.Series(realized[m])
        deciles = pd.qcut(p, 10, labels=False, duplicates="drop")
        top = r[deciles == deciles.max()].mean()
        bot = r[deciles == 0].mean()
        out.append({
            "tier": label,
            "n": int(m.sum()),
            "median_zhvi": float(np.median(zhvi[m])),
            "top_decile": float(top),
            "bot_decile": float(bot),
            "spread": float(top - bot),
        })
    return pd.DataFrame(out)


def chart_price_stratification_table() -> str:
    """HTML table of decile spread per ZHVI tier for post-2012 LightGBM fwd_3y."""
    df = price_stratified_decile_spread()
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["spread"]):
            rows.append(
                f"<tr><td>{escape(r['tier'])}</td><td>{r['n']:,}</td>"
                f"<td colspan='4'><i>not enough test rows</i></td></tr>"
            )
            continue
        rows.append(
            f"<tr>"
            f"<td>{escape(r['tier'])}</td>"
            f"<td>{r['n']:,}</td>"
            f"<td>${r['median_zhvi']:,.0f}</td>"
            f"<td>{r['top_decile']*100:+.2f}%</td>"
            f"<td>{r['bot_decile']*100:+.2f}%</td>"
            f"<td>{r['spread']*100:+.2f}%</td>"
            f"</tr>"
        )
    return _wrap_table(f"""
    <table class='headline'>
      <tr><th>price tier (ZHVI)</th><th>test rows</th><th>median ZHVI</th>
          <th>top decile realized</th><th>bottom decile realized</th><th>decile spread</th></tr>
      {"".join(rows)}
    </table>
    """)


def chart_headline_table() -> str:
    """Per-horizon best-model summary, reported for BOTH temporal splits.

    Layout: two row-groups (one per split), separated by a styled subheader row,
    so the post-2012 main numbers and the pre-2008 GFC-stress numbers are visible
    side-by-side instead of only post-2012.
    """
    section_rows = []
    for split in ["post_2012", "pre_2008"]:
        best = (
            RESULTS[RESULTS["split"] == split]
            .sort_values(["horizon", "decile_spread"], ascending=[True, False])
            .groupby("horizon").head(1)
            .sort_values("horizon")
        )
        section_rows.append(
            f"<tr class='split-header'><td colspan='6'>{escape(SPLIT_LABEL.get(split, split))}</td></tr>"
        )
        for _, r in best.iterrows():
            section_rows.append(
                f"<tr>"
                f"<td>{HORIZON_CLASS.get(r['horizon'], r['horizon'])}</td>"
                f"<td>{escape(r['model'])}</td>"
                f"<td>{r['spearman']:+.3f}</td>"
                f"<td>{r['decile_spread']*100:+.2f}%</td>"
                f"<td>{r['top_decile_mean']*100:+.2f}%</td>"
                f"<td>{r['bot_decile_mean']*100:+.2f}%</td>"
                f"</tr>"
            )
    return _wrap_table(f"""
    <table class='headline'>
      <tr><th>horizon</th><th>best model</th><th>Spearman ρ</th><th>decile spread</th><th>top decile</th><th>bottom decile</th></tr>
      {"".join(section_rows)}
    </table>
    """)


def _summarize_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Group by horizon and compute mean/std/95% bands for Spearman + decile spread."""
    def q025(x):
        return float(np.quantile(x.dropna(), 0.025)) if x.notna().any() else float("nan")

    def q975(x):
        return float(np.quantile(x.dropna(), 0.975)) if x.notna().any() else float("nan")

    g = df.groupby("horizon").agg(
        n_folds=("fold", "count"),
        mean_spearman=("spearman_test", "mean"),
        std_spearman=("spearman_test", "std"),
        q025_spearman=("spearman_test", q025),
        q975_spearman=("spearman_test", q975),
        mean_spread=("decile_spread_test", "mean"),
        std_spread=("decile_spread_test", "std"),
    ).reset_index()
    # Force horizon ordering 1y -> 3y -> 5y
    g["__order"] = g["horizon"].map({"fwd_1y": 0, "fwd_3y": 1, "fwd_5y": 2}).fillna(99)
    g = g.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return g


def chart_walkforward_table() -> str:
    """HTML table summarizing walk-forward CV per horizon (LightGBM, post_2012, demean=False)."""
    if WALKFORWARD is None:
        return "<p><i>Walk-forward CV results not available.</i></p>"
    sub = WALKFORWARD[
        (WALKFORWARD["split"] == "post_2012")
        & (WALKFORWARD["model"] == "lightgbm")
        & (WALKFORWARD["demean"] == False)  # noqa: E712
    ]
    if sub.empty:
        return "<p><i>Walk-forward CV results not available.</i></p>"
    g = _summarize_cv(sub)
    rows = []
    for _, r in g.iterrows():
        rows.append(
            f"<tr>"
            f"<td>{HORIZON_CLASS.get(r['horizon'], r['horizon'])}</td>"
            f"<td>{int(r['n_folds'])}</td>"
            f"<td>{r['mean_spearman']:+.3f}</td>"
            f"<td>{r['std_spearman']:.3f}</td>"
            f"<td>[{r['q025_spearman']:+.3f}, {r['q975_spearman']:+.3f}]</td>"
            f"<td>{r['mean_spread']*100:+.2f}%</td>"
            f"<td>{r['std_spread']*100:.2f}%</td>"
            f"</tr>"
        )
    return _wrap_table(f"""
    <table class='headline'>
      <tr><th>horizon</th><th>n folds</th><th>mean Spearman ρ</th><th>std ρ</th>
          <th>95% interval ρ</th><th>mean decile spread</th><th>std decile spread</th></tr>
      {"".join(rows)}
    </table>
    """)


def chart_spatialcv_tables() -> tuple:
    """Two HTML tables: spatial CV summary for raw + demeaned LightGBM per horizon."""
    if SPATIALCV is None:
        return ("<p><i>Spatial CV results not available.</i></p>",
                "<p><i>Spatial CV results not available.</i></p>",
                {})

    def render(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p><i>(empty)</i></p>"
        g = _summarize_cv(df)
        rows = []
        for _, r in g.iterrows():
            rows.append(
                f"<tr>"
                f"<td>{HORIZON_CLASS.get(r['horizon'], r['horizon'])}</td>"
                f"<td>{int(r['n_folds'])}</td>"
                f"<td>{r['mean_spearman']:+.3f}</td>"
                f"<td>{r['std_spearman']:.3f}</td>"
                f"<td>{r['mean_spread']*100:+.2f}%</td>"
                f"<td>{r['std_spread']*100:.2f}%</td>"
                f"</tr>"
            )
        return _wrap_table(f"""
        <table class='headline'>
          <tr><th>horizon</th><th>n folds</th><th>mean Spearman ρ</th><th>std ρ</th>
              <th>mean decile spread</th><th>std decile spread</th></tr>
          {"".join(rows)}
        </table>
        """)

    raw = SPATIALCV[(SPATIALCV["model"] == "lightgbm") & (SPATIALCV["demean"] == False)]  # noqa: E712
    dem = SPATIALCV[(SPATIALCV["model"] == "lightgbm") & (SPATIALCV["demean"] == True)]  # noqa: E712

    # Stash fwd_3y numbers for inline interpretation in the body copy.
    quotes = {}
    for label, src in (("raw", raw), ("dem", dem)):
        s3 = src[src["horizon"] == "fwd_3y"]
        if not s3.empty:
            quotes[f"{label}_spearman"] = float(s3["spearman_test"].mean())
            quotes[f"{label}_spread"] = float(s3["decile_spread_test"].mean())

    return render(raw), render(dem), quotes


def leaderboard_band_stats() -> dict:
    """Compute median p10/p90 and the average band width across the zip leaderboard."""
    df = LEADERBOARD.copy()
    out = {
        "median_p10": float(df["pred_p10"].median()) if "pred_p10" in df else float("nan"),
        "median_p90": float(df["pred_p90"].median()) if "pred_p90" in df else float("nan"),
        "median_band": float((df["pred_p90"] - df["pred_p10"]).median()) if "pred_p10" in df and "pred_p90" in df else float("nan"),
    }
    return out


def chart_backtest_chart() -> go.Figure | None:
    """Time-series of top-10% / universe / bottom-10% realized fwd_3y per score month."""
    if BACKTEST_PRICE is None:
        return None
    df = BACKTEST_PRICE.sort_values("score_month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["score_month"], y=df["mean_realized_top10"], mode="lines+markers",
                              name="Top 10% basket", line=dict(color="#1a9850", width=2)))
    fig.add_trace(go.Scatter(x=df["score_month"], y=df["mean_realized_universe"], mode="lines",
                              name="Universe mean", line=dict(color="#666", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=df["score_month"], y=df["mean_realized_bot10"], mode="lines+markers",
                              name="Bottom 10% basket", line=dict(color="#a50026", width=2)))
    fig.update_yaxes(tickformat=".1%", title="Realized fwd_3y annualized return")
    fig.update_xaxes(title="Score month (basket formation date)")
    fig.update_layout(
        title="Top-decile vs bottom-decile vs universe — actual realized 3y returns per pick month",
        height=420, margin=dict(l=60, r=20, t=60, b=40), legend=dict(orientation="h", y=-0.2),
    )
    return fig


def chart_backtest_summary_table() -> str:
    """Two-row summary comparing price-only vs total-return model backtest performance."""
    rows = []
    for label, df in (("Price-only (fwd_3y)", BACKTEST_PRICE), ("Total-return (fwd_3y_total)", BACKTEST_TOTAL)):
        if df is None or df.empty:
            continue
        univ_size = int(df["n_zips_universe"].mean())
        top_size = int(df["n_zips_top10"].mean())
        top_mean = float(df["mean_realized_top10"].mean()) * 100
        univ_mean = float(df["mean_realized_universe"].mean()) * 100
        bot_mean = float(df["mean_realized_bot10"].mean()) * 100
        excess = top_mean - univ_mean
        top_bot = top_mean - bot_mean
        hit_rate = (df["mean_realized_top10"] > df["mean_realized_universe"]).mean() * 100
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td>{len(df)}</td><td>{univ_size:,}</td><td>~{top_size:,}</td>"
            f"<td>{top_mean:+.2f}%</td><td>{univ_mean:+.2f}%</td><td>{bot_mean:+.2f}%</td>"
            f"<td><b>{excess:+.2f} pp</b></td><td>{top_bot:+.2f} pp</td>"
            f"<td><b>{hit_rate:.1f}%</b></td></tr>"
        )
    return _wrap_table(f"""
    <table class='headline'>
      <tr><th>model</th><th>months</th><th>universe / mo</th><th>top-10% basket</th>
          <th>top-10% realized</th><th>universe realized</th><th>bottom-10% realized</th>
          <th>top excess</th><th>top − bot spread</th><th>hit rate (top &gt; univ)</th></tr>
      {"".join(rows)}
    </table>
    """)


def chart_total_leaderboard_top_n(n: int = 30) -> str:
    """Top-N zips by predicted total-return (price + ZORI rent carry)."""
    if ZIP_LB_TOTAL is None:
        return "<p><i>Total-return leaderboard not built — run scripts/run_phase2_total.py.</i></p>"
    df = ZIP_LB_TOTAL.copy().head(n)
    df["zhvi_fmt"] = df["zhvi"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    df["zori_fmt"] = df["zori"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    df["yield_fmt"] = df["rent_yield_t"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    df["pred_fmt"] = df["predicted_fwd_3y_total_annualized"].apply(lambda x: f"{x*100:+.2f}%")
    df["drivers_html"] = df["drivers"].apply(_humanize_drivers)
    df = df[["zip", "cbsa", "zhvi_fmt", "zori_fmt", "yield_fmt", "pred_fmt", "drivers_html"]]
    df.columns = ["ZIP", "CBSA", "ZHVI", "ZORI", "Gross yield", "Pred fwd_3y total ann.",
                  "Top SHAP drivers (hover for description)"]
    rows = []
    for row in df.values:
        cells = []
        for i, c in enumerate(row):
            if i == len(row) - 1:  # drivers cell, already HTML
                cells.append(f"<td>{c}</td>")
            else:
                cells.append(f"<td>{c}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "<tr>" + "".join(f"<th>{h}</th>" for h in df.columns) + "</tr>"
    return _wrap_table(f"<table class='leaderboard'>{header}{''.join(rows)}</table>")


# ---------------------------------------------------------------------------
# Macro context (section 6): timing chart + hedge comparison.
# Built from scripts/macro_context.py outputs.
# ---------------------------------------------------------------------------
TIMING_PARQUET = PROCESSED / "timing_universe_mean.parquet"
HEDGE_PARQUET = PROCESSED / "hedge_comparison.parquet"

# Friendlier display names for the hedge-comparison tables/charts.
HEDGE_LABELS = {
    "SP500":             "S&P 500 (price)",
    "SP500_TR":          "S&P 500 (total return)",
    "VTI_broad_market":  "VTI (broad US equity)",
    "Gold_futures":      "Gold (front-month futures)",
    "REIT_iShares_IYR":  "REITs (IYR ETF)",
    "REIT_Vanguard_VNQ": "REITs (VNQ ETF)",
    "Bitcoin":           "Bitcoin",
    "RE_ZHVI_top100":    "US residential RE (ZHVI top-100 mean)",
}
# Order for charts/tables: most-canonical / longest-history first.
HEDGE_ORDER = [
    "RE_ZHVI_top100", "SP500", "SP500_TR", "VTI_broad_market",
    "Gold_futures", "REIT_iShares_IYR", "REIT_Vanguard_VNQ", "Bitcoin",
]


def _load_timing() -> pd.DataFrame | None:
    if not TIMING_PARQUET.exists():
        return None
    df = pd.read_parquet(TIMING_PARQUET)
    df["year_month"] = pd.to_datetime(df["year_month"])
    return df.sort_values("year_month").reset_index(drop=True)


def _load_hedge() -> pd.DataFrame | None:
    if not HEDGE_PARQUET.exists():
        return None
    df = pd.read_parquet(HEDGE_PARQUET)
    df["year_month"] = pd.to_datetime(df["year_month"])
    return df.sort_values(["asset", "year_month"]).reset_index(drop=True)


def chart_timing_universe_mean(start_year: int = 2018) -> go.Figure | None:
    """Predicted vs realized universe-mean fwd_3y, post-2012 LightGBM."""
    df = _load_timing()
    if df is None or df.empty:
        return None
    df = df[df["year_month"] >= pd.Timestamp(f"{start_year}-01-01")].copy()
    latest = df.iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year_month"], y=df["predicted_universe_mean"],
        mode="lines", name="Predicted (model)",
        line=dict(color="#2c5282", width=2),
        hovertemplate="<b>%{x|%Y-%m}</b><br>Predicted: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["year_month"], y=df["realized_universe_mean"],
        mode="lines", name="Realized (subsequent 3y, annualized)",
        line=dict(color="#1a9850", width=2, dash="dash"),
        hovertemplate="<b>%{x|%Y-%m}</b><br>Realized: %{y:.2%}<extra></extra>",
    ))
    # Highlight the most recent month.
    fig.add_trace(go.Scatter(
        x=[latest["year_month"]], y=[latest["predicted_universe_mean"]],
        mode="markers+text", showlegend=False,
        marker=dict(size=11, color="#2c5282", line=dict(color="white", width=2)),
        text=[f"  current: {latest['predicted_universe_mean']:+.2%}"],
        textposition="middle right", textfont=dict(size=11, color="#2c5282"),
        hovertemplate="<b>%{x|%Y-%m}</b><br>Current: %{y:.2%}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#888", dash="dot", width=1))
    fig.update_yaxes(tickformat=".1%",
                     title="universe-mean fwd_3y annualized return")
    fig.update_xaxes(title="month")
    fig.update_layout(
        title=("Universe-mean fwd_3y: model prediction vs realized "
               "(top-100 metro ZIPs)"),
        height=420, margin=dict(l=70, r=140, t=60, b=40),
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def timing_summary_stats() -> dict:
    """Headline numbers for inline copy under the timing chart."""
    df = _load_timing()
    if df is None or df.empty:
        return {}
    latest = df.iloc[-1]
    current = float(latest["predicted_universe_mean"])
    all_preds = df["predicted_universe_mean"].dropna()
    median = float(all_preds.median())
    q25 = float(all_preds.quantile(0.25))
    q75 = float(all_preds.quantile(0.75))
    # Percentile of the current prediction across history.
    percentile = float((all_preds < current).mean()) * 100
    if percentile <= 25:
        verdict = "bearish"
    elif percentile >= 75:
        verdict = "bullish"
    else:
        verdict = "neutral"
    return {
        "current": current,
        "current_month": latest["year_month"],
        "n_zips": int(latest["n_zips"]),
        "p10": float(latest["predicted_universe_p10"]),
        "p90": float(latest["predicted_universe_p90"]),
        "median_hist": median,
        "q25_hist": q25,
        "q75_hist": q75,
        "percentile": percentile,
        "verdict": verdict,
    }


def chart_hedge_comparison(start_year: int = 2000) -> go.Figure | None:
    """Cumulative real-return index per asset, indexed to 100 at start."""
    df = _load_hedge()
    if df is None or df.empty:
        return None
    df = df[df["year_month"] >= pd.Timestamp(f"{start_year}-01-01")].copy()

    fig = go.Figure()
    palette = {
        "RE_ZHVI_top100":    "#d62728",  # red — the home team
        "SP500":             "#1f77b4",
        "SP500_TR":          "#3a5da9",
        "VTI_broad_market":  "#5fa2dd",
        "Gold_futures":      "#bf9000",
        "REIT_iShares_IYR":  "#2ca02c",
        "REIT_Vanguard_VNQ": "#7fbf7f",
        "Bitcoin":           "#9b59b6",
    }
    for asset in HEDGE_ORDER:
        sub = df[df["asset"] == asset].dropna(subset=["cum_real"])
        if sub.empty:
            continue
        # Re-base to 100 at this asset's first observation (so short-history
        # assets like BTC start at 100 in 2014, not show a giant offset).
        first = float(sub["cum_real"].iloc[0])
        norm = sub["cum_real"] / first * 100.0
        fig.add_trace(go.Scatter(
            x=sub["year_month"], y=norm,
            mode="lines", name=HEDGE_LABELS.get(asset, asset),
            line=dict(color=palette.get(asset, "#666"), width=2),
            hovertemplate=("<b>%{x|%Y-%m}</b><br>"
                           f"{HEDGE_LABELS.get(asset, asset)}: "
                           "%{y:.1f}<extra></extra>"),
        ))
    fig.update_yaxes(type="log",
                     title="cumulative real return (log scale, indexed to 100)")
    fig.update_xaxes(title="month")
    fig.update_layout(
        title=("Cumulative real return per asset since 2000 "
               "(BTC/VNQ/IYR rebased to 100 at first observation)"),
        height=460, margin=dict(l=70, r=20, t=60, b=40),
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def hedge_historical_table() -> str:
    """Median 3y / 5y real return, std, and share-positive across all rolling
    windows since 2000."""
    df = _load_hedge()
    if df is None or df.empty:
        return "<p><i>Hedge comparison not built — run scripts/macro_context.py.</i></p>"
    rows = []
    for asset in HEDGE_ORDER:
        sub = df[df["asset"] == asset]
        if sub.empty:
            continue
        r3 = sub["ann_3y_real"].dropna()
        r5 = sub["ann_5y_real"].dropna()
        if r3.empty:
            continue
        first_year = int(sub["year_month"].min().year)
        rows.append({
            "asset": HEDGE_LABELS.get(asset, asset),
            "since": first_year,
            "median_3y_real": float(r3.median()),
            "std_3y_real": float(r3.std()),
            "share_pos_3y": float((r3 > 0).mean()),
            "median_5y_real": float(r5.median()) if not r5.empty else float("nan"),
            "share_pos_5y": float((r5 > 0).mean()) if not r5.empty else float("nan"),
        })
    if not rows:
        return "<p><i>Hedge comparison empty.</i></p>"
    body = []
    for r in rows:
        five = ("—" if pd.isna(r["median_5y_real"])
                else f"{r['median_5y_real']*100:+.2f}%")
        five_pos = ("—" if pd.isna(r["share_pos_5y"])
                    else f"{r['share_pos_5y']*100:.0f}%")
        body.append(
            f"<tr><td>{escape(r['asset'])}</td>"
            f"<td>{r['since']}</td>"
            f"<td>{r['median_3y_real']*100:+.2f}%</td>"
            f"<td>{r['std_3y_real']*100:.2f}%</td>"
            f"<td>{r['share_pos_3y']*100:.0f}%</td>"
            f"<td>{five}</td>"
            f"<td>{five_pos}</td></tr>"
        )
    return _wrap_table(
        "<table class='headline'>"
        "<tr><th>asset</th><th>history starts</th>"
        "<th>median 3y real return</th><th>std 3y real</th>"
        "<th>3y windows positive</th>"
        "<th>median 5y real return</th><th>5y windows positive</th></tr>"
        + "".join(body) + "</table>"
    )


def hedge_trailing_table() -> str:
    """Trailing 3y real + nominal return ending the most recent month, plus
    its spread vs the trailing 10y median for that asset."""
    df = _load_hedge()
    if df is None or df.empty:
        return "<p><i>Hedge comparison not built — run scripts/macro_context.py.</i></p>"
    rows = []
    latest_month = df["year_month"].max()
    rows_data = []
    for asset in HEDGE_ORDER:
        sub = df[df["asset"] == asset].sort_values("year_month")
        if sub.empty:
            continue
        # Most-recent (or last non-null) trailing 3y window.
        latest = sub.dropna(subset=["ann_3y_real"]).tail(1)
        if latest.empty:
            continue
        ann_real = float(latest["ann_3y_real"].iloc[0])
        ann_nom = float(latest["ann_3y_nominal"].iloc[0])
        ts = latest["year_month"].iloc[0]
        # Trailing-10y median of 3y-real windows ending up to today.
        cutoff = ts - pd.DateOffset(years=10)
        window = sub[(sub["year_month"] > cutoff)
                     & (sub["year_month"] <= ts)]["ann_3y_real"].dropna()
        med10 = float(window.median()) if not window.empty else float("nan")
        spread = ann_real - med10 if not np.isnan(med10) else float("nan")
        rows_data.append({
            "asset": HEDGE_LABELS.get(asset, asset),
            "ts": ts,
            "ann_real": ann_real,
            "ann_nom": ann_nom,
            "med10": med10,
            "spread": spread,
        })

    # Sort by trailing real return descending.
    rows_data.sort(key=lambda r: r["ann_real"], reverse=True)
    for r in rows_data:
        sp_html = ("—" if np.isnan(r["spread"])
                   else f"{r['spread']*100:+.2f} pp")
        med_html = ("—" if np.isnan(r["med10"])
                    else f"{r['med10']*100:+.2f}%")
        rows.append(
            f"<tr><td>{escape(r['asset'])}</td>"
            f"<td>{r['ts'].date()}</td>"
            f"<td>{r['ann_real']*100:+.2f}%</td>"
            f"<td>{r['ann_nom']*100:+.2f}%</td>"
            f"<td>{med_html}</td>"
            f"<td>{sp_html}</td></tr>"
        )
    return _wrap_table(
        "<table class='headline'>"
        "<tr><th>asset</th><th>window ends</th>"
        "<th>trailing 3y real return</th><th>trailing 3y nominal</th>"
        "<th>trailing-10y median 3y real</th>"
        "<th>spread vs 10y median</th></tr>"
        + "".join(rows) + "</table>"
    )


def hedge_summary_for_copy() -> dict:
    """Stash numbers for inline body copy."""
    df = _load_hedge()
    if df is None or df.empty:
        return {}
    medians = (
        df.dropna(subset=["ann_3y_real"])
          .groupby("asset")["ann_3y_real"].median()
          .sort_values(ascending=False)
    )
    # Trailing 3y winner.
    trailing = []
    for asset in HEDGE_ORDER:
        sub = df[df["asset"] == asset].dropna(subset=["ann_3y_real"])
        if sub.empty:
            continue
        last = sub.iloc[-1]
        trailing.append((asset, float(last["ann_3y_real"])))
    trailing.sort(key=lambda x: x[1], reverse=True)
    zhvi_sub = df[df["asset"] == "RE_ZHVI_top100"].dropna(subset=["ann_3y_real"])
    zhvi_trailing = float(zhvi_sub["ann_3y_real"].iloc[-1]) if not zhvi_sub.empty else float("nan")
    return {
        "medians": [(HEDGE_LABELS.get(a, a), float(v)) for a, v in medians.items()],
        "trailing": [(HEDGE_LABELS.get(a, a), v) for a, v in trailing],
        "zhvi_trailing": zhvi_trailing,
    }


def backtest_summary_stats() -> dict:
    """Pull headline numbers for inline TL;DR copy."""
    def stats(df):
        if df is None or df.empty:
            return {"hit": float("nan"), "excess": float("nan"), "spread": float("nan"), "months": 0}
        return {
            "hit": float((df["mean_realized_top10"] > df["mean_realized_universe"]).mean()) * 100,
            "excess": float((df["mean_realized_top10"] - df["mean_realized_universe"]).mean()) * 100,
            "spread": float((df["mean_realized_top10"] - df["mean_realized_bot10"]).mean()) * 100,
            "months": int(len(df)),
        }
    return {"price": stats(BACKTEST_PRICE), "total": stats(BACKTEST_TOTAL)}


# ---------------------------------------------------------------------------
# Extra visuals: decile-by-decile curve, equity curve, calibration scatter,
# quantile fan, CV schematics, forward-return illustration, SHAP waterfalls.
# All built around the fwd_3y / post_2012 LightGBM (headline horizon × split).
# ---------------------------------------------------------------------------

def _headline_test_predictions(horizon: str = "fwd_3y", split: str = "post_2012") -> pd.DataFrame:
    """Score the headline model on its test window. Cached on the function attr."""
    cache_key = f"_cache_{horizon}_{split}"
    if hasattr(_headline_test_predictions, cache_key):
        return getattr(_headline_test_predictions, cache_key)
    booster = lgb.Booster(model_file=str(ARTIFACTS / f"{horizon}__{split}__lgbm.txt"))
    features = (ARTIFACTS / f"{horizon}__{split}__lgbm.features.txt").read_text().strip().split("\n")
    mask = FS[f"is_test_{split}"].fillna(False) & FS[horizon].notna()
    X = FS.loc[mask, features].copy()
    X = X.fillna(X.median(numeric_only=True))
    out = pd.DataFrame({
        "zip": FS.loc[mask, "zip"].to_numpy(),
        "cbsa": FS.loc[mask, "cbsa"].to_numpy(),
        "year_month": FS.loc[mask, "year_month"].to_numpy(),
        "predicted": booster.predict(X),
        "realized": FS.loc[mask, horizon].to_numpy(),
    }).dropna(subset=["predicted", "realized"]).reset_index(drop=True)
    setattr(_headline_test_predictions, cache_key, out)
    return out


def chart_decile_curve() -> go.Figure:
    """All 10 deciles realized return — not just the top-bottom spread. Shows
    whether the signal is monotone across the whole rank."""
    df = _headline_test_predictions()
    df["decile"] = pd.qcut(df["predicted"], 10, labels=False, duplicates="drop") + 1
    g = df.groupby("decile")["realized"].agg(["mean", "std", "count"]).reset_index()
    colors = ["#a50026"] + ["#5a7ec3"] * 8 + ["#1a9850"]
    fig = go.Figure(go.Bar(
        x=g["decile"], y=g["mean"], marker_color=colors,
        error_y=dict(type="data", array=g["std"] / np.sqrt(g["count"]), thickness=1),
        hovertemplate="Decile %{x}<br>Realized fwd_3y ann.: %{y:.2%}<br>n=%{customdata:,}<extra></extra>",
        customdata=g["count"],
    ))
    fig.add_hline(y=df["realized"].mean(), line=dict(color="#666", dash="dash"),
                  annotation_text="universe mean", annotation_position="top right")
    fig.update_yaxes(tickformat=".1%", title="Realized fwd_3y annualized return")
    fig.update_xaxes(title="Predicted decile (1 = lowest, 10 = highest)", dtick=1)
    fig.update_layout(
        title="Realized return by predicted decile — fwd_3y post-2012 (LightGBM)",
        height=380, margin=dict(l=60, r=20, t=60, b=50), showlegend=False,
    )
    return fig


def chart_equity_curve() -> go.Figure | None:
    """Wealth-per-cohort: $1 invested at each score_month, terminal value after a 3y hold.
    Three series: top-decile basket vs universe vs bottom-decile."""
    if BACKTEST_PRICE is None or BACKTEST_PRICE.empty:
        return None
    df = BACKTEST_PRICE.sort_values("score_month").copy()
    df["score_month"] = pd.to_datetime(df["score_month"])
    for col in ["mean_realized_top10", "mean_realized_universe", "mean_realized_bot10"]:
        df[col.replace("mean_realized_", "wealth_")] = (1.0 + df[col]) ** 3

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["score_month"], y=df["wealth_top10"], mode="lines+markers",
        name="$1 → top-decile basket (3y hold)", line=dict(color="#1a9850", width=2),
        hovertemplate="Entry %{x|%Y-%m}<br>Terminal: $%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["score_month"], y=df["wealth_universe"], mode="lines",
        name="$1 → universe (equal-weight)", line=dict(color="#666", width=1, dash="dash"),
        hovertemplate="Entry %{x|%Y-%m}<br>Terminal: $%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["score_month"], y=df["wealth_bot10"], mode="lines+markers",
        name="$1 → bottom-decile basket (3y hold)", line=dict(color="#a50026", width=2),
        hovertemplate="Entry %{x|%Y-%m}<br>Terminal: $%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line=dict(color="#999", width=1))
    fig.update_yaxes(title="Terminal wealth from $1 (3-year hold)", tickformat="$.2f")
    fig.update_xaxes(title="Cohort entry month")
    fig.update_layout(
        title="Per-cohort terminal wealth — $1 invested at each month, held 3 years",
        height=420, margin=dict(l=60, r=20, t=60, b=40), legend=dict(orientation="h", y=-0.2),
    )
    return fig


def chart_calibration_scatter() -> go.Figure:
    """Predicted vs realized fwd_3y, binned into 20 quantile buckets.
    Distinguishes a good ranker from a well-calibrated forecaster."""
    df = _headline_test_predictions()
    df["bin"] = pd.qcut(df["predicted"], 20, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(
        pred_mean=("predicted", "mean"),
        real_mean=("realized", "mean"),
        n=("realized", "size"),
        real_std=("realized", "std"),
    ).reset_index()
    lo = float(min(g["pred_mean"].min(), g["real_mean"].min()))
    hi = float(max(g["pred_mean"].max(), g["real_mean"].max()))
    pad = (hi - lo) * 0.05

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
        mode="lines", line=dict(color="#999", dash="dash"), name="perfect calibration",
        hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=g["pred_mean"], y=g["real_mean"], mode="markers",
        marker=dict(size=9, color=g["bin"], colorscale="Viridis",
                    showscale=False, line=dict(width=1, color="white")),
        error_y=dict(type="data", array=g["real_std"] / np.sqrt(g["n"]), thickness=1),
        name="binned (predicted, realized)",
        customdata=np.stack([g["n"], g["bin"] + 1], axis=1),
        hovertemplate=("Bucket %{customdata[1]}/20<br>"
                       "Mean predicted: %{x:.2%}<br>"
                       "Mean realized: %{y:.2%}<br>"
                       "n=%{customdata[0]:,}<extra></extra>"),
    ))
    fig.update_xaxes(title="Mean predicted fwd_3y (annualized)", tickformat=".1%")
    fig.update_yaxes(title="Mean realized fwd_3y (annualized)", tickformat=".1%")
    fig.update_layout(
        title="Calibration — predicted vs realized in 20 quantile buckets (post-2012 test)",
        height=440, margin=dict(l=60, r=20, t=60, b=40), legend=dict(orientation="h", y=-0.18),
    )
    return fig


def chart_quantile_fan() -> go.Figure | None:
    """Universe-mean predicted return over time, with p10-p90 band from the quantile models."""
    if not TIMING_PARQUET.exists():
        return None
    df = pd.read_parquet(TIMING_PARQUET)
    df["year_month"] = pd.to_datetime(df["year_month"])
    df = df.sort_values("year_month").reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year_month"], y=df["predicted_universe_p90"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["year_month"], y=df["predicted_universe_p10"],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(44, 82, 130, 0.18)",
        name="80% prediction interval (p10–p90)",
        hovertemplate="<b>%{x|%Y-%m}</b><br>p10: %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["year_month"], y=df["predicted_universe_median"],
        mode="lines", name="median forecast (p50)",
        line=dict(color="#2c5282", width=2),
        hovertemplate="<b>%{x|%Y-%m}</b><br>p50: %{y:.2%}<extra></extra>",
    ))
    if "realized_universe_mean" in df.columns and df["realized_universe_mean"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["year_month"], y=df["realized_universe_mean"],
            mode="lines", name="realized (subsequent 3y, annualized)",
            line=dict(color="#1a9850", width=2, dash="dash"),
            hovertemplate="<b>%{x|%Y-%m}</b><br>realized: %{y:.2%}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.update_yaxes(tickformat=".1%", title="Predicted universe-mean fwd_3y (annualized)")
    fig.update_xaxes(title="Score month")
    fig.update_layout(
        title="Universe-mean forecast with 80% uncertainty band — quantile LightGBM",
        height=430, margin=dict(l=60, r=20, t=60, b=40), legend=dict(orientation="h", y=-0.2),
    )
    return fig


def chart_walkforward_schematic() -> go.Figure:
    """Strip diagram of train + 1-year-test windows sliding across the calendar.
    Communicates 'rolling origin' visually with no real data needed."""
    folds = [
        ("Fold 1", 2013, 2018, 2018),
        ("Fold 2", 2013, 2019, 2019),
        ("Fold 3", 2013, 2020, 2020),
        ("Fold 4", 2013, 2021, 2021),
        ("Fold 5", 2013, 2022, 2022),
    ]
    fig = go.Figure()
    for label, train_start, train_end_exclusive, test_year in folds:
        fig.add_trace(go.Bar(
            y=[label], x=[train_end_exclusive - train_start],
            base=[train_start], orientation="h",
            marker_color="#5a7ec3", name="train", showlegend=(label == "Fold 1"),
            hovertemplate=f"Train: {train_start} – {train_end_exclusive - 1}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=[label], x=[1], base=[test_year], orientation="h",
            marker_color="#1a9850", name="test (1y window)", showlegend=(label == "Fold 1"),
            hovertemplate=f"Test: {test_year}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", title="Walk-forward CV — test year slides forward; train uses everything before it",
        xaxis_title="Calendar year", yaxis_title="",
        height=300, margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    fig.update_xaxes(dtick=1, range=[2012.5, 2023.5])
    return fig


def chart_spatial_fold_matrix() -> go.Figure | None:
    """Show how the top-100 CBSAs partition into 5 disjoint folds.
    Each dot = one CBSA, sized by population, colored by fold assignment."""
    if TOP_METROS is None:
        return None
    from sklearn.model_selection import GroupKFold
    cbsas = TOP_METROS.copy().reset_index(drop=True)
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(cbsas["cbsa"].to_numpy())
    fold_map = {}
    gkf = GroupKFold(n_splits=5)
    X = np.arange(len(shuffled)).reshape(-1, 1)
    for i, (_, test_idx) in enumerate(gkf.split(X, groups=shuffled)):
        for j in test_idx:
            fold_map[shuffled[j]] = i + 1
    cbsas["fold"] = cbsas["cbsa"].map(fold_map)
    cbsas["state_primary"] = cbsas["state"].str.split("-").str[0]
    cbsas["city_primary"] = cbsas["core_name"].str.split("-").str[0]
    cbsas = cbsas.sort_values("population", ascending=False).reset_index(drop=True)
    cbsas["rank"] = cbsas.index + 1
    fig = go.Figure()
    palette = {1: "#a50026", 2: "#fdae61", 3: "#5a7ec3", 4: "#1a9850", 5: "#762a83"}
    for f in sorted(cbsas["fold"].unique()):
        sub = cbsas[cbsas["fold"] == f]
        fig.add_trace(go.Scatter(
            x=sub["rank"], y=[f] * len(sub),
            mode="markers",
            marker=dict(size=np.sqrt(sub["population"] / 50_000),
                        color=palette[f], opacity=0.75, line=dict(color="white", width=0.5)),
            name=f"Fold {f}",
            customdata=np.stack([sub["city_primary"], sub["state_primary"], sub["population"]], axis=1),
            hovertemplate=("<b>%{customdata[0]}, %{customdata[1]}</b><br>"
                           "Population: %{customdata[2]:,}<br>"
                           f"Held out in fold {f}<extra></extra>"),
        ))
    fig.update_yaxes(title="Spatial-CV fold (held out)", dtick=1, autorange="reversed")
    fig.update_xaxes(title="Top-100 CBSA, ranked by population (1 = NYC, 100 = smallest)")
    fig.update_layout(
        title="Spatial CV — each top-100 metro is held out in exactly one of five folds",
        height=320, margin=dict(l=60, r=20, t=60, b=50),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def chart_forward_return_illustration(anchor_str: str = "2018-01-01") -> go.Figure | None:
    """One illustrative ZIP: ZHVI line with shaded fwd_1y/3y/5y windows from one anchor date."""
    anchor = pd.Timestamp(anchor_str)
    candidates = LEADERBOARD["zip"].head(5).tolist() if LEADERBOARD is not None else []
    candidates += ["94110", "10025", "60618"]
    chosen = None
    for z in candidates:
        sub = FS[(FS["zip"] == z) & FS["zhvi"].notna()].sort_values("year_month")
        if len(sub) < 200:
            continue
        sub_dates = pd.to_datetime(sub["year_month"])
        if sub_dates.min() <= anchor and sub_dates.max() >= anchor + pd.DateOffset(years=5):
            chosen = (z, sub)
            break
    if chosen is None:
        return None
    z, sub = chosen
    sub = sub.copy()
    sub["year_month"] = pd.to_datetime(sub["year_month"])
    anchor_row = sub.loc[sub["year_month"] == anchor]
    if anchor_row.empty:
        return None
    anchor_zhvi = float(anchor_row["zhvi"].iloc[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["year_month"], y=sub["zhvi"], mode="lines",
        line=dict(color="#2c5282", width=2),
        name=f"ZHVI — ZIP {z}",
        hovertemplate="<b>%{x|%Y-%m}</b><br>ZHVI: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[anchor], y=[anchor_zhvi], mode="markers+text",
        marker=dict(size=12, color="#1a1a1a", symbol="diamond"),
        text=[f"prediction date<br>{anchor.strftime('%Y-%m')}"],
        textposition="bottom right", textfont=dict(size=11), showlegend=False, hoverinfo="skip",
    ))
    # Draw widest band first so narrower bands paint over it; layered shading
    # lets the eye see all three windows at once (1y darkest, 5y faintest).
    bands = [("fwd_5y", "rgba(165,0,38,.13)", 5),
             ("fwd_3y", "rgba(44,82,130,.16)", 3),
             ("fwd_1y", "rgba(26,152,80,.25)", 1)]
    for name, color, years in bands:
        fig.add_vrect(
            x0=anchor, x1=anchor + pd.DateOffset(years=years),
            fillcolor=color, line_width=0,
        )
    # Manual annotations at the right edge of each band so labels don't stack.
    for name, _, years in bands:
        end = anchor + pd.DateOffset(years=years)
        fig.add_annotation(
            x=end, y=1.0, xref="x", yref="y domain",
            text=f"<b>{name}</b> end ({end.strftime('%Y-%m')})",
            showarrow=False, font=dict(size=11, color="#444"),
            xanchor="right", yanchor="bottom", yshift=2,
        )
    fig.add_vline(x=anchor, line=dict(color="#1a1a1a", width=1, dash="dot"))
    fig.update_yaxes(title="ZHVI ($)", tickformat="$,.0f")
    fig.update_xaxes(title="Month")
    fig.update_layout(
        title=f"What 'forward return' means — example: ZIP {z}",
        height=380, margin=dict(l=60, r=20, t=60, b=40), showlegend=False,
    )
    return fig


def chart_shap_waterfall(horizon: str = "fwd_3y", split: str = "post_2012",
                         k: int = 10) -> tuple[go.Figure | None, go.Figure | None]:
    """Two waterfalls: one for a top-decile and one for a bottom-decile pick at the
    most recent scoring month — the k features with the largest |SHAP| contribution."""
    booster = lgb.Booster(model_file=str(ARTIFACTS / f"{horizon}__{split}__lgbm.txt"))
    features = (ARTIFACTS / f"{horizon}__{split}__lgbm.features.txt").read_text().strip().split("\n")
    df = _headline_test_predictions(horizon, split)
    if df.empty:
        return None, None
    latest = df["year_month"].max()
    df = df[df["year_month"] == latest].copy()
    if df.empty:
        return None, None
    top_row = df.loc[df["predicted"].idxmax()]
    bot_row = df.loc[df["predicted"].idxmin()]

    train_mask = FS[f"is_train_{split}"].fillna(False)
    med = FS.loc[train_mask, features].median(numeric_only=True)
    explainer = shap.TreeExplainer(booster)

    def _build(row, label_color):
        mask = (FS["zip"] == row["zip"]) & (FS["year_month"] == row["year_month"])
        X_one = FS.loc[mask, features].copy()
        if X_one.empty:
            return None
        X_one = X_one.fillna(med)
        sv = explainer.shap_values(X_one)
        if isinstance(sv, list):
            sv = sv[0]
        contrib = sv[0]
        base = float(explainer.expected_value if np.isscalar(explainer.expected_value)
                     else explainer.expected_value[0])
        order = np.argsort(np.abs(contrib))[::-1][:k]
        top_feats = [features[i] for i in order]
        top_vals = [float(contrib[i]) for i in order]
        top_x = [float(X_one.iloc[0][f]) for f in top_feats]
        idx = np.argsort(top_vals)
        top_feats = [top_feats[i] for i in idx]
        top_vals = [top_vals[i] for i in idx]
        top_x = [top_x[i] for i in idx]
        display = [feat_label(f)[0] for f in top_feats]
        descrip = [feat_label(f)[1] for f in top_feats]
        bar_colors = ["#1a9850" if v > 0 else "#a50026" for v in top_vals]

        fig = go.Figure(go.Bar(
            x=top_vals, y=display, orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.4f}" for v in top_vals], textposition="outside",
            customdata=np.stack([top_feats, descrip, top_x], axis=1),
            hovertemplate=("<b>%{y}</b><br>"
                           "<i>%{customdata[0]}</i><br>"
                           "feature value at this ZIP: %{customdata[2]:.4f}<br>"
                           "SHAP contribution: %{x:+.4f}<br><br>"
                           "%{customdata[1]}<extra></extra>"),
        ))
        fig.add_vline(x=0, line=dict(color="#666", width=1))
        title = (f"ZIP {row['zip']} · CBSA {row['cbsa']} · "
                 f"{pd.Timestamp(row['year_month']).strftime('%Y-%m')}<br>"
                 f"<span style='color:{label_color};font-size:13px'>"
                 f"predicted fwd_3y ann. = {row['predicted']*100:+.2f}%, "
                 f"realized = {row['realized']*100:+.2f}%, "
                 f"model base rate = {base*100:+.2f}%</span>")
        fig.update_xaxes(title="Per-feature SHAP contribution (log-return units)", tickformat="+.3f")
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            height=420, margin=dict(l=240, r=80, t=80, b=40), showlegend=False,
        )
        return fig

    return _build(top_row, "#1a9850"), _build(bot_row, "#a50026")




# ---------------------------------------------------------------------------
# Shared HTML scaffolding (head, CSS, TOC, sortable-table JS, scroll-spy).
# Both index.html and methods.html share this; only the body + title differ.
# ---------------------------------------------------------------------------

PAGE_TITLE_BASE = "arbok — what predicts US residential real-estate returns?"
OG_DESCRIPTION = (
    "A quant study of what predicts US residential real-estate returns: "
    "3.73M-row zip-month panel, 140+ free public predictors across 11 "
    "thematic classes, LightGBM + SHAP attribution, backtested 2018-2021."
)[:200]


def _html_head(page_title: str) -> str:
    """Shared <head>: meta tags, OG/Twitter cards, favicon, inlined CSS block.

    Plotly is inlined into the first chart of each page's body (see _div with
    include_js=True), not into <head>, so this returns ONLY the static head
    elements + the shared <style> block.
    """
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>{escape(page_title)}</title>
  <link rel='icon' href='assets/favicon.svg?v={BUILD_DATE}' type='image/svg+xml'>
  <meta name='description' content="{escape(OG_DESCRIPTION)}">
  <meta property='og:type' content='article'>
  <meta property='og:title' content="{escape(page_title)}">
  <meta property='og:description' content="{escape(OG_DESCRIPTION)}">
  <meta property='og:image' content='assets/og-image.png?v={BUILD_DATE}'>
  <meta property='og:url' content='{escape(SITE_URL)}'>
  <meta name='twitter:card' content='summary_large_image'>
  <meta name='twitter:title' content="{escape(page_title)}">
  <meta name='twitter:description' content="{escape(OG_DESCRIPTION)}">
  <meta name='twitter:image' content='assets/og-image.png?v={BUILD_DATE}'>
  <style>
    /* CSS Grid layout: sidebar TOC on the left, article on the right.
       Below 1100px the TOC column collapses; the mobile <details> jump-menu
       at the top of <article> picks up the slack. */
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
            margin: 0; color: #1a1a1a; line-height: 1.55; }}
    .layout {{ display: grid;
               grid-template-columns: 14em minmax(0, 1100px);
               gap: 2em;
               max-width: 1400px;
               margin: 2em auto;
               padding: 0 1em; }}
    nav.toc {{ position: sticky; top: 2em; align-self: start;
               font-size: .88em; max-height: calc(100vh - 4em);
               overflow-y: auto; padding-right: .5em; }}
    nav.toc h4 {{ margin: 0 0 .6em 0; color: #2c5282; font-size: .82em;
                  text-transform: uppercase; letter-spacing: .05em; }}
    nav.toc ul {{ list-style: none; padding: 0; margin: 0; }}
    nav.toc li {{ margin: .35em 0; }}
    nav.toc a {{ color: #444; text-decoration: none;
                 border-left: 2px solid transparent; padding-left: .6em;
                 display: block; line-height: 1.35; }}
    nav.toc a:hover {{ color: #2c5282; }}
    nav.toc a.active {{ color: #2c5282; font-weight: 600;
                        border-left-color: #2c5282; }}
    nav.toc .cross-link {{ margin-top: 1.4em; padding-top: .8em;
                           border-top: 1px solid #e0e6ef; font-size: .9em; }}
    nav.toc .cross-link a {{ color: #2c5282; font-weight: 500;
                              border-left: none; padding-left: 0; }}
    article {{ min-width: 0; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; max-width: 1100px; }}
      nav.toc {{ display: none; }}
    }}
    details.toc-mobile {{ display: none; margin: 1em 0;
                          background: #f8f9fb; border: 1px solid #e0e6ef;
                          padding: .6em 1em; border-radius: 4px; }}
    details.toc-mobile summary {{ cursor: pointer; font-weight: 600;
                                  color: #2c5282; }}
    details.toc-mobile ul {{ list-style: none; padding-left: 0;
                              margin: .8em 0 .2em 0; }}
    details.toc-mobile li {{ margin: .3em 0; }}
    details.toc-mobile a {{ color: #2c5282; text-decoration: none; }}
    @media (max-width: 1100px) {{
      details.toc-mobile {{ display: block; }}
    }}
    h1 {{ border-bottom: 2px solid #1a1a1a; padding-bottom: .3em; margin-bottom: .2em; }}
    h2 {{ margin-top: 2.2em; color: #2c5282; padding-bottom: .2em;
          border-bottom: 1px solid #e0e6ef; scroll-margin-top: 1em; }}
    h3 {{ color: #2c5282; margin-top: 1.8em; scroll-margin-top: 1em; }}
    p {{ margin: .8em 0; }}
    table {{ border-collapse: collapse; margin: 1em 0; }}
    th, td {{ padding: .35em .8em; border-bottom: 1px solid #ddd; text-align: left;
              font-size: .92em; }}
    th {{ background: #f3f5f9; }}
    table.headline td:nth-child(4) {{ font-weight: bold; color: #2c5282; }}
    table.headline tr.split-header td {{ background: #eef2f8; font-weight: 600;
                                          color: #2c5282; font-size: .85em;
                                          text-transform: uppercase; letter-spacing: .04em;
                                          padding-top: .55em; }}
    table.leaderboard {{ font-family: ui-monospace, 'SF Mono', monospace;
                         font-size: .82em; }}
    table.leaderboard td:nth-child(6) {{ font-weight: bold; color: #2c5282; }}
    /* Sortable-table affordance: a small arrow rendered after the header text
       so users see the columns are clickable. The JS at the bottom of <body>
       toggles 'sort-asc' / 'sort-desc' classes that swap the indicator. */
    table.sortable th {{ cursor: pointer; user-select: none; position: relative;
                          padding-right: 1.4em; }}
    table.sortable th::after {{ content: '\\2195'; position: absolute;
                                 right: .4em; color: #aaa; font-size: .85em; }}
    table.sortable th.sort-asc::after {{ content: '\\25B2'; color: #2c5282; }}
    table.sortable th.sort-desc::after {{ content: '\\25BC'; color: #2c5282; }}
    /* Key-takeaway callout: appears immediately under each <h2>. */
    .takeaway {{ background: #eef4fb; border-left: 4px solid #2c5282;
                 padding: .8em 1.1em; margin: .8em 0 1.4em 0;
                 border-radius: 0 4px 4px 0; font-size: .96em; }}
    .takeaway b {{ color: #2c5282; }}
    /* Tooltips (preserved exactly from the previous report). */
    table.leaderboard span.tt {{ position: relative; border-bottom: 1px dotted #999;
                                 cursor: help; outline: none; }}
    table.leaderboard span.tt .tt-body {{
        display: none; position: absolute; left: 0; top: 1.4em; z-index: 30;
        background: #1a1a1a; color: #fff; padding: .45em .65em;
        border-radius: 4px; font-size: .82em; line-height: 1.35;
        max-width: 22em; white-space: normal;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        box-shadow: 0 2px 6px rgba(0,0,0,.25);
    }}
    table.leaderboard span.tt:hover .tt-body,
    table.leaderboard span.tt:focus .tt-body,
    table.leaderboard span.tt:focus-within .tt-body {{ display: block; }}
    .meta {{ color: #666; font-size: .9em; margin-bottom: 1.8em; }}
    .summary {{ background: #f3f5f9; border-left: 4px solid #2c5282;
                padding: 1em 1.4em; margin: 1.5em 0; border-radius: 0 4px 4px 0; }}
    .summary h3 {{ margin-top: 0; color: #2c5282; }}
    .primer {{ background: #f8f9fb; border: 1px solid #e0e6ef;
               padding: 1em 1.4em; margin: 1.5em 0; border-radius: 4px;
               font-size: .94em; }}
    .primer h3 {{ margin-top: 0; color: #2c5282; }}
    .primer dt {{ font-weight: 600; margin-top: .55em; }}
    .primer dd {{ margin: 0 0 .2em 0; }}
    .caveat {{ background: #fff8e6; border-left: 4px solid #c9a227;
               padding: .9em 1.2em; margin: 1.2em 0; font-size: .93em;
               border-radius: 0 4px 4px 0; }}
    code {{ background: #f3f5f9; padding: 1px 5px; border-radius: 3px;
            font-size: .9em; }}
    .footnote {{ color: #666; font-size: .85em; margin-top: 3em;
                 border-top: 1px solid #eee; padding-top: 1em; }}
    .cross-page {{ background: #f3f5f9; border-left: 4px solid #2c5282;
                    padding: 1em 1.4em; margin: 2.5em 0 1em 0;
                    border-radius: 0 4px 4px 0; }}
    .cross-page a {{ color: #2c5282; font-weight: 600; }}
    .nojs-banner {{ background: #fff8e6; border-left: 4px solid #c9a227;
                    padding: .9em 1.2em; margin: 1em 0; font-size: .93em;
                    border-radius: 0 4px 4px 0; }}
    details {{ margin: .8em 0; }}
    details summary {{ cursor: pointer; font-weight: 600; color: #2c5282; }}
    details[open] summary {{ margin-bottom: .6em; }}
    ul {{ margin: .4em 0; }}
    li {{ margin: .2em 0; }}
    div[style*='overflow-x:auto'] {{ -webkit-overflow-scrolling: touch; }}
  </style>
</head>"""


# Vanilla-JS sortable-table + scroll-spy. Inlined at the bottom of <body> on
# both pages. Keeps the report dependency-free.
PAGE_SCRIPTS = """
<script>
(function(){
  // ---- Sortable tables ----
  function parseCell(text) {
    // Parse '+5.23%' / '-1.40%' / '$1,234,567' / '12,345' / '—' robustly.
    if (text == null) return -Infinity;
    var t = String(text).trim();
    if (t === '' || t === '—' || t === '-' || t === 'n/a') return -Infinity;
    // Strip $, commas, percent, plus/minus prefix
    var cleaned = t.replace(/[$,%\\s]/g, '');
    if (cleaned.charAt(0) === '+') cleaned = cleaned.slice(1);
    var n = parseFloat(cleaned);
    if (!isNaN(n) && /^[\\-+]?\\d/.test(cleaned)) return n;
    return text.toLowerCase();
  }
  function compare(a, b) {
    if (typeof a === 'number' && typeof b === 'number') return a - b;
    if (a < b) return -1; if (a > b) return 1; return 0;
  }
  function sortTable(tbl, colIdx, asc) {
    var rows = Array.from(tbl.querySelectorAll('tbody tr, tr')).filter(function(r){
      return r.querySelector('td');
    });
    rows.sort(function(r1, r2){
      var a = parseCell(r1.children[colIdx] && r1.children[colIdx].innerText);
      var b = parseCell(r2.children[colIdx] && r2.children[colIdx].innerText);
      var c = compare(a, b);
      return asc ? c : -c;
    });
    var parent = rows[0] && rows[0].parentNode;
    if (!parent) return;
    rows.forEach(function(r){ parent.appendChild(r); });
  }
  document.querySelectorAll('table.sortable').forEach(function(tbl){
    var ths = tbl.querySelectorAll('th');
    ths.forEach(function(th, i){
      th.addEventListener('click', function(){
        var asc = !th.classList.contains('sort-asc');
        ths.forEach(function(h){ h.classList.remove('sort-asc','sort-desc'); });
        th.classList.add(asc ? 'sort-asc' : 'sort-desc');
        sortTable(tbl, i, asc);
      });
    });
  });

  // ---- Scroll-spy: highlight the TOC entry for the section currently in view ----
  var tocLinks = Array.from(document.querySelectorAll('nav.toc a[href^="#"]'));
  if (tocLinks.length && 'IntersectionObserver' in window) {
    var idToLink = {};
    tocLinks.forEach(function(a){ idToLink[a.getAttribute('href').slice(1)] = a; });
    var sections = Object.keys(idToLink)
      .map(function(id){ return document.getElementById(id); })
      .filter(Boolean);
    var obs = new IntersectionObserver(function(entries){
      entries.forEach(function(en){
        if (en.isIntersecting) {
          tocLinks.forEach(function(a){ a.classList.remove('active'); });
          var link = idToLink[en.target.id];
          if (link) link.classList.add('active');
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px', threshold: 0 });
    sections.forEach(function(s){ obs.observe(s); });
  }
})();
</script>
"""


def _slugify(s: str) -> str:
    """Convert a section heading to an HTML id. Strips numbering / punctuation."""
    import re
    s = re.sub(r"^\s*(NEW\s+)?Section\s+\d+\s*[·:\-]?\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*\d+[a-z]?\s*[·:\-]\s*", "", s)
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s-]+", "-", s)
    return s or "section"


def _build_toc(sections: list[tuple[str, str]], cross_link_label: str,
               cross_link_href: str) -> tuple[str, str]:
    """Return (sidebar_html, mobile_details_html).

    `sections` is a list of (id, label) tuples in document order.
    """
    items = "\n".join(
        f"      <li><a href='#{escape(sid)}'>{escape(label)}</a></li>"
        for sid, label in sections
    )
    sidebar = f"""<nav class='toc' aria-label='Section navigation'>
    <h4>On this page</h4>
    <ul>
{items}
    </ul>
    <div class='cross-link'>
      <a href='{escape(cross_link_href)}'>&rarr; {escape(cross_link_label)}</a>
    </div>
  </nav>"""
    mobile_items = "\n".join(
        f"        <li><a href='#{escape(sid)}'>{escape(label)}</a></li>"
        for sid, label in sections
    )
    mobile = f"""<details class='toc-mobile'>
      <summary>Jump to section &#x25BE;</summary>
      <ul>
{mobile_items}
      </ul>
      <p style='margin:.5em 0 0 0;'><a href='{escape(cross_link_href)}'>&rarr; {escape(cross_link_label)}</a></p>
    </details>"""
    return sidebar, mobile


def _reading_time(body_html: str) -> int:
    """Estimate minutes to read the prose @ 250 wpm.

    Counts only the textual content of <p>, <li>, <h2>, <h3>, <h4>, <summary>,
    <dt>, <dd>, and <blockquote> elements — i.e. the prose a reader actually
    reads. Tables, charts, scripts, and styles are excluded because skimmers
    don't read every cell at 250 wpm; including them inflated the estimate.
    """
    import re
    text = re.sub(r"<script[\s\S]*?</script>", " ", body_html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    prose_tags = (r"p|li|h2|h3|h4|summary|dt|dd|blockquote")
    chunks = re.findall(rf"<(?:{prose_tags})\b[^>]*>([\s\S]*?)</(?:{prose_tags})>", text, flags=re.I)
    words = 0
    for c in chunks:
        c = re.sub(r"<[^>]+>", " ", c)
        words += len(re.findall(r"\w+", c))
    minutes = max(1, round(words / 250))
    return minutes


# ---------------------------------------------------------------------------
# Page builders.
# ---------------------------------------------------------------------------

def _build_index_html(ctx: dict) -> str:
    """Landing page: where to buy, overlays, macro context, alt ranking, headline."""

    # Section ids + labels for the TOC (in document order).
    sections = [
        ("executive-summary", "Executive summary"),
        ("how-to-read", "How to read this report"),
        ("where-to-buy", "1. Where might the model want to buy today?"),
        ("personal-overlay", "2. Personal-overlay shortlists (Phase 4)"),
        ("macro-context", "3. Macro context: timing + hedges"),
        ("alt-ranking", "4. Alternative ranking: yield-aware total return"),
        ("headline-results", "5. Headline results"),
    ]
    toc_sidebar, toc_mobile = _build_toc(
        sections,
        cross_link_label="Methodology + deep dive",
        cross_link_href="methods.html",
    )

    # Pull the renumbered overlay HTML (h2 9 -> Section 2).
    overlay_html = _rewrite_overlay_headings(ctx["overlay_section_html"])

    body_inner = f"""
  {toc_mobile}
  <h1>What predicts US residential real-estate returns?</h1>
  <p class='meta' id='meta-line'>arbok study &middot; generated {BUILD_DATE} &middot; all data from free / public sources</p>

  <div class='summary' id='executive-summary'>
    <h3>Executive summary</h3>
    <ul>
      <li><b>Headline decile spread.</b> On the 3-year horizon (post-2012 split), the model's
          top-decile ZIPs averaged <b>{ctx['hd_top']:+.2f}%</b> realized returns vs. the
          bottom decile at <b>{ctx['hd_bot']:+.2f}%</b> &mdash; a <b>{ctx['hd_spread']:+.2f}-point</b>
          out-of-sample spread.</li>
      <li><b>Backtest hit-rate.</b> Replaying month-by-month from 2018-01 through 2021-12
          ({ctx['bt_price_n']} basket-formation months &times; 3y hold), the top-decile basket
          beat the equal-weight universe in <b>{ctx['bt_price_hit']:.1f}%</b> of months at
          roughly <b>+{ctx['bt_price_excess']:.2f}&nbsp;pp/yr</b> alpha.</li>
      <li><b>Timing verdict.</b> The model's current ({ctx['ts_month']}) universe-mean
          predicted fwd_3y annualized return is <b>{ctx['ts_current']*100:+.2f}%</b> &mdash;
          the <b>{ctx['ts_pct']:.0f}th percentile</b> of post-2012 history. The model is currently
          <b>{ctx['ts_verdict']}</b> vs. its historical baseline.</li>
      <li><b>Hedge comparison.</b> US residential RE comes <b>last in an 8-asset cohort</b> on
          median 3y real return (<b>{ctx['zhvi_3y_median']*100:+.1f}%</b>); BTC and gold lead,
          REITs/equities mid-pack. Caveat: ZHVI is unlevered and ignores rent.</li>
      <li><b>Non-obvious finding.</b> Drawdown vs. the trailing-60-month ZHVI peak is the
          #1 mean-abs SHAP feature at fwd_1y &mdash; cross-sectional mean reversion is the
          dominant short-horizon signal, not demographics or schools.</li>
    </ul>
  </div>

  <div class='primer' id='how-to-read'>
    <h3>How to read this report (a 60-second primer)</h3>
    <p>This report uses some specialized terms repeatedly. The shortest possible definitions:</p>
    <dl>
      <dt>ZIP / metro (CBSA)</dt>
      <dd>A ZIP code is a USPS postal area (~22,000 across the US). A "metro" &mdash; Census's
          Core-Based Statistical Area or <b>CBSA</b> &mdash; is a grouping of nearby ZIPs that
          share a labor market. We restrict to the top 100 metros by population.</dd>
      <dt>Forward return (fwd_1y, fwd_3y, fwd_5y)</dt>
      <dd>"fwd_3y" is the price change of a typical home in a ZIP over the next 3 years,
          annualized. The model uses only information available <i>at the start</i> of the
          holding period &mdash; no peeking ahead.</dd>
      <dt>ZHVI / ZORI</dt>
      <dd>Zillow's monthly Home Value Index (the price level we predict) and Observed Rent Index
          (the yield-aware variant of the model).</dd>
      <dt>LightGBM</dt>
      <dd>A gradient-boosted decision-tree model &mdash; the standard ML workhorse for tabular
          data. The headline model in this report; the appendix compares it to ElasticNet and a
          demographics-only hedonic baseline.</dd>
      <dt>SHAP</dt>
      <dd>For each prediction, SHAP attributes a contribution to every input feature.
          Top-SHAP features tell us what the model relies on most. See the methodology page.</dd>
      <dt>Decile spread</dt>
      <dd>The realized return of the top 10% of ZIPs the model ranked highest, minus the
          bottom 10%. The bottom-line "would picking this model's top zips have helped?" number.</dd>
      <dt>Backtest</dt>
      <dd>Replay history. At each past month, we run the model on data available <i>then</i>,
          take its top-decile picks, and check what those ZIPs actually returned 3 years later.</dd>
    </dl>
    <p style='margin-top: 1em;'><a href='methods.html'>Want the full glossary, validation
       methodology, and SHAP-by-horizon? &rarr; Read the deep dive</a></p>
  </div>

  <h2 id='where-to-buy'>1. Where might the model want to buy today?</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> Kansas City, Memphis, Detroit and St. Louis-area ZIPs dominate the
    top picks; predictions cluster around <b>+5&ndash;7%</b> annualized over 3 years with a
    median <b>&plusmn;{ctx['band_width']:.0f} pp</b> uncertainty band per ZIP. Bet sizing
    should reflect the band, not the point estimate.
  </div>
  {ctx['map_link_html']}
  <p>Using the post-2012 LightGBM trained on 3-year forward returns, we scored every zip in the
     <b>top-100</b> US metros at the most recent month with enough feature coverage to make a
     confident prediction. The table below is rolled up to <b>one row per metro (CBSA)</b>,
     showing the metro's principal (largest-population) city, the best-scoring ZIP inside that
     metro, the mean predicted return across the metro's top-5 ZIPs, and how many of its ZIPs
     land in the top 10% of all predictions nationally. <b>Hover over any feature</b> for its
     description; <b>click any column header</b> to sort.</p>
  {ctx['leaderboard_html']}

  <h3>1a. ZIP-level detail (top 20)</h3>
  <p>The same model expanded back to per-ZIP rows for users who want to drill into specific
     neighborhoods rather than cities.</p>
  {ctx['zip_leaderboard_html']}

  <div class='caveat'>
    <b>Important caveats on the leaderboard:</b>
    <ul style='margin-top: .4em; margin-bottom: 0;'>
      <li><b>The p10/p90 bands tell the real story.</b> Top zips show point predictions of
          +5% to +7%, but the 80% prediction interval per zip spans roughly
          [<b>{ctx['band_p10']:+.1f}%</b>, <b>{ctx['band_p90']:+.1f}%</b>] &mdash; a
          <b>{ctx['band_width']:.1f}-point</b> band on median.</li>
      <li>The negative SHAP contribution from <code>fred.real_rate_10y</code> appears for every
          zip &mdash; it's not a zip-specific signal, just "rates are high right now."</li>
      <li>This is a model output, not investment advice. The model is unaware of zoning,
          school districts, walkability, specific listings, or your personal constraints.</li>
    </ul>
  </div>

  <h2 id='personal-overlay'>2. Personal-overlay shortlists (Phase 4)</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> Two pre-filtered shortlists. The owner-occupier picks land mostly in
    DC-metro suburbs ({escape(ctx['owner_occ_concentration'])}); the investor picks spread
    across ~15 mid-tier metros at a median <b>13.7% gross rent yield</b>.
  </div>
  {overlay_html}

  <h2 id='macro-context'>3. Macro context: is now a good time? And how does RE stack up against other hedges?</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> The model predicts the broad RE universe will return
    <b>{ctx['ts_current']*100:+.2f}%/yr</b> over the next 3 years &mdash; the
    <b>{ctx['ts_pct']:.0f}th percentile</b> of post-2012 history. Across 25 years of monthly
    rolling windows, residential RE has trailed every other inflation hedge in our cohort
    (BTC, gold, equities, REITs) before leverage and rent are added back.
  </div>
  <p>Everything above is about <i>where</i> to buy. The two questions below are about
     <i>when</i> and <i>what else</i>: does the model think the broad RE universe is cheap or
     expensive right now, and how has private RE actually fared against the other classic
     inflation hedges (stocks, gold, REITs, BTC) when you put them on the same axis?</p>

  <h3>3a. Is now a good time to buy?</h3>
  <p>For each month from 2014 through the latest in the panel, we score every top-100-metro ZIP
     with the headline post-2012 LightGBM and take the universe mean of the predicted fwd_3y
     annualized return. The model's prediction at month <i>t</i> can be compared to the
     <i>realized</i> universe-mean fwd_3y return three years later (dashed line) for all months
     where t+3y is observable.</p>
  {ctx['timing_chart_html']}
  <p>The model's current ({ctx['ts_month']}) universe-mean predicted fwd_3y annualized return is
     <b>{ctx['ts_current']*100:+.2f}%</b> (median of the per-ZIP distribution that month;
     {ctx['ts_n_zips']:,} ZIPs scored, 80% prediction interval roughly
     [<b>{ctx['ts_p10']*100:+.2f}%</b>, <b>{ctx['ts_p90']*100:+.2f}%</b>]). The historical
     median across all scored months in the post-2012 panel is <b>{ctx['ts_med']*100:+.2f}%</b>
     with an interquartile range of [<b>{ctx['ts_q25']*100:+.2f}%</b>,
     <b>{ctx['ts_q75']*100:+.2f}%</b>]. The current prediction sits at the
     <b>{ctx['ts_pct']:.0f}th percentile</b> of history &mdash; the model is currently
     <b>{ctx['ts_verdict']}</b> vs the historical baseline (n = {ctx['ts_n_hist']} months scored).</p>

  <div class='caveat'>
    <b>Read the timing chart carefully.</b> The model was trained on data &le; 2017-12 with a
    test window of 2018-01 onward, so anything from 2014-2017 is in-sample. From 2018 forward,
    the gap between blue (predicted) and green (realized) is honest out-of-sample error. The
    model under-predicted the 2020-2022 COVID-era surge &mdash; a regime the training set
    didn't cover &mdash; and is now predicting near-zero / mildly negative returns because
    trailing real-rate prints are at decade-plus highs.
  </div>

  <h4>How wide is the model's uncertainty over time?</h4>
  <p>Three separate LightGBM models trained at the 10th, 50th, and 90th conditional quantiles
     produce an 80% prediction band around the median. When the band is narrow, the model is
     confident; when it widens, it is hedging.</p>
  {ctx['fan_html']}

  <h3>3b. Real estate vs other inflation hedges</h3>
  <p>Real estate is rarely held in isolation in a real portfolio &mdash; the question "is now
     a good time to buy a house" is really "is real estate the best use of the marginal dollar
     right now". We put residential RE on the same chart as the S&amp;P 500, broad equities
     (VTI), publicly-traded REITs (IYR / VNQ), gold, and Bitcoin, all in <b>real</b> terms
     (net of CPI). Residential RE is measured <i>unlevered</i> and <i>net of nothing</i>
     (rent, maintenance, property tax, transaction costs all ignored) &mdash; conservative
     for RE since the typical homeowner has both leverage and carrying costs.</p>

  <h4>Historical 3y / 5y annualized real returns (rolling, since 2000)</h4>
  {ctx['hedge_hist_html']}

  <h4>Trailing 3-year real returns ending the most recent month</h4>
  {ctx['hedge_trail_html']}

  <h4>Cumulative real return per asset, 2000-now (log scale)</h4>
  {ctx['hedge_chart_html']}

  <p><b>What beat what.</b> Across the full 2000-now history of monthly rolling windows, the
     median 3-year annualized real return ranking is dominated by Bitcoin
     (<b>{ctx['hedge_winner_3y_median'][1]*100:+.1f}%</b>, on a much shorter post-2014 history
     and with extreme tail risk) and then by gold and broad-equity / S&amp;P 500 total return.
     REITs sit in the <b>+4&ndash;6%</b> range. The Zillow ZHVI top-100 mean &mdash; our
     private-RE benchmark, <i>before</i> leverage and <i>before</i> rent &mdash; comes in at
     <b>{ctx['zhvi_3y_median']*100:+.2f}%</b> median 3y real, ranking
     <b>#{ctx['zhvi_rank']}/{ctx['n_assets']}</b>. The trailing-3y leader as of the latest
     month is <b>{escape(ctx['hedge_winner_trailing'][0])}</b> at
     <b>{ctx['hedge_winner_trailing'][1]*100:+.2f}%</b> real.</p>

  <div class='caveat'>
    <b>Honest caveats on the hedge comparison:</b>
    <ul style='margin-top: .4em; margin-bottom: 0;'>
      <li><b>BTC selection bias.</b> Bitcoin's history starts in 2014, a strictly post-GFC,
          mostly-bull-market regime. Other assets include 2000-2002 and 2008-2009.</li>
      <li><b>REITs are not private RE.</b> IYR/VNQ are leveraged, mark-to-market, and earn
          public-market liquidity premia. Best free TR proxy, but different animal.</li>
      <li><b>ZHVI ignores rent, maintenance, and property tax.</b> The yield-aware
          <code>fwd_3y_total</code> model in Section 4 partially addresses this per ZIP.</li>
      <li><b>No leverage assumed.</b> Real-estate investors typically run 4&ndash;5&times;
          leverage; the equity assets here are unlevered.</li>
    </ul>
  </div>

  <h2 id='alt-ranking'>4. Alternative ranking: yield-aware total return</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> A separate model trained on price + rent carry produces a substantially
    different shortlist &mdash; only ~22% overlap with the price-only top-50 &mdash; tilted
    toward rust-belt high-yield ZIPs (Detroit, Cleveland, Dayton) at a median <b>15.8%</b>
    gross rent yield. Useful for cash-flow investors; smaller universe than the price-only run.
  </div>
  <p>Everything in Section 1 ranks zips by predicted <i>price</i> appreciation. An investor
     who cares about yield (rent flowing in, not just paper appreciation) wants a different
     ranking: a separate LightGBM trained on
     <code>fwd_3y_total = price_return + ZORI rent yield carry</code>. The leaderboard below
     shows the top 30 zips by that target. Caveat: ZORI covers only ~11% of zip-months, so
     this ranking's universe is much smaller (~2K zips vs ~10K for the price-only model).</p>
  {ctx['total_leaderboard_html']}
  <p>The decision-different finding: only <b>11 of 50</b> zips appear in both this and the
     price-only top 50 &mdash; a 22% overlap. Price-only tilts toward sunbelt fringe (Tulsa,
     Kansas City, Memphis, New Orleans) at median rent yield ~12.5%. Total-return is dominated
     by rust-belt high-yield zips (Detroit &times;7, Cleveland &times;5, Dayton, Akron, Toledo,
     Birmingham) at median rent yield ~15.8%. Same model architecture, just yield-aware
     &rarr; very different shortlist.</p>

  <h2 id='headline-results'>5. Headline results</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> The best-performing model for each horizon, on both temporal splits.
    The post-2012 block is the main result; the pre-2008 block is a stress test through the
    housing crisis. A signal that shows up in both is much harder to dismiss as overfitting
    to one regime.
  </div>
  <p>The table below picks the best-performing model for each horizon, reported on <b>both</b>
     temporal splits. The post-2012 block is the main result (larger, more recent test window);
     the pre-2008 block is a stress test through the 2008 housing crisis. A signal that shows
     up in both &mdash; same sign, same order-of-magnitude spread &mdash; is much harder to
     dismiss as overfitting to one regime.</p>
  {ctx['headline_html']}

  <div class='caveat'>
    <b>Why R&sup2; is not reported as the primary metric:</b> traditional R&sup2; is negative
    on both test windows for nearly every model. This is not a model failure &mdash; it's a
    property of the test windows. Both contain regime shifts (housing crash; rate shock + COVID)
    where the mean realized return is very different from the training period. <i>Spatial
    ranking</i> (Spearman + decile spread) is intact and is what matters for a buy-this-zip-
    not-that-zip decision.
  </div>

  <div class='cross-page'>
    <b>Want to see the methodology, validation, and feature attribution?</b>
    &rarr; <a href='methods.html'>Read the deep dive</a> (CV robustness, portfolio backtest,
    SHAP by horizon, price stratification, caveats).
  </div>

  <p class='footnote'>arbok &middot; model-first, personal-overlay layered on top &middot;
     all data free / public sources &middot; code at <code>src/arbok/</code></p>
"""

    minutes = _reading_time(body_inner)
    # Inject reading-time into the meta line.
    body_inner = body_inner.replace(
        f"arbok study &middot; generated {BUILD_DATE} &middot; all data from free / public sources",
        f"~{minutes} min read &middot; generated {BUILD_DATE} &middot; all data from free / public sources",
        1,
    )

    return f"""{_html_head(PAGE_TITLE_BASE)}
<body>
  <noscript>
    <div class='nojs-banner'>
      <b>Interactive charts require JavaScript.</b> The data tables and the
      written analysis in each section will still render without it; the
      hover-tooltips, sortable columns, and Plotly visualisations will not.
    </div>
  </noscript>
  <div class='layout'>
  {toc_sidebar}
  <article>
{body_inner}
  </article>
  </div>
{PAGE_SCRIPTS}
</body>
</html>
"""


def _build_methods_html(ctx: dict) -> str:
    """Methodology + deep-dive page: CV robustness, backtest, SHAP, stratification."""

    sections = [
        ("why-this-study", "Why this study exists"),
        ("setup", "The setup in one minute"),
        ("confidence", "1. Confidence: walk-forward + spatial robustness"),
        ("backtest", "2. Did this actually work? — portfolio backtest"),
        ("decile-spread", "3. Decile spread, every model × horizon × split"),
        ("shap", "4. What did the model learn? — SHAP feature importance"),
        ("price-stratification", "5. Does the signal survive price stratification?"),
        ("caveats", "Caveats and what's not in this run"),
    ]
    toc_sidebar, toc_mobile = _build_toc(
        sections,
        cross_link_label="Back to the results page",
        cross_link_href="index.html",
    )

    body_inner = f"""
  {toc_mobile}
  <h1>Methodology &amp; deep dive</h1>
  <p class='meta' id='meta-line'>arbok study &middot; generated {BUILD_DATE} &middot; all data from free / public sources</p>

  <p><a href='index.html'>&larr; Back to the results landing page</a></p>

  <h2 id='why-this-study'>Why this study exists</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> Most "where should I buy?" advice is qualitative or single-metric.
    This study casts a wide net (140+ candidate predictors) and lets out-of-sample model
    performance + SHAP attribution decide what actually moves forward returns.
  </div>
  <p>Residential real estate is the largest asset class most people will ever own, and the
     conventional wisdom for picking <i>where</i> and <i>when</i> to buy is dominated by
     qualitative heuristics (school districts, "up-and-coming neighborhoods", proximity to
     Whole Foods) or single-metric proxies (population growth, median income). This study
     asks: which of these signals actually predict forward returns, and by how much,
     separated by time horizon &mdash; and are there underweighted predictors that the
     textbook approach misses?</p>
  <p>The framing is deliberately quantitative. We built a 3.73-million-row panel of every zip
     code in the top 100 US metros, monthly, 2000-2026, joined with 140+ candidate predictors
     organized into 11 thematic classes. We trained gradient-boosted models to predict 1-, 3-,
     and 5-year forward home-price returns on two backtest windows: pre-2008 (GFC stress test)
     and post-2012 (rate-shock + COVID).</p>

  <h2 id='setup'>The setup in one minute</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> Predict 1y / 3y / 5y forward home-price returns per ZIP, using only
    information available at the start of the holding period. Two temporal splits (pre-2008,
    post-2012) plus walk-forward and spatial CV.
  </div>

  <h3>What we predict</h3>
  <p>For each zip code at each month, we compute the forward home-price return (Zillow ZHVI
     index) over three horizons &mdash; 1 year, 3 years, 5 years &mdash; annualized for the
     latter two. These are the <i>targets</i> our models try to learn from upstream features.</p>
  <p>The chart below illustrates what those forward windows look like for a single real ZIP.
     The diamond marks an example "prediction date"; the three shaded windows are the 1-, 3-,
     and 5-year forward periods we score the model against. <i>Predict-only-on-what-you-knew-
     at-time-t, score-on-what-actually-happened-by-time-t+h</i> is the discipline that makes
     the backtest honest.</p>
  {ctx['fwd_illus_html']}

  <h3>How we predict it</h3>
  <p>For each (zip, month) row, we join a wide set of features that were <i>knowable at that
     time</i> (with explicit publication-lag adjustment per source &mdash; e.g., HMDA mortgage
     data isn't available until 9 months after the year it describes). We then train two
     families of models per horizon:</p>
  <ul>
    <li><b>Baselines:</b> predict the global mean; predict the within-metro mean; OLS
        regression on Census demographics only ("hedonic"). These are what a real model has
        to beat.</li>
    <li><b>Real models:</b> ElasticNet (regularized linear, interpretable) and LightGBM
        (gradient-boosted trees, captures interactions). Both report Spearman rank correlation
        and decile spread on out-of-sample test data; LightGBM additionally produces SHAP
        feature attributions.</li>
  </ul>

  <h3>How we validate</h3>
  <p>Two temporal splits, both reported in this run:</p>
  <ul>
    <li><b>pre_2008 split:</b> train on data through 2007-12, test on 2008-2013. Stress-tests
        whether the model breaks across the 2008 housing crisis (the "GFC" &mdash; Global
        Financial Crisis).</li>
    <li><b>post_2012 split:</b> train through 2017-12, test on 2018-2024. Stress-tests across
        the Fed's 2022 rate hike cycle (the "rate shock") and the COVID-era demand surge.</li>
  </ul>
  <p>This run additionally reports <b>spatial cross-validation</b> (hold out entire CBSAs) and
     <b>walk-forward CV</b> (rolling 12-month test windows) on top of the two static splits
     &mdash; see Section 1 below.</p>

  <details>
    <summary>14 thematic feature classes (click to expand)</summary>
    <p>The current feature store covers 22 data sources organized into 11 thematic classes,
       producing 140+ modeling features after coverage filtering. Hover over any feature name
       in the SHAP charts below for a one-line description.</p>
    <ul>
      <li><b>Macro / rates:</b> 30Y mortgage, 10Y TIPS, M2, Case-Shiller, lumber, US
          unemployment + their 1/3/12-month deltas (FRED)</li>
      <li><b>Inventory:</b> months-of-supply, days-on-market, price cuts, active/new/pending
          listings per zip (Realtor.com)</li>
      <li><b>Demographics:</b> population by age cohort, household income, education, home
          value, rent, owner-occupancy at ZCTA (Census ACS, 10 vintages 2013-2022); per-capita
          personal income at county (BEA)</li>
      <li><b>Migration:</b> county-to-county AGI flow (IRS SOI)</li>
      <li><b>Supply:</b> building permits at MSA (Census BPS); business establishments +
          employment per zip (Census ZBP)</li>
      <li><b>Jobs:</b> wages + YoY growth at county (BLS QCEW); monthly unemployment rate at
          county (BLS LAUS)</li>
      <li><b>Affordability / cost of living:</b> effective property-tax rate per ZCTA (ACS
          B25103 / B25077); HUD Small-Area Fair Market Rent per ZIP; residential electricity
          price per state (EIA)</li>
      <li><b>Healthcare:</b> CMS Hospital Compare star ratings + bed counts by county; County
          Health Rankings composite + headline measures (life expectancy, premature death,
          smoking, obesity)</li>
      <li><b>Schools:</b> NCES district-year enrollment, pupil-teacher ratio, free-lunch
          share, per-pupil spending &mdash; county-broadcast</li>
      <li><b>Politics:</b> county presidential vote share + winning margin (MIT Election Lab,
          2000-2024)</li>
      <li><b>Climate:</b> trailing-10-year disaster declarations by category (OpenFEMA);
          flood, wildfire, hurricane, heatwave risk scores at tract (FEMA NRI); annual PM2.5
          + ozone air quality (EPA AQS)</li>
      <li><b>Amenities + infrastructure:</b> EV charging stations per zip (DOE AFDC)</li>
      <li><b>Behavioral:</b> monthly Wikipedia pageviews per metro article (search-interest
          proxy)</li>
      <li><b>Derived (no new data):</b> gross rental yield = 12 &times; ZORI / ZHVI; rolling
          24-mo ZHVI volatility; ZHVI drawdown vs trailing-60-mo peak</li>
    </ul>
  </details>

  <details>
    <summary>Pending modules with code ready but waiting on credentials (click to expand)</summary>
    <p>HMDA (tract-level mortgage records &mdash; currently only state-level aggregations),
       FCC BDC broadband, NOAA VIIRS satellite nightlights, Foursquare POIs (Whole Foods /
       coffee / breweries), FBI NIBRS county-year crime rates (needs <code>CDE_API_KEY</code>),
       USAspending federal $ flows.</p>
  </details>

  <h2 id='confidence'>1. Confidence: walk-forward and spatial robustness</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> The headline fwd_3y Spearman survives both stronger CV regimes.
    Walk-forward over {ctx['wf3_n']} rolling 12-month windows gives
    &rho; = <b>{ctx['wf3_mean']:+.3f} &plusmn; {ctx['wf3_std']:.3f}</b>; spatial-CV
    (whole-CBSA holdout) on the within-metro residual gives &rho; =
    <b>{ctx['sp_dem_sp']:+.3f}</b> &mdash; the genuine cross-sectional alpha after stripping
    out the "expensive zips stay expensive" level effect.
  </div>
  <p>A single train/test split can flatter or punish a model by accident. Two stronger CV
     regimes re-run the post-2012 LightGBM to bound how much of the headline number is
     structural vs. noise.</p>

  <h3>1a. Walk-forward CV (rolling-origin)</h3>
  <p>Train through year T&minus;1, test on year T; roll the cutoff from 2018-12 through
     2022-12. Five test years per horizon (fewer for fwd_5y, which would need observability
     beyond the panel's end). The schematic below shows the sliding window: each row is one
     fold, blue is the training range, green is the held-out test year.</p>
  {ctx['wf_schematic_html']}
  {ctx['walkforward_html']}
  <p>For the headline fwd_3y horizon, mean Spearman &rho; = <b>{ctx['wf3_mean']:+.3f}</b> with
     std <b>{ctx['wf3_std']:.3f}</b> across {ctx['wf3_n']} folds &mdash; the
     <b>+{ctx['static_fwd3y_spearman']:.3f}</b> figure from the single static post-2012 split
     survives in expectation, but with meaningful year-to-year noise. 2020 is the variance
     driver, as expected.</p>

  <h3>1b. Spatial CV (whole-CBSA holdout)</h3>
  <p>5-fold CBSA holdout, with train+test both restricted to &le; 2017-12 so the only
     stressor is the spatial axis (no temporal regime shift). Two variants: the <b>raw</b>
     model predicts the price return directly; the <b>demeaned</b> model predicts the
     within-metro residual &mdash; i.e., the model is given each ZIP's return after
     subtracting the average return of every other ZIP in the same metro that month. That
     isolates "can the model rank ZIPs <i>inside</i> a metro?" from the easier question
     "can the model tell expensive metros from cheap ones?".</p>
  <p>The plot below shows the actual partition: every top-100 metro is assigned to exactly
     one of five folds (color). Bubble size = metro population.</p>
  {ctx['sp_fold_html']}
  <h4>Raw fwd_3y</h4>
  {ctx['spatial_raw_html']}
  <h4>Demeaned (within-metro residual) fwd_3y</h4>
  {ctx['spatial_dem_html']}
  <p><b>The headline lesson.</b> The raw model's Spearman &rho; = <b>{ctx['sp_raw_sp']:+.3f}</b>
     with decile spread <b>{ctx['sp_raw_spread']*100:+.2f}%</b> looks spectacular &mdash; but
     most of it is a level effect ("expensive zips stay expensive"). The demeaned model's
     Spearman &rho; = <b>{ctx['sp_dem_sp']:+.3f}</b> with decile spread
     <b>{ctx['sp_dem_spread']*100:+.2f}%</b> is the genuine cross-sectional alpha.</p>

  <h2 id='backtest'>2. Did this actually work? — portfolio backtest</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> Replaying the strategy month-by-month from 2018-01 through 2021-12,
    the top-decile basket beat the equal-weight universe in <b>{ctx['bt_price_hit']:.1f}%</b>
    of months at <b>+{ctx['bt_price_excess']:.2f} pp/yr</b> alpha. The yield-aware variant
    hits <b>{ctx['bt_total_hit']:.1f}%</b> at <b>+{ctx['bt_total_excess']:.2f} pp/yr</b> on a
    smaller yield-observable universe.
  </div>
  <p>The most decision-relevant question. We replay history: at each month from 2018-01
     through 2021-12, we run the model with only the data it could have seen at that time,
     take its top-decile picks (a "basket" of about 1,000 ZIPs), and look up what those ZIPs
     actually returned 3 years later. Compare to the universe mean (equal-weight all ZIPs)
     and to the bottom-decile basket. <b>Alpha</b> is the basket's excess annualized return
     over the universe mean.</p>
  {ctx['backtest_chart_html']}
  {ctx['backtest_table_html']}
  <p><b>The model adds skill.</b> The price-only fwd_3y headline model's top decile beat the
     universe mean in <b>{ctx['bt_price_hit']:.1f}%</b> of months ({ctx['bt_price_n']}/{ctx['bt_price_n']}),
     earning roughly <b>+{ctx['bt_price_excess']:.2f} percentage points per year</b> alpha
     and a <b>+{ctx['bt_price_spread']:.2f} pp/yr</b> top-vs-bottom spread. The yield-aware
     <code>fwd_3y_total</code> model is in another class &mdash; <b>{ctx['bt_total_hit']:.1f}%</b>
     hit rate, <b>+{ctx['bt_total_excess']:.2f} pp/yr</b> alpha &mdash; but its scoring
     universe is much smaller (only ZIPs where Zillow publishes rent data).</p>

  <h3>2a. Per-cohort wealth: $1 in, $? out after 3 years</h3>
  <p>The same backtest expressed as terminal wealth. For each entry month, $1 invested in the
     top-decile basket, the universe, or the bottom-decile basket is compounded forward for
     the full 3-year hold.</p>
  {ctx['eq_curve_html']}

  <h3>2b. Calibration &mdash; is the model a good ranker, a good forecaster, or both?</h3>
  <p>A model can rank ZIPs correctly yet still be mis-calibrated &mdash; predicting +3% when
     realized averages +7%. We bin every test-set prediction into 20 quantile buckets and
     plot mean predicted vs mean realized in each bucket. Points on the dashed line are
     perfectly calibrated.</p>
  {ctx['calib_html']}
  <p>The cloud's <i>slope</i> is the rank quality (steeper = stronger ranking); the cloud's
     <i>shift</i> relative to the diagonal is the calibration error. A model can have a steep
     slope but sit far below the diagonal &mdash; useful for picking but biased in level.
     That pattern is exactly what we'd expect in a regime-shift test window like post-2012,
     and is why R&sup2; is not the headline metric.</p>

  <h2 id='decile-spread'>3. Decile spread, every model &times; horizon &times; split</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> LightGBM wins decisively on the medium and long horizons in the
    post-2012 split; the demographics-only hedonic baseline edges it at fwd_1y &mdash;
    a useful sanity check that the signal lives in the data, not the algorithm.
  </div>
  <p>Each bar is one model on one horizon on one test window. Positive values mean the
     model's top-decile picks beat its bottom-decile picks; negative values mean it
     anti-ranked. The left panel is the pre-2008 split (smaller and noisier); the right is
     post-2012 (the main result).</p>
  {ctx['decile_html']}

  <h3>3a. Realized return by predicted decile (headline horizon &times; split)</h3>
  <p>The chart above collapses each (model, horizon, split) into a single "top minus bottom"
     number. The one below zooms into the headline (post-2012 LightGBM, fwd_3y) and shows the
     realized return of <i>every</i> decile, not just the extremes. A monotone climb from
     decile 1 &rarr; decile 10 is the strong-form claim: the model is ordering ZIPs, not just
     separating the very best from the very worst.</p>
  {ctx['decile_curve_html']}

  <p>LightGBM wins decisively on the medium and long horizons in the post-2012 split. On the
     1-year horizon, the demographics-only hedonic baseline (a simple linear regression using
     only Census ACS variables) actually edges LightGBM &mdash; a useful sanity check that
     when a simple baseline ties a sophisticated model, the signal probably comes from the
     data more than the algorithm.</p>

  <h2 id='shap'>4. What did the model learn? — SHAP feature importance</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> Real rates dominate medium/long-horizon SHAP (10Y TIPS yield is the
    biggest mean-abs feature at fwd_3y and fwd_5y); drawdown vs trailing peak dominates at
    fwd_1y (mean reversion). County Republican vote share and Wikipedia pageviews show up in
    the top 10 at multiple horizons &mdash; identity / attention proxies the textbook
    real-estate model doesn't include.
  </div>
  <p>For each horizon, we compute mean absolute SHAP value per feature on the training
     sample. SHAP decomposes each prediction into per-feature contributions. The bars below
     show the 15 features that the LightGBM model used most heavily.</p>
  <h3>Short horizon (1 year)</h3>
  {ctx['shap_1y_html']}
  <h3>Medium horizon (3 years)</h3>
  {ctx['shap_3y_html']}
  <h3>Long horizon (5 years)</h3>
  {ctx['shap_5y_html']}

  <h3>4a. Per-ZIP attribution &mdash; why this ZIP, not that one?</h3>
  <p>The aggregate SHAP charts above answer "which features does the model rely on, on
     average?" The two waterfalls below answer the more decision-relevant version: <i>for
     this specific ZIP, which features pushed its predicted return up or down, and by how
     much?</i> Green bars push the prediction higher than the model's base rate; red bars
     push it lower.</p>
  <h4>Top-decile example</h4>
  {ctx['shap_top_html']}
  <h4>Bottom-decile example</h4>
  {ctx['shap_bot_html']}
  <p>The pattern that recurs across nearly every top-decile ZIP: a large positive contribution
     from the drawdown feature combined with a uniform negative contribution from the 10Y real
     rate. Bottom-decile ZIPs typically show the opposite.</p>

  <p><b>Four findings worth flagging:</b></p>
  <ul>
    <li><b>Drawdown is the #1 short-horizon predictor.</b> The derived "ZHVI drawdown vs
        trailing 60-month peak" feature carries the largest mean-abs SHAP at fwd_1y. Zips
        that have fallen most relative to their recent peak tend to snap back hardest &mdash;
        a clean cross-sectional mean-reversion signal.</li>
    <li><b>Real rates dominate medium/long by a wide margin.</b> At fwd_3y, 10Y TIPS yield's
        mean-abs SHAP is more than 4&times; the next-largest feature; at fwd_5y, more than
        3&times;. Medium and long-horizon ranking is mostly about macro conditions, not
        zip-specific properties.</li>
    <li><b>New political + behavioral features are earning their keep.</b> County Republican
        vote share (MIT Election Lab) appears in the top 10 for every horizon &mdash; and at
        fwd_5y, both party vote shares appear, so the model is reading the political-identity
        gradient, not taking sides. Wikipedia metro-article pageviews ranks #3 at fwd_3y.</li>
    <li><b>Climate has receded to a short-horizon signal.</b> FEMA disaster counts and NRI
        wildfire risk drop out of the top 10 for fwd_3y and fwd_5y under the expanded feature
        pool, while staying in fwd_1y &mdash; most likely capturing short-term storm-driven
        price dynamics.</li>
  </ul>

  <h2 id='price-stratification'>5. Does the signal survive price stratification?</h2>
  <div class='takeaway'>
    <b>Bottom line:</b> Decile spreads of <b>{ctx['lo_spread']*100:+.2f}%</b> /
    <b>{ctx['mid_spread']*100:+.2f}%</b> / <b>{ctx['hi_spread']*100:+.2f}%</b> across
    &lt;$200K / $200K&ndash;$500K / $500K+ tiers &mdash; the model isn't merely buying cheap
    homes.
  </div>
  <p>A common worry with any model that ranks zip codes is that it has secretly learned a
     price-level proxy: cheap zips mean-revert upward, expensive zips compound more slowly,
     and the "alpha" is just a roundabout way of buying low. To check that, we split the
     post-2012 test set into three ZHVI tiers and re-compute the LightGBM
     <code>fwd_3y</code> decile spread <i>within</i> each tier.</p>
  {ctx['price_strat_html']}
  <p><b>Interpretation.</b> The decile spread is <b>{ctx['lo_spread']*100:+.2f}%</b> for
     sub-$200K homes, <b>{ctx['mid_spread']*100:+.2f}%</b> in the $200K&ndash;$500K middle,
     and <b>{ctx['hi_spread']*100:+.2f}%</b> for $500K+ homes &mdash; all positive, all
     out-of-sample. The two cheaper tiers carry essentially the same spread, and the $500K+
     tier roughly halves but stays positive. This is the pattern you'd expect if the model
     has real cross-sectional signal at every price point and luxury markets simply have
     flatter forward returns &mdash; not the pattern you'd expect if the model were secretly
     a price-mean-reversion proxy.</p>

  <h2 id='caveats'>Caveats and what's not in this run</h2>
  <ul>
    <li><b>HMDA mortgage records are in the feature store but didn't reach the trained
        model.</b> Tract-year origination counts + institutional-share are loaded at ~12%
        panel coverage, but the data only spans 2020 / 2022 / 2023. The post-2012 split's
        train window ends at 2017-12 &mdash; entirely pre-HMDA &mdash; so prep's coverage
        filter drops every HMDA column from X.</li>
    <li><b>Four data sources still pending.</b> FCC broadband, NOAA VIIRS satellite
        nightlights, Foursquare POIs, and FBI NIBRS crime aggregates (needs
        <code>CDE_API_KEY</code>) are wired into the assembler but blocked on credentials.</li>
    <li><b>Owner-occupier overlay is geographically narrow.</b> {escape(ctx['owner_occ_concentration'])}
        because the filter chain collapses to wealthy suburbs of one dominant metro.</li>
    <li><b>Total-return modeling now done but on a smaller universe.</b> ZORI rent index
        covers only ~11% of zip-months, so <code>fwd_3y_total</code> trains on ~215K rows vs
        ~1.98M for price-only.</li>
    <li><b>ElasticNet regressed under the new QuantileTransformer scaling.</b> The Wave-1 fix
        solved an RMSE blowup but flipped fwd_3y Spearman to negative. LightGBM is unaffected
        and remains the headline model.</li>
    <li><b>Pre-2008 coverage is thin.</b> Realtor inventory (2016+), QCEW (2010+), ACS
        (2013+), BPS (2011+), IRS-SOI (2011+) all start post-GFC. The pre_2008 results in
        the headline table train on 27 features vs 41 post-2012.</li>
    <li><b>fwd_5y walk-forward CV has 3 folds, not 5</b> &mdash; the 2021-12 and 2022-12
        cutoffs would need fwd_5y observability through 2026-12 / 2027-12, beyond the
        panel's end.</li>
  </ul>

  <div class='cross-page'>
    <b>Back to the results?</b> &rarr; <a href='index.html'>Where to buy, overlay shortlists,
    macro context, alternative ranking, and the headline summary table</a>.
  </div>

  <p class='footnote'>arbok &middot; model-first, personal-overlay layered on top &middot;
     all data free / public sources &middot; code at <code>src/arbok/</code></p>
"""

    minutes = _reading_time(body_inner)
    body_inner = body_inner.replace(
        f"arbok study &middot; generated {BUILD_DATE} &middot; all data from free / public sources",
        f"~{minutes} min read &middot; generated {BUILD_DATE} &middot; all data from free / public sources",
        1,
    )

    return f"""{_html_head(PAGE_TITLE_BASE + " — methodology")}
<body>
  <noscript>
    <div class='nojs-banner'>
      <b>Interactive charts require JavaScript.</b> The data tables and the
      written analysis in each section will still render without it; the
      hover-tooltips, sortable columns, and Plotly visualisations will not.
    </div>
  </noscript>
  <div class='layout'>
  {toc_sidebar}
  <article>
{body_inner}
  </article>
  </div>
{PAGE_SCRIPTS}
</body>
</html>
"""


def _rewrite_overlay_headings(html: str) -> str:
    """The overlay artifact still emits '9 ·' / '9a ·' / '9b ·' / '9c ·' headings.
    Rewrite to fit the new index-page numbering (Section 2)."""
    import re
    if not html:
        return html
    # Replace top-level "9 · Personal-overlay shortlists ..." with no number (the
    # parent h2 on the page already carries the section heading).
    html = re.sub(
        r"<h2>9\s*[·‧.:\-]\s*Personal-overlay shortlists[^<]*</h2>",
        "",
        html,
    )
    # Renumber 9a → 2a, 9b → 2b, 9c → 2c.
    html = re.sub(r"(<h3>)9([abc])\s*[·‧.:\-]\s*", r"\g<1>2\g<2>. ", html)
    return html


def main() -> None:
    """Compute all data once; render index.html + methods.html."""
    print("Building charts…")
    decile = chart_decile_spread()
    shap_figs = [chart_shap_for(h) for h in HORIZONS]
    leaderboard_html_raw = chart_city_leaderboard_top_n(30)
    zip_leaderboard_html_raw = chart_leaderboard_top_n(20)
    total_leaderboard_html_raw = chart_total_leaderboard_top_n(n=30)
    # Inject the `sortable` class on all three leaderboard tables.
    def _make_sortable(s: str) -> str:
        return s.replace("<table class='leaderboard'>",
                          "<table class='leaderboard sortable'>", 1)
    leaderboard_html = _make_sortable(leaderboard_html_raw)
    zip_leaderboard_html = _make_sortable(zip_leaderboard_html_raw)
    total_leaderboard_html = _make_sortable(total_leaderboard_html_raw)

    headline_html = chart_headline_table()
    price_strat_html = chart_price_stratification_table()

    strat_df = price_stratified_decile_spread()
    strat_lookup = {r["tier"]: r["spread"] for _, r in strat_df.iterrows()}
    lo_spread = strat_lookup.get("<$200K", float("nan"))
    mid_spread = strat_lookup.get("$200K–$500K", float("nan"))
    hi_spread = strat_lookup.get("$500K+", float("nan"))

    walkforward_html = chart_walkforward_table()
    if WALKFORWARD is not None:
        sub_wf = WALKFORWARD[
            (WALKFORWARD["split"] == "post_2012")
            & (WALKFORWARD["model"] == "lightgbm")
            & (WALKFORWARD["demean"] == False)  # noqa: E712
        ]
        wf_summary = _summarize_cv(sub_wf).set_index("horizon")
        wf3_mean = float(wf_summary.loc["fwd_3y", "mean_spearman"]) if "fwd_3y" in wf_summary.index else float("nan")
        wf3_std = float(wf_summary.loc["fwd_3y", "std_spearman"]) if "fwd_3y" in wf_summary.index else float("nan")
        wf3_n = int(wf_summary.loc["fwd_3y", "n_folds"]) if "fwd_3y" in wf_summary.index else 0
    else:
        wf3_mean = wf3_std = float("nan")
        wf3_n = 0

    spatial_raw_html, spatial_dem_html, spatial_quotes = chart_spatialcv_tables()
    sp_raw_sp = spatial_quotes.get("raw_spearman", float("nan"))
    sp_dem_sp = spatial_quotes.get("dem_spearman", float("nan"))
    sp_raw_spread = spatial_quotes.get("raw_spread", float("nan"))
    sp_dem_spread = spatial_quotes.get("dem_spread", float("nan"))

    post3 = (
        RESULTS[(RESULTS["split"] == "post_2012") & (RESULTS["horizon"] == "fwd_3y")]
        .sort_values("decile_spread", ascending=False)
        .head(1)
    )
    if not post3.empty:
        hd_spread = float(post3["decile_spread"].iloc[0]) * 100
        hd_top = float(post3["top_decile_mean"].iloc[0]) * 100
        hd_bot = float(post3["bot_decile_mean"].iloc[0]) * 100
    else:
        hd_spread = hd_top = hd_bot = float("nan")

    lgbm3 = RESULTS[
        (RESULTS["split"] == "post_2012") & (RESULTS["horizon"] == "fwd_3y")
        & (RESULTS["model"] == "lightgbm")
    ]
    static_fwd3y_spearman = float(lgbm3["spearman"].iloc[0]) if not lgbm3.empty else float("nan")

    band = leaderboard_band_stats()
    band_p10 = band["median_p10"] * 100
    band_p90 = band["median_p90"] * 100
    band_width = band["median_band"] * 100

    map_path = PROCESSED / "zip_decile_map.html"
    if map_path.exists():
        map_link_html = (
            "<p>Hover any zip for ZIP code, metro, ZHVI, predicted fwd_3y return, and decile rank. "
            "Drag to pan; scroll to zoom. "
            f"<a href='zip_decile_map.html?v={BUILD_DATE}' target='_blank'>Open the map full-screen &#x2197;</a></p>"
            f"<iframe src='zip_decile_map.html?v={BUILD_DATE}' "
            "style='width:100%; height:600px; border:1px solid #ccc; border-radius:4px;' "
            "loading='lazy' title='ZIP decile map'></iframe>"
        )
    else:
        map_link_html = "<p class='caveat'>(Map render pending.)</p>"

    # Backtest charts/tables: include_js is False — both pages' first chart
    # (forward-return-illustration on methods, timing-chart on index) will be
    # the inlining anchor for that page's Plotly bundle.
    backtest_chart_fig = chart_backtest_chart()
    backtest_chart_html = _div(
        backtest_chart_fig, include_js=False,
        aria_label="Time-series line chart: top-decile basket vs universe mean vs bottom-decile basket realized fwd_3y returns by score month",
    ) if backtest_chart_fig is not None else "<p><i>Backtest not yet built.</i></p>"
    backtest_table_html = chart_backtest_summary_table()
    bt = backtest_summary_stats()
    bt_price_hit = bt["price"]["hit"]
    bt_price_excess = bt["price"]["excess"]
    bt_price_spread = bt["price"]["spread"]
    bt_price_n = bt["price"]["months"]
    bt_total_hit = bt["total"]["hit"]
    bt_total_excess = bt["total"]["excess"]
    bt_total_spread = bt["total"]["spread"]

    overlay_section_html = OVERLAY_HTML if OVERLAY_HTML else "<p><i>Personal overlay not yet built.</i></p>"
    owner_occ_concentration = owner_occupier_concentration_stat()

    # Macro context: timing chart sits on index — so the timing chart is the
    # first Plotly chart there and carries include_js=True. Methods page's
    # first Plotly chart is the fwd_illus_fig.
    timing_fig = chart_timing_universe_mean()
    if timing_fig is not None:
        timing_chart_html = _div(
            timing_fig, include_js=True,
            aria_label="Time series line chart: universe-mean predicted fwd_3y vs subsequent realized fwd_3y, post-2012 LightGBM",
        )
    else:
        timing_chart_html = ("<p class='caveat'>Timing chart not built yet &mdash; "
                             "run <code>python scripts/macro_context.py</code>.</p>")
    timing_stats = timing_summary_stats()
    if timing_stats:
        ts_current = timing_stats["current"]
        ts_month = timing_stats["current_month"].strftime("%Y-%m")
        ts_p10 = timing_stats["p10"]
        ts_p90 = timing_stats["p90"]
        ts_med = timing_stats["median_hist"]
        ts_q25 = timing_stats["q25_hist"]
        ts_q75 = timing_stats["q75_hist"]
        ts_pct = timing_stats["percentile"]
        ts_verdict = timing_stats["verdict"]
        ts_n_zips = timing_stats["n_zips"]
        _timing_df = _load_timing()
        ts_n_hist = int(_timing_df.shape[0]) if _timing_df is not None else 0
    else:
        ts_current = ts_p10 = ts_p90 = ts_med = ts_q25 = ts_q75 = ts_pct = float("nan")
        ts_month = "—"
        ts_verdict = "n/a"
        ts_n_zips = 0
        ts_n_hist = 0

    hedge_hist_html = hedge_historical_table()
    hedge_trail_html = hedge_trailing_table()
    hedge_fig = chart_hedge_comparison()
    hedge_chart_html = _div(
        hedge_fig, include_js=False,
        aria_label="Log-scale cumulative real return per asset since 2000: residential RE vs S&P 500, gold, REITs, and Bitcoin",
    ) if hedge_fig is not None else "<p class='caveat'>Hedge chart not built yet.</p>"
    hedge_summary = hedge_summary_for_copy()
    hedge_winner_3y_median = (
        hedge_summary["medians"][0] if hedge_summary.get("medians") else ("n/a", float("nan"))
    )
    hedge_winner_trailing = (
        hedge_summary["trailing"][0] if hedge_summary.get("trailing") else ("n/a", float("nan"))
    )
    if hedge_summary.get("medians"):
        labels = [a for a, _ in hedge_summary["medians"]]
        zhvi_label = HEDGE_LABELS.get("RE_ZHVI_top100", "RE_ZHVI_top100")
        try:
            zhvi_rank = labels.index(zhvi_label) + 1
            zhvi_3y_median = hedge_summary["medians"][zhvi_rank - 1][1]
        except ValueError:
            zhvi_rank = -1
            zhvi_3y_median = float("nan")
    else:
        zhvi_rank = -1
        zhvi_3y_median = float("nan")
    n_assets = len(hedge_summary.get("medians", [])) or 0

    # Methods-page explanatory visuals.
    print("  · forward-return illustration")
    fwd_illus_fig = chart_forward_return_illustration()
    # fwd_illus is the FIRST Plotly chart on methods.html — inline the bundle.
    if fwd_illus_fig is not None:
        fwd_illus_html = _div(
            fwd_illus_fig, include_js=True,
            aria_label="Illustrative ZIP ZHVI line chart with shaded fwd_1y, fwd_3y, and fwd_5y forward windows from a single anchor date",
        )
        methods_js_inlined = True
    else:
        fwd_illus_html = "<p class='caveat'>Forward-return illustration unavailable.</p>"
        methods_js_inlined = False

    print("  · walk-forward schematic")
    wf_schematic_html = _div(
        chart_walkforward_schematic(), include_js=not methods_js_inlined,
        aria_label="Schematic of walk-forward cross-validation",
    )
    methods_js_inlined = True

    print("  · spatial-CV fold matrix")
    sp_fold_fig = chart_spatial_fold_matrix()
    sp_fold_html = _div(
        sp_fold_fig, include_js=False,
        aria_label="Scatter of top-100 CBSAs colored by spatial-CV fold assignment",
    ) if sp_fold_fig is not None else ""

    print("  · equity curve")
    eq_curve_fig = chart_equity_curve()
    eq_curve_html = _div(
        eq_curve_fig, include_js=False,
        aria_label="Per-cohort terminal wealth from $1 invested in the top-decile basket, universe, and bottom-decile basket held for 3 years",
    ) if eq_curve_fig is not None else ""

    print("  · calibration scatter")
    calib_html = _div(
        chart_calibration_scatter(), include_js=False,
        aria_label="Calibration scatter: mean predicted vs realized fwd_3y across 20 quantile bins",
    )

    print("  · decile curve")
    decile_curve_html = _div(
        chart_decile_curve(), include_js=False,
        aria_label="Realized fwd_3y annualized return by predicted decile",
    )

    print("  · SHAP waterfalls (top + bot example ZIPs)")
    shap_top_fig, shap_bot_fig = chart_shap_waterfall()
    shap_top_html = _div(
        shap_top_fig, include_js=False,
        aria_label="SHAP waterfall chart for the highest-predicted ZIP at the latest scoring month",
    ) if shap_top_fig is not None else ""
    shap_bot_html = _div(
        shap_bot_fig, include_js=False,
        aria_label="SHAP waterfall chart for the lowest-predicted ZIP at the latest scoring month",
    ) if shap_bot_fig is not None else ""

    # decile + SHAP figures for methods.
    decile_html = _div(
        decile, include_js=False,
        aria_label="Decile-spread bar chart by horizon and split, grouped by model",
    )
    shap_1y_html = _div(
        shap_figs[0], include_js=False,
        aria_label="Top-15 SHAP feature importance bar chart for fwd_1y at the post_2012 split",
    )
    shap_3y_html = _div(
        shap_figs[1], include_js=False,
        aria_label="Top-15 SHAP feature importance bar chart for fwd_3y at the post_2012 split",
    )
    shap_5y_html = _div(
        shap_figs[2], include_js=False,
        aria_label="Top-15 SHAP feature importance bar chart for fwd_5y at the post_2012 split",
    )

    # Quantile fan: appears in index (after timing chart) — include_js=False
    # because timing_chart_html already inlined.
    fan_fig = chart_quantile_fan()
    fan_html = _div(
        fan_fig, include_js=False,
        aria_label="Universe-mean fwd_3y median forecast with 80% prediction band from quantile LightGBM models",
    ) if fan_fig is not None else ""

    ctx = dict(
        # Headline / decile-spread block
        hd_spread=hd_spread, hd_top=hd_top, hd_bot=hd_bot,
        static_fwd3y_spearman=static_fwd3y_spearman,
        headline_html=headline_html,
        decile_html=decile_html,
        decile_curve_html=decile_curve_html,
        # Backtest
        bt_price_hit=bt_price_hit, bt_price_excess=bt_price_excess,
        bt_price_spread=bt_price_spread, bt_price_n=bt_price_n,
        bt_total_hit=bt_total_hit, bt_total_excess=bt_total_excess,
        bt_total_spread=bt_total_spread,
        backtest_chart_html=backtest_chart_html,
        backtest_table_html=backtest_table_html,
        eq_curve_html=eq_curve_html,
        calib_html=calib_html,
        # CV
        wf3_mean=wf3_mean, wf3_std=wf3_std, wf3_n=wf3_n,
        walkforward_html=walkforward_html,
        wf_schematic_html=wf_schematic_html,
        sp_raw_sp=sp_raw_sp, sp_dem_sp=sp_dem_sp,
        sp_raw_spread=sp_raw_spread, sp_dem_spread=sp_dem_spread,
        spatial_raw_html=spatial_raw_html,
        spatial_dem_html=spatial_dem_html,
        sp_fold_html=sp_fold_html,
        # SHAP
        shap_1y_html=shap_1y_html,
        shap_3y_html=shap_3y_html,
        shap_5y_html=shap_5y_html,
        shap_top_html=shap_top_html,
        shap_bot_html=shap_bot_html,
        # Stratification
        lo_spread=lo_spread, mid_spread=mid_spread, hi_spread=hi_spread,
        price_strat_html=price_strat_html,
        # Leaderboards
        leaderboard_html=leaderboard_html,
        zip_leaderboard_html=zip_leaderboard_html,
        total_leaderboard_html=total_leaderboard_html,
        # Bands
        band_p10=band_p10, band_p90=band_p90, band_width=band_width,
        # Map
        map_link_html=map_link_html,
        # Overlay
        overlay_section_html=overlay_section_html,
        owner_occ_concentration=owner_occ_concentration,
        # Macro context
        timing_chart_html=timing_chart_html,
        fan_html=fan_html,
        ts_current=ts_current, ts_month=ts_month, ts_p10=ts_p10, ts_p90=ts_p90,
        ts_med=ts_med, ts_q25=ts_q25, ts_q75=ts_q75, ts_pct=ts_pct,
        ts_verdict=ts_verdict, ts_n_zips=ts_n_zips, ts_n_hist=ts_n_hist,
        # Hedges
        hedge_hist_html=hedge_hist_html,
        hedge_trail_html=hedge_trail_html,
        hedge_chart_html=hedge_chart_html,
        hedge_winner_3y_median=hedge_winner_3y_median,
        hedge_winner_trailing=hedge_winner_trailing,
        zhvi_rank=zhvi_rank, zhvi_3y_median=zhvi_3y_median, n_assets=n_assets,
        # Methods-only
        fwd_illus_html=fwd_illus_html,
    )

    # Render both pages.
    index_html = _build_index_html(ctx)
    methods_html = _build_methods_html(ctx)

    out_index = PROCESSED / "index.html"
    out_methods = PROCESSED / "methods.html"
    out_index.write_text(index_html)
    out_methods.write_text(methods_html)
    print(f"Wrote {out_index}  ({out_index.stat().st_size/1024:.1f} KB)")
    print(f"Wrote {out_methods}  ({out_methods.stat().st_size/1024:.1f} KB)")

    # Remove the stale single-page report.html (the workflow used to copy it
    # to _site/index.html; we now copy index.html + methods.html directly).
    old = PROCESSED / "report.html"
    if old.exists():
        old.unlink()
        print(f"Removed legacy {old}")


if __name__ == "__main__":
    main()
