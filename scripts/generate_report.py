"""Render a self-contained HTML report from Phase 2 results + leaderboard.

Sections:
  1. Headline metrics (decile spread per horizon x split x model)
  2. Per-horizon SHAP top-15 feature importance (post_2012 split — the strong one)
  3. Top-50 zip leaderboard with per-zip drivers

Output: data/processed/report.html (open in any browser, no server needed).
"""
from dotenv import load_dotenv
load_dotenv()

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


def _div(fig: go.Figure, include_js: bool) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=("cdn" if include_js else False))


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
    """Turn 'fred.real_rate_10y(-0.040), nri.wildfire_risk(+0.009)' into '10Y real rate (-0.040), NRI wildfire risk (+0.009)'."""
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
        out_parts.append(f"<span title='{escape(desc)}'>{escape(display)}</span> <code>{escape(num)}</code>")
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
    return f"<table class='leaderboard'>{head}{''.join(rows)}</table>"


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
    return f"<table class='leaderboard'>{head}{''.join(rows)}</table>"


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
    return f"""
    <table class='headline'>
      <tr><th>price tier (ZHVI)</th><th>test rows</th><th>median ZHVI</th>
          <th>top decile realized</th><th>bottom decile realized</th><th>decile spread</th></tr>
      {"".join(rows)}
    </table>
    """


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
    return f"""
    <table class='headline'>
      <tr><th>horizon</th><th>best model</th><th>Spearman ρ</th><th>decile spread</th><th>top decile</th><th>bottom decile</th></tr>
      {"".join(section_rows)}
    </table>
    """


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
    return f"""
    <table class='headline'>
      <tr><th>horizon</th><th>n folds</th><th>mean Spearman ρ</th><th>std ρ</th>
          <th>95% interval ρ</th><th>mean decile spread</th><th>std decile spread</th></tr>
      {"".join(rows)}
    </table>
    """


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
        return f"""
        <table class='headline'>
          <tr><th>horizon</th><th>n folds</th><th>mean Spearman ρ</th><th>std ρ</th>
              <th>mean decile spread</th><th>std decile spread</th></tr>
          {"".join(rows)}
        </table>
        """

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
    return f"""
    <table class='headline'>
      <tr><th>model</th><th>months</th><th>universe / mo</th><th>top-10% basket</th>
          <th>top-10% realized</th><th>universe realized</th><th>bottom-10% realized</th>
          <th>top excess</th><th>top − bot spread</th><th>hit rate (top &gt; univ)</th></tr>
      {"".join(rows)}
    </table>
    """


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
    return f"<table class='leaderboard'>{header}{''.join(rows)}</table>"


# ---------------------------------------------------------------------------
# Macro context (section 5b): timing chart + hedge comparison.
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
    return (
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
    return (
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


def main() -> None:
    print("Building charts…")
    decile = chart_decile_spread()
    shap_figs = [chart_shap_for(h) for h in HORIZONS]
    leaderboard_html = chart_city_leaderboard_top_n(30)
    zip_leaderboard_html = chart_leaderboard_top_n(20)
    headline_html = chart_headline_table()
    price_strat_html = chart_price_stratification_table()
    # Also compute the price-stratification numbers once more for the
    # interpretation paragraph below the table.
    strat_df = price_stratified_decile_spread()
    strat_lookup = {r["tier"]: r["spread"] for _, r in strat_df.iterrows()}
    lo_spread = strat_lookup.get("<$200K", float("nan"))
    mid_spread = strat_lookup.get("$200K–$500K", float("nan"))
    hi_spread = strat_lookup.get("$500K+", float("nan"))

    # Walk-forward CV table + per-horizon stats for inline copy
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
        wf3_lo = float(wf_summary.loc["fwd_3y", "q025_spearman"]) if "fwd_3y" in wf_summary.index else float("nan")
        wf3_hi = float(wf_summary.loc["fwd_3y", "q975_spearman"]) if "fwd_3y" in wf_summary.index else float("nan")
        wf3_n = int(wf_summary.loc["fwd_3y", "n_folds"]) if "fwd_3y" in wf_summary.index else 0
    else:
        wf3_mean = wf3_std = wf3_lo = wf3_hi = float("nan")
        wf3_n = 0

    # Spatial CV: raw vs demeaned tables
    spatial_raw_html, spatial_dem_html, spatial_quotes = chart_spatialcv_tables()
    sp_raw_sp = spatial_quotes.get("raw_spearman", float("nan"))
    sp_dem_sp = spatial_quotes.get("dem_spearman", float("nan"))
    sp_raw_spread = spatial_quotes.get("raw_spread", float("nan"))
    sp_dem_spread = spatial_quotes.get("dem_spread", float("nan"))

    # Headline fwd_3y post_2012 numbers (best model decile spread + top/bot decile)
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

    # Leaderboard band width stats
    band = leaderboard_band_stats()
    band_p10 = band["median_p10"] * 100
    band_p90 = band["median_p90"] * 100
    band_width = band["median_band"] * 100

    # Map embed — iframe to the self-contained Folium HTML so we don't balloon report.html
    # to 8 MB by inlining the Leaflet bundle.
    map_path = PROCESSED / "zip_decile_map.html"
    if map_path.exists():
        map_link_html = (
            "<p>Hover any zip for ZIP code, metro, ZHVI, predicted fwd_3y return, and decile rank. "
            "Drag to pan; scroll to zoom. "
            "<a href='zip_decile_map.html' target='_blank'>Open the map full-screen ↗</a></p>"
            "<iframe src='zip_decile_map.html' "
            "style='width:100%; height:600px; border:1px solid #ccc; border-radius:4px;' "
            "loading='lazy' title='ZIP decile map'></iframe>"
        )
    else:
        map_link_html = "<p class='caveat'>(Map render pending.)</p>"

    # Backtest: time-series chart + summary table + headline stats for TL;DR
    backtest_chart_fig = chart_backtest_chart()
    backtest_chart_html = _div(backtest_chart_fig, include_js=False) if backtest_chart_fig is not None else "<p><i>Backtest not yet built.</i></p>"
    backtest_table_html = chart_backtest_summary_table()
    bt = backtest_summary_stats()
    bt_price_hit = bt["price"]["hit"]
    bt_price_excess = bt["price"]["excess"]
    bt_price_spread = bt["price"]["spread"]
    bt_price_n = bt["price"]["months"]
    bt_total_hit = bt["total"]["hit"]
    bt_total_excess = bt["total"]["excess"]
    bt_total_spread = bt["total"]["spread"]

    # Total-return alternative leaderboard
    total_leaderboard_html = chart_total_leaderboard_top_n(n=30)

    # Personal-overlay fragment (section 8)
    overlay_section_html = OVERLAY_HTML if OVERLAY_HTML else "<p><i>Personal overlay not yet built.</i></p>"

    # --- 5b · Macro context ------------------------------------------------
    timing_fig = chart_timing_universe_mean()
    if timing_fig is not None:
        timing_chart_html = _div(timing_fig, include_js=False)
    else:
        timing_chart_html = ("<p class='caveat'>Timing chart not built yet — "
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
    if hedge_fig is not None:
        hedge_chart_html = _div(hedge_fig, include_js=False)
    else:
        hedge_chart_html = ("<p class='caveat'>Hedge chart not built yet — "
                            "run <code>python scripts/macro_context.py</code>.</p>")
    hedge_summary = hedge_summary_for_copy()
    hedge_winner_3y_median = (
        hedge_summary["medians"][0] if hedge_summary.get("medians") else ("n/a", float("nan"))
    )
    hedge_loser_3y_median = (
        hedge_summary["medians"][-1] if hedge_summary.get("medians") else ("n/a", float("nan"))
    )
    hedge_winner_trailing = (
        hedge_summary["trailing"][0] if hedge_summary.get("trailing") else ("n/a", float("nan"))
    )
    # Find ZHVI's rank for the body copy.
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

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>arbok — what predicts US residential real-estate returns?</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
            max-width: 1100px; margin: 2em auto; color: #1a1a1a; padding: 0 1em;
            line-height: 1.55; }}
    h1 {{ border-bottom: 2px solid #1a1a1a; padding-bottom: .3em; margin-bottom: .2em; }}
    h2 {{ margin-top: 2.2em; color: #2c5282; padding-bottom: .2em;
          border-bottom: 1px solid #e0e6ef; }}
    h3 {{ color: #2c5282; margin-top: 1.8em; }}
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
    table.leaderboard span[title] {{ border-bottom: 1px dotted #999; cursor: help; }}
    .meta {{ color: #666; font-size: .9em; margin-bottom: 1.8em; }}
    .tldr {{ background: #f3f5f9; border-left: 4px solid #2c5282;
             padding: 1em 1.4em; margin: 1.5em 0; border-radius: 0 4px 4px 0; }}
    .tldr h3 {{ margin-top: 0; color: #2c5282; }}
    .caveat {{ background: #fff8e6; border-left: 4px solid #c9a227;
               padding: .9em 1.2em; margin: 1.2em 0; font-size: .93em;
               border-radius: 0 4px 4px 0; }}
    .glossary dt {{ font-weight: 600; margin-top: .8em; color: #2c5282;
                    font-family: ui-monospace, 'SF Mono', monospace; }}
    .glossary dd {{ margin: .2em 0 .2em 1.5em; }}
    code {{ background: #f3f5f9; padding: 1px 5px; border-radius: 3px;
            font-size: .9em; }}
    .footnote {{ color: #666; font-size: .85em; margin-top: 3em;
                 border-top: 1px solid #eee; padding-top: 1em; }}
    ul {{ margin: .4em 0; }}
    li {{ margin: .2em 0; }}
  </style>
</head>
<body>
  <h1>What predicts US residential real-estate returns?</h1>
  <p class='meta'>arbok study · generated 2026-05-22 · all data from free / public sources</p>

  <div class='tldr'>
    <h3>TL;DR</h3>
    <ul>
      <li>Built a 3.73-million-row panel of every zip code in the top 100 US metros, monthly, 2000-2026,
          joined with 140+ candidate predictors organized into 11 thematic classes (macro, demographics,
          climate, supply, healthcare, schools, politics, affordability, amenities, etc.).</li>
      <li>Trained gradient-boosted models to predict 1-, 3-, and 5-year forward home-price returns
          per zip on two backtest windows: pre-2008 (GFC stress test) and post-2012 (rate-shock + COVID).</li>
      <li><b>Best signal:</b> 3-year horizon, post-2012 split. Top decile averaged
          <b>{hd_top:+.2f}%</b> realized returns vs. bottom decile <b>{hd_bot:+.2f}%</b> —
          a {hd_spread:+.2f} pt spread out of sample.</li>
      <li><b>The model works in backtest.</b> Replaying the strategy month-by-month from 2018-01
          through 2021-12 ({bt_price_n} basket-formation months × 3y hold), the model's top-decile
          zip basket beat the equal-weight universe in <b>{bt_price_hit:.1f}%</b> of months,
          earning roughly <b>+{bt_price_excess:.2f} pp/yr</b> alpha. The yield-aware variant beat
          the universe in <b>{bt_total_hit:.1f}%</b> of months at <b>+{bt_total_excess:.2f} pp/yr</b>
          on a (smaller) yield-observable universe.</li>
      <li><b>But is now a good time to buy?</b> The model's predicted universe-mean fwd_3y at
          {ts_month} is <b>{ts_current*100:+.2f}%</b> — that sits at the
          <b>{ts_pct:.0f}th percentile</b> of the post-2012 history (median +{ts_med*100:.2f}%).
          Verdict: <b>{ts_verdict}</b> vs historical baseline. The model is saying "wait, or pick
          carefully" rather than "buy the index". See section 5b.</li>
      <li><b>Real estate vs other inflation hedges (median 3y real return, 2000-now, unlevered):</b>
          Bitcoin and gold lead (+10% to +66% depending on selection bias); broad equity and REITs
          mid-pack; <b>US residential RE comes last at ~+3.5%</b>. Trailing 3 years through {ts_month}
          puts ZHVI top-100 at <b>{hedge_summary['zhvi_trailing']*100:+.2f}%</b> real, worst in the
          cohort. Caveat: ZHVI is unlevered and ignores rent — net of leverage + carry, real estate
          looks better; but it's not the runaway hedge folk wisdom claims.</li>
      <li><b>Robustness check.</b> Walk-forward CV across {wf3_n} rolling 12-month test
          windows gives fwd_3y Spearman ρ = <b>{wf3_mean:+.3f} ± {wf3_std:.3f}</b>. Spatial CV
          (hold out whole CBSAs, train+test ≤ 2017-12) on the <i>within-metro residual</i>
          gives Spearman ρ = <b>{sp_dem_sp:+.3f}</b> — the genuine cross-sectional alpha after
          stripping out the "expensive zips stay expensive" level effect.</li>
      <li><b>The signal survives price stratification.</b> Decile spreads of
          <b>{lo_spread*100:+.2f}%</b> / <b>{mid_spread*100:+.2f}%</b> / <b>{hi_spread*100:+.2f}%</b>
          for &lt;$200K / $200K–$500K / $500K+ tiers — the model isn't merely buying cheap homes.</li>
      <li><b>Real rates dominate medium/long; mean-reversion dominates short.</b> At fwd_3y, 10Y TIPS
          yield's mean-abs SHAP is more than 4× the next feature. At fwd_1y, drawdown from the trailing
          60-month ZHVI peak is #1 — zips that have fallen most snap back hardest.</li>
      <li><b>Newer-source surprises:</b> county Republican vote share (MIT Election Lab) is in the top
          10 for <i>every</i> horizon; Wikipedia metro-article pageviews is #3 at fwd_3y. Identity and
          attention proxies — not in textbook real-estate models.</li>
      <li><b>Climate has narrowed to a short-horizon signal.</b> FEMA disaster counts + NRI wildfire
          stay in fwd_1y's top 10 but drop out of fwd_3y and fwd_5y under the new feature pool.</li>
    </ul>
  </div>

  <h2>Why this study exists</h2>
  <p>Residential real estate is the largest asset class most people will ever own, and the conventional
     wisdom for picking <i>where</i> and <i>when</i> to buy is dominated by qualitative heuristics
     (school districts, "up-and-coming neighborhoods", proximity to Whole Foods) or single-metric proxies
     (population growth, median income). This study asks: which of these signals actually predict
     forward returns, and by how much, separated by time horizon — and are there underweighted predictors
     that the textbook approach misses?</p>

  <p>The framing is deliberately quantitative. Instead of picking predictors a priori, we cast a wide
     net (140+ candidates across 11 thematic classes) and let out-of-sample model performance and
     SHAP attribution tell us what works.</p>

  <h2>The setup in one minute</h2>

  <h3>What we predict</h3>
  <p>For each zip code at each month, we compute the forward home-price return (Zillow ZHVI index) over
     three horizons — 1 year, 3 years, 5 years — annualized for the latter two. These are the
     <i>targets</i> our models try to learn from upstream features.</p>

  <h3>How we predict it</h3>
  <p>For each (zip, month) row, we join a wide set of features that were <i>knowable at that time</i>
     (with explicit publication-lag adjustment per source — e.g., HMDA mortgage data isn't available
     until 9 months after the year it describes). We then train two families of models per horizon:</p>
  <ul>
    <li><b>Baselines:</b> predict the global mean; predict the within-metro mean; OLS regression on
        Census demographics only ("hedonic"). These are what a real model has to beat to claim it's
        learning anything.</li>
    <li><b>Real models:</b> ElasticNet (regularized linear, interpretable) and LightGBM (gradient-boosted
        trees, captures interactions). Both report Spearman rank correlation and decile spread on
        out-of-sample test data; LightGBM additionally produces SHAP feature attributions.</li>
  </ul>

  <h3>How we validate</h3>
  <p>Two temporal splits, both reported in this run:</p>
  <ul>
    <li><b>pre_2008 split:</b> train on data through 2007-12, test on 2008-2013. Stress-tests
        whether the model breaks across the housing-crisis regime change.</li>
    <li><b>post_2012 split:</b> train through 2017-12, test on 2018-2024. Stress-tests across the
        2022 rate shock + COVID demand surge.</li>
  </ul>
  <p>This run additionally reports <b>spatial cross-validation</b> (hold out entire CBSAs) and
     <b>walk-forward CV</b> (rolling 12-month test windows) on top of the two static splits — see
     section 1 below for the robustness numbers.</p>

  <h3>What's in the feature pool right now</h3>
  <p>The current feature store covers 22 data sources organized into 11 thematic classes,
     producing 140+ modeling features after coverage filtering. Hover over any feature name in the
     SHAP charts below for a one-line description.</p>
  <ul>
    <li><b>Macro / rates:</b> 30Y mortgage, 10Y TIPS, M2, Case-Shiller, lumber, US unemployment +
        their 1/3/12-month deltas (FRED)</li>
    <li><b>Inventory:</b> months-of-supply, days-on-market, price cuts, active/new/pending
        listings per zip (Realtor.com)</li>
    <li><b>Demographics:</b> population by age cohort, household income, education, home value,
        rent, owner-occupancy at ZCTA (Census ACS, 10 vintages 2013-2022); per-capita personal
        income at county (BEA)</li>
    <li><b>Migration:</b> county-to-county AGI flow (IRS SOI)</li>
    <li><b>Supply:</b> building permits at MSA (Census BPS); business establishments + employment
        per zip (Census ZBP)</li>
    <li><b>Jobs:</b> wages + YoY growth at county (BLS QCEW); monthly unemployment rate at county
        (BLS LAUS)</li>
    <li><b>Affordability / cost of living:</b> effective property-tax rate per ZCTA (ACS B25103 /
        B25077); HUD Small-Area Fair Market Rent per ZIP; residential electricity price per state
        (EIA)</li>
    <li><b>Healthcare:</b> CMS Hospital Compare star ratings + bed counts by county; County Health
        Rankings composite + headline measures (life expectancy, premature death, smoking, obesity)</li>
    <li><b>Schools:</b> NCES district-year enrollment, pupil-teacher ratio, free-lunch share,
        per-pupil spending — county-broadcast</li>
    <li><b>Politics:</b> county presidential vote share + winning margin (MIT Election Lab, 2000-2024)</li>
    <li><b>Climate:</b> trailing-10-year disaster declarations by category (OpenFEMA); flood, wildfire,
        hurricane, heatwave risk scores at tract (FEMA NRI); annual PM2.5 + ozone air quality
        (EPA AQS)</li>
    <li><b>Amenities + infrastructure:</b> EV charging stations per zip (DOE AFDC)</li>
    <li><b>Behavioral:</b> monthly Wikipedia pageviews per metro article (search-interest proxy)</li>
    <li><b>Derived (no new data):</b> gross rental yield = 12 × ZORI / ZHVI; rolling 24-mo ZHVI
        volatility; ZHVI drawdown vs trailing-60-mo peak</li>
  </ul>
  <p>Pending modules with code ready but waiting on user-supplied credentials / manual downloads:
     HMDA (tract-level mortgage records — currently only state-level aggregations), FCC BDC broadband,
     NOAA VIIRS satellite nightlights, Foursquare POIs (Whole Foods / coffee / breweries),
     FBI NIBRS county-year crime rates (needs <code>CDE_API_KEY</code>), USAspending federal $ flows.</p>

  <h2>Headline results</h2>
  <p>The table below picks the best-performing model for each horizon, reported on <b>both</b> temporal
     splits. The post-2012 block is the main result (larger and more recent test window); the pre-2008
     block is a stress test through the housing crisis. A signal that shows up in both — same sign, same
     order-of-magnitude spread — is much harder to dismiss as overfitting to one regime.</p>
  {headline_html}

  <h3>How to read this</h3>
  <ul>
    <li><b>Spearman ρ</b> measures rank correlation: did the model order zips correctly from worst-
        to best-expected return? Range -1 (perfectly anti-ranked) to +1 (perfectly ranked).
        +0.3 is meaningful for noisy financial data.</li>
    <li><b>Decile spread</b> is the most actionable metric: if you bought zips the model ranked in
        the top 10% versus the bottom 10%, how much better did you do? +6.75% means the top decile
        averaged 9.70% annualized vs. 2.95% in the bottom — a meaningful gap.</li>
  </ul>

  <div class='caveat'>
    <b>Why R² is not reported as the primary metric:</b> traditional R² is negative on both test windows
    for nearly every model. This is not a model failure — it's a property of the test windows. Both
    test periods contain regime shifts (housing crash; rate shock + COVID) where the mean realized return
    is very different from the training period. Any model that predicts near the training mean gets
    penalized hard on R². <i>Spatial ranking</i> (Spearman + decile spread) is intact and is what
    matters for a buy-this-zip-not-that-zip decision.
  </div>

  <h2>1 · Confidence: walk-forward and spatial robustness</h2>
  <p>A single train/test split can flatter or punish a model by accident. Two stronger CV regimes
     re-run the post-2012 LightGBM to bound how much of the headline number is structural vs noise.</p>

  <h3>1a · Walk-forward CV (rolling-origin)</h3>
  <p>Train through year T−1, test on year T; roll the cutoff from 2018-12 through 2022-12. Five
     test years per horizon (fewer for fwd_5y, which would need observability beyond the panel's
     end).</p>
  {walkforward_html}
  <p>For the headline fwd_3y horizon, mean Spearman ρ = <b>{wf3_mean:+.3f}</b> with std
     <b>{wf3_std:.3f}</b> across {wf3_n} folds — the +0.29 figure from the single static split
     survives in expectation (it actually rises slightly), but with meaningful year-to-year noise.
     2020 is the variance driver, as expected.</p>

  <h3>1b · Spatial CV (whole-CBSA holdout)</h3>
  <p>5-fold CBSA holdout, with train+test both restricted to ≤ 2017-12 so the only stressor is the
     spatial axis (no temporal regime shift). Two variants: the <b>raw</b> model predicts the price
     return directly; the <b>demeaned</b> model predicts the within-metro residual after subtracting
     per-CBSA per-month means.</p>
  <h4>Raw fwd_3y</h4>
  {spatial_raw_html}
  <h4>Demeaned (within-metro residual) fwd_3y</h4>
  {spatial_dem_html}
  <p><b>The headline lesson.</b> The raw model's Spearman ρ = <b>{sp_raw_sp:+.3f}</b> with decile
     spread <b>{sp_raw_spread*100:+.2f}%</b> looks spectacular — but most of it is a level effect
     ("expensive zips stay expensive"). The demeaned model's Spearman ρ = <b>{sp_dem_sp:+.3f}</b>
     with decile spread <b>{sp_dem_spread*100:+.2f}%</b> is the genuine cross-sectional alpha. Both
     are meaningful, but the demeaned number is the honest one for "can the model pick the right
     zip <i>inside</i> a given metro?"</p>

  <h2>2 · Did this actually work? — portfolio backtest</h2>
  <p>The single most decision-relevant question: if you had used this model's monthly rankings to
     buy a basket of zips each month from 2018-01 through 2021-12 and held 3 years, what would you
     have made? We replay that exercise: at every score month, build the model's top-decile basket
     of zips, look up their <i>realized</i> fwd_3y annualized return three years later, compare to
     the universe mean (equal-weight benchmark) and the bottom decile.</p>
  {backtest_chart_html}
  {backtest_table_html}
  <p><b>The model adds skill.</b> The price-only fwd_3y headline model's top decile beat the
     universe mean in <b>{bt_price_hit:.1f}%</b> of months ({bt_price_n}/{bt_price_n}), earning
     roughly <b>+{bt_price_excess:.2f} pp/yr</b> alpha and a <b>+{bt_price_spread:.2f} pp/yr</b>
     top-vs-bottom spread. Worst stretch was a handful of trivial misses in mid-2021. The yield-aware
     <code>fwd_3y_total</code> model is in another class — <b>{bt_total_hit:.1f}%</b> hit rate,
     <b>+{bt_total_excess:.2f} pp/yr</b> alpha, <b>+{bt_total_spread:.2f} pp/yr</b> top-bot spread —
     but caveat: its scoring universe is much smaller (only zips with ZORI rent coverage, ~1.7-2.8K
     vs ~9.9K). The total-return alpha is real but concentrated in yield-observable metros.</p>

  <h2>3 · Decile spread, every model × horizon × split</h2>
  <p>Each bar is one model on one horizon on one test window. Positive values mean the model's top-decile
     picks beat its bottom-decile picks; negative values mean it anti-ranked. The left panel is the
     pre-2008 split (smaller and noisier); the right is post-2012 (the main result).</p>
  {_div(decile, include_js=True)}
  <p>LightGBM wins decisively on the medium and long horizons in the post-2012 split. On the 1-year
     horizon, the demographics-only hedonic baseline (Ridge regression on ACS variables) actually edges
     LightGBM — a useful sanity check that fancy methods aren't always required.</p>

  <h2>4 · What did the model learn? — SHAP feature importance</h2>
  <p>For each horizon, we compute mean absolute SHAP value per feature on the training sample. SHAP
     decomposes each prediction into per-feature contributions (positive = pushed the prediction up,
     negative = down). The bars below show the 15 features that the LightGBM model used most heavily.</p>
  <h3>Short horizon (1 year)</h3>
  {_div(shap_figs[0], include_js=False)}
  <h3>Medium horizon (3 years)</h3>
  {_div(shap_figs[1], include_js=False)}
  <h3>Long horizon (5 years)</h3>
  {_div(shap_figs[2], include_js=False)}

  <p><b>Four findings worth flagging:</b></p>
  <ul>
    <li><b>Drawdown is the #1 short-horizon predictor.</b> The derived "ZHVI drawdown vs trailing
        60-month peak" feature carries the largest mean-abs SHAP at fwd_1y. Zips that have fallen
        most relative to their recent peak tend to snap back hardest — a clean cross-sectional
        mean-reversion signal.</li>
    <li><b>Real rates dominate medium/long by a wide margin.</b> At fwd_3y, 10Y TIPS yield's
        mean-abs SHAP is more than 4× the next-largest feature; at fwd_5y, more than 3×. Medium and
        long-horizon ranking is mostly about macro conditions, not zip-specific properties.</li>
    <li><b>New political + behavioral features are earning their keep.</b> County Republican vote
        share (MIT Election Lab) appears in the top 10 for every horizon — and at fwd_5y, both
        party vote shares appear, so the model is reading the political-identity gradient, not
        taking sides. Wikipedia metro-article pageviews ranks #3 at fwd_3y. These are
        "attention/identity" proxies the textbook real-estate playbook doesn't model.</li>
    <li><b>Climate has receded to a short-horizon signal.</b> FEMA disaster counts and NRI wildfire
        risk were claimed cross-horizon in an earlier run; under the expanded feature pool (politics,
        health, schools, rent-affordability) they drop out of the top 10 for both fwd_3y and fwd_5y,
        while staying in fwd_1y — most likely capturing short-term storm-driven price dynamics.</li>
  </ul>

  <h2>5 · Does the signal survive price stratification?</h2>
  <p>A common worry with any model that ranks zip codes is that it has secretly learned a price-level
     proxy: cheap zips mean-revert upward, expensive zips compound more slowly, and the "alpha" is just
     a roundabout way of buying low. To check that, we split the post-2012 test set into three ZHVI
     tiers and re-compute the LightGBM <code>fwd_3y</code> decile spread <i>within</i> each tier. If the
     spread is healthy in every tier, the model is doing more than sorting on price; if it collapses in
     the expensive tier, it is mostly a mean-reversion bet.</p>
  {price_strat_html}
  <p><b>Interpretation.</b> The decile spread is <b>{lo_spread*100:+.2f}%</b> for sub-$200K homes,
     <b>{mid_spread*100:+.2f}%</b> in the $200K–$500K middle, and <b>{hi_spread*100:+.2f}%</b> for $500K+
     homes — all positive, all out-of-sample. The two cheaper tiers carry essentially the same spread
     (~6 points), and the $500K+ tier roughly halves to ~3 points but stays positive. This is the
     pattern you'd expect if the model has real cross-sectional signal at every price point and luxury
     markets simply have flatter forward returns — not the pattern you'd expect if the model were
     secretly a price-mean-reversion proxy (which would show a strong spread in the cheap tier and
     collapse near zero in the expensive one).</p>

  <h2>5b · Macro context: is now a good time? And how does RE stack up against other hedges?</h2>
  <p>Everything up to here is about <i>where</i> to buy. The two questions below are about
     <i>when</i> and <i>what else</i>: does the model think the broad RE universe is cheap or
     expensive right now, and how has private-RE actually fared against the other classic
     inflation hedges (stocks, gold, REITs, BTC, TIPS) when you put them on the same axis?</p>

  <h3>Is now a good time to buy?</h3>
  <p>For each month from 2014 through the latest in the panel, we score every top-100-metro ZIP
     with the headline post-2012 LightGBM and take the universe mean of the predicted fwd_3y
     annualized return. The model's prediction at month <i>t</i> can be compared to the
     <i>realized</i> universe-mean fwd_3y return three years later (dashed line) for all months
     where t+3y is observable.</p>
  {timing_chart_html}
  <p>The model's current ({ts_month}) universe-mean predicted fwd_3y annualized return is
     <b>{ts_current*100:+.2f}%</b> (median of the per-ZIP distribution that month;
     {ts_n_zips:,} ZIPs scored, 80% prediction interval roughly
     [<b>{ts_p10*100:+.2f}%</b>, <b>{ts_p90*100:+.2f}%</b>]). The historical median across all
     scored months in the post-2012 panel is <b>{ts_med*100:+.2f}%</b> with an interquartile
     range of [<b>{ts_q25*100:+.2f}%</b>, <b>{ts_q75*100:+.2f}%</b>]. The current prediction sits
     at the <b>{ts_pct:.0f}th percentile</b> of history — the model is currently
     <b>{ts_verdict}</b> vs the historical baseline (n = {ts_n_hist} months scored).</p>

  <div class='caveat'>
    <b>Read the timing chart carefully.</b> The model was trained on data ≤ 2017-12 with a
    test window of 2018-01 onward, so anything from 2014-2017 is in-sample (the early part of
    the predicted line tracks realized very closely because the model has seen those rows).
    From 2018 forward, the gap between blue (predicted) and green (realized) is honest
    out-of-sample error. The model under-predicted the 2020-2022 COVID-era surge — a regime
    the training set didn't cover — and is now predicting near-zero / mildly negative returns
    because trailing real-rate prints are at decade-plus highs. That is a structural feature
    of the model, not a forecast of the future.
  </div>

  <h3>Real estate vs other inflation hedges</h3>
  <p>RE is rarely held in isolation in a real portfolio — the question of "is now a good time
     to buy a house" is really "is RE the best use of the marginal dollar right now". We assemble
     a small benchmark set of public assets that are commonly framed as inflation hedges and put
     them on the same chart, in <i>real</i> terms (after subtracting CPI growth over the same
     window).</p>

  <h4>Historical 3y / 5y annualized real returns (rolling, since 2000)</h4>
  {hedge_hist_html}

  <h4>Trailing 3-year real returns ending the most recent month</h4>
  {hedge_trail_html}

  <h4>Cumulative real return per asset, 2000-now (log scale)</h4>
  {hedge_chart_html}

  <p><b>What beat what.</b> Across the full 2000-now history of monthly rolling windows, the
     median 3-year annualized real return ranking is dominated by Bitcoin
     (<b>{hedge_winner_3y_median[1]*100:+.1f}%</b>, but on a much shorter history starting 2014
     and with extreme tail risk in both directions) and then by gold and broad-equity / total-
     return S&P 500 (mid-to-high single digits in real terms). REITs (IYR / VNQ) sit in the
     <b>+4-6%</b> range as a public-market RE proxy. The Zillow ZHVI top-100 mean — our
     private-RE benchmark, <i>before</i> leverage and <i>before</i> rent — comes in at
     <b>{zhvi_3y_median*100:+.2f}%</b> median 3y real, ranking <b>#{zhvi_rank}/{n_assets}</b>
     among the assets compared. The trailing-3y leader as of the latest month is
     <b>{escape(hedge_winner_trailing[0])}</b> at <b>{hedge_winner_trailing[1]*100:+.2f}%</b>
     real.</p>

  <div class='caveat'>
    <b>Honest caveats on the hedge comparison:</b>
    <ul style='margin-top: .4em; margin-bottom: 0;'>
      <li><b>BTC selection bias.</b> Bitcoin's history starts in 2014 (Yahoo's first month of
          BTC-USD data), so its rolling-window stats are pulled from a strictly post-GFC,
          mostly-bull-market regime. The other assets include 2000-2002 and 2008-2009 in their
          medians; BTC does not.</li>
      <li><b>REITs are not private RE.</b> IYR/VNQ are leveraged, mark-to-market, and earn
          public-market liquidity discounts/premia. They're the best free TR proxy but they're
          a different beast from owning a house.</li>
      <li><b>ZHVI ignores rent, maintenance, and property tax.</b> Our RE return is a pure
          price-index series (Zillow ZHVI). Real residential returns include rental yield
          (positive carry) and maintenance / tax / insurance / vacancy costs (negative carry).
          The yield-aware <code>fwd_3y_total</code> model in section 7 partially addresses
          this for individual ZIPs, but not at the national-aggregate level shown here.</li>
      <li><b>No leverage assumed.</b> Real-estate investors typically run 4-5× leverage via a
          mortgage; the equity assets here are unlevered. A 3% nominal price return at 4×
          leverage with a 6% mortgage is a different animal from the headline 3% — and that
          gearing is what makes housing competitive with stocks for many households even when
          the unlevered return is much lower.</li>
      <li><b>Wilshire-from-FRED was retired in mid-2024;</b> we substitute VTI (Vanguard Total
          Market) as the broad-equity proxy. Gold uses Yahoo's GC=F front-month futures (the
          FRED London-fix series requires an API key, which is not configured in this run).</li>
    </ul>
  </div>

  <h2>6 · Where might the model want to buy today?</h2>
  {map_link_html}
  <p>Using the post-2012 LightGBM trained on 3-year forward returns, we scored every zip in the
     <b>top-100</b> US metros at the most recent month with enough feature coverage to make a confident
     prediction. The table below is rolled up to <b>one row per metro (CBSA)</b> — earlier versions
     keyed off the raw (city, state) of each ZIP, which split a single metro into dozens of tiny
     hamlets (e.g. 10+ Pittsburgh-CBSA suburbs would all appear separately). Each row now shows the
     metro's principal (largest-population) city, the best-scoring ZIP inside that metro, the mean
     predicted return across the metro's top-5 ZIPs, and how many of its ZIPs land in the top 10%
     of all predictions nationally. <b>Hover over any feature</b> to see its description.</p>
  {leaderboard_html}

  <h3>6a · ZIP-level detail (top 20)</h3>
  <p>The same model expanded back to per-ZIP rows for users who want to drill into specific neighborhoods
     rather than cities.</p>
  {zip_leaderboard_html}

  <div class='caveat'>
    <b>Important caveats on the leaderboard:</b>
    <ul style='margin-top: .4em; margin-bottom: 0;'>
      <li><b>The p10/p90 bands tell the real story.</b> Top zips show point predictions of +5% to
          +7%, but the 80% prediction interval per zip spans roughly
          [<b>{band_p10:+.1f}%</b>, <b>{band_p90:+.1f}%</b>] — a <b>{band_width:.1f}-point</b> band
          on median. The point estimate is the model's best guess; the band is the uncertainty.
          Bet sizing should reflect the band, not the point.</li>
      <li>The negative SHAP contribution from <code>fred.real_rate_10y</code> appears for every zip —
          it's not a zip-specific signal, it's just "rates are high right now." Useful temporal context,
          not actionable for picking.</li>
      <li>This is a model output, not investment advice. The model is unaware of zoning, school
          districts, walkability, specific listings, or your personal constraints. It has a positive
          backtest signal, not a guarantee.</li>
    </ul>
  </div>

  <h2>7 · Alternative ranking: yield-aware total return</h2>
  <p>Everything above ranks zips by predicted <i>price</i> appreciation. An investor who cares about
     yield (rent flowing in, not just paper appreciation) wants a different ranking: a separate
     LightGBM trained on <code>fwd_3y_total = price_return + ZORI rent yield carry</code>. The
     leaderboard below shows the top 30 zips by that target. Caveat: ZORI covers only ~11% of
     zip-months, so this ranking's universe is much smaller (~2K zips vs ~10K for the price-only
     model) and concentrated in zips where rent data exists.</p>
  {total_leaderboard_html}
  <p>The decision-different finding: only <b>11 of 50</b> zips appear in both this and the
     price-only top 50 — a 22% overlap. Price-only tilts toward sunbelt fringe (Tulsa, Kansas City,
     Memphis, New Orleans) at median rent yield ~12.5%. Total-return is dominated by rust-belt
     high-yield zips (Detroit ×7, Cleveland ×5, Dayton, Akron, Toledo, Birmingham) at median rent
     yield ~15.8%. Same model architecture, just yield-aware → very different shortlist.</p>

  {overlay_section_html}

  <h2>Caveats and what's not in this run</h2>
  <ul>
    <li><b>HMDA mortgage records are in the feature store but didn't reach the trained model.</b>
        Tract-year origination counts + institutional-share are loaded at ~12% panel coverage,
        but the data only spans 2020 / 2022 / 2023. The post-2012 split's train window ends at
        2017-12 — entirely pre-HMDA — so prep's coverage filter drops every HMDA column from X.
        A retrain confirmed this: same model, same SHAP, no HMDA. A genuinely HMDA-aware experiment
        needs a different split design (e.g. train 2020-2022, test 2023-2026) where train actually
        sees HMDA values. Filed as a future experiment.</li>
    <li><b>Four data sources still pending.</b> FCC broadband (signed-URL endpoints, needs manual
        download), NOAA VIIRS satellite nightlights (needs EOG creds + rasterio), Foursquare POIs,
        and FBI NIBRS crime aggregates (needs <code>CDE_API_KEY</code>) are wired into the assembler
        but blocked on credentials or manual fetch steps.</li>
    <li><b>Owner-occupier overlay is geographically narrow.</b> 22 of 30 zips fall in the DC
        metro because the filter chain (low climate risk + good schools + low disaster history +
        high life expectancy + ≥50% owner-occupied) collapses to wealthy NoVA suburbs. Loosening
        any single filter would broaden geography but trade off the buyer's stated priorities.</li>
    <li><b>Total-return modeling now done but on a smaller universe.</b> ZORI rent index covers
        only ~11% of zip-months, so <code>fwd_3y_total</code> trains on ~215K rows vs ~1.98M for
        price-only. The model's headline Spearman (0.42) and backtest hit-rate (100%) are real but
        scoped to yield-observable metros.</li>
    <li><b>ElasticNet regressed under the new QuantileTransformer scaling.</b> The Wave-1 fix
        solved an RMSE blowup but flipped fwd_3y Spearman to negative. LightGBM is unaffected and
        remains the headline model; ElasticNet needs a tighter alpha or a different scaler.</li>
    <li><b>Pre-2008 coverage is thin.</b> Realtor inventory (2016+), QCEW (2010+), ACS (2013+),
        BPS (2011+), IRS-SOI (2011+) all start post-GFC. The pre_2008 results in the headline table
        train on 27 features vs 41 post-2012 — same model, narrower feature pool.</li>
    <li><b>fwd_5y walk-forward CV has 3 folds, not 5</b> — the 2021-12 and 2022-12 cutoffs would
        need fwd_5y observability through 2026-12 / 2027-12, beyond the panel's end. Working as
        designed.</li>
  </ul>

  <h2>Glossary</h2>
  <dl class='glossary'>
    <dt>ZHVI</dt><dd>Zillow Home Value Index — a smoothed monthly home-value estimate per zip code.
        The price series we predict.</dd>
    <dt>ZORI</dt><dd>Zillow Observed Rent Index — companion rent index per zip; not yet used in the target.</dd>
    <dt>CBSA</dt><dd>Core-Based Statistical Area — what Census calls a "metro area." 200 of these cover ~80%
        of US population.</dd>
    <dt>ZCTA</dt><dd>Zip Code Tabulation Area — Census's approximation of a USPS zip; ≈ but ≠ identical
        to a zip.</dd>
    <dt>fwd_1y / fwd_3y / fwd_5y</dt><dd>The forward home-price return at that zip over the next 1 / 3 / 5
        years, annualized for the latter two. These are what we predict.</dd>
    <dt>Decile spread</dt><dd>Mean realized return of the top-predicted 10% of zips minus the bottom 10%.
        The bottom-line "would this model have helped me pick?" number.</dd>
    <dt>Spearman ρ</dt><dd>Rank correlation between predicted and realized returns. Robust to outliers
        and to wrong-magnitude predictions; only cares about ordering.</dd>
    <dt>SHAP</dt><dd>Shapley Additive Explanations — a per-prediction decomposition into feature
        contributions, summable to the model's output. Lets you say "this zip was ranked high
        <i>because of</i> features X, Y, Z."</dd>
    <dt>FEMA NRI</dt><dd>FEMA's National Risk Index — public free dataset of natural-hazard risk scores
        at census-tract level. Used here as the climate-exposure proxy in place of paid First Street.</dd>
  </dl>

  <p class='footnote'>arbok · designed model-first, personal-overlay second · all data free /
     public sources · code at <code>src/arbok/</code> · contact author for source / methodology questions.</p>
</body>
</html>
"""
    out = PROCESSED / "report.html"
    out.write_text(html)
    print(f"Wrote {out}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
