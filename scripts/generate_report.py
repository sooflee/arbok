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
FS = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
ZIP_GEO = pd.read_parquet(PROCESSED / "zip_geo.parquet") if (PROCESSED / "zip_geo.parquet").exists() else None
TOP200 = pd.read_csv(PROCESSED / "top200_metros.csv") if (PROCESSED / "top200_metros.csv").exists() else None
if TOP200 is not None:
    TOP200["cbsa"] = TOP200["cbsa"].astype(str).str.zfill(5)
    # Pick the largest-population city per CBSA as the metro's display name.
    # core_name like 'Pittsburgh' or 'New York-Newark-Jersey City'; first token is the
    # principal (largest-pop) city by Census convention.
    TOP200["principal_city"] = TOP200["core_name"].str.split("-").str[0]
    TOP200["principal_state"] = TOP200["state"].str.split("-").str[0]

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
    that CBSA (the principal city from top200_metros.csv).
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

    # Attach principal city/state from top200_metros (population-largest in CBSA).
    if TOP200 is not None:
        df = df.merge(
            TOP200[["cbsa", "principal_city", "principal_state", "population"]],
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
    df["strong"] = df["strong_zip_count"].astype(str)
    df["drivers_html"] = df["drivers"].apply(_humanize_drivers)
    # Column order matches the zip leaderboard so the shared CSS rule
    # (.leaderboard td:nth-child(6) bold/blue) lights up the predicted-return column.
    df = df[["location", "cbsa", "best_zip", "best_zhvi", "pred_top5", "pred_best", "strong", "drivers_html"]]
    df.columns = ["Metro (principal city)", "CBSA", "Best ZIP", "Best ZIP price", "Mean pred (top 5 zips)", "Pred for best ZIP", "Zips in top 10%", "Top SHAP drivers for best ZIP (hover)"]
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
    df["drivers_html"] = df["drivers"].apply(_humanize_drivers)
    df = df[["zip", "location", "cbsa", "zhvi_fmt", "zori_fmt", "predicted", "drivers_html"]]
    df.columns = ["ZIP", "City, State", "CBSA", "ZHVI", "ZORI", "Pred fwd_3y ann.", "Top SHAP drivers (hover for feature description)"]
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
  <p class='meta'>arbok study · generated 2026-05-21 · all data from free / public sources</p>

  <div class='tldr'>
    <h3>TL;DR</h3>
    <ul>
      <li>Built a 3.73-million-row panel of every zip code in the top 200 US metros, monthly, 2000-2026,
          joined with 50+ candidate predictors organized into 10 thematic classes (macro, demographics,
          climate, supply, amenities, etc.).</li>
      <li>Trained gradient-boosted models to predict 1-, 3-, and 5-year forward home-price returns
          per zip on two backtest windows: pre-2008 (GFC stress test) and post-2012 (rate-shock + COVID).</li>
      <li><b>Best signal:</b> 3-year horizon, post-2012 split. Top decile of model-predicted zips averaged
          <b>9.7% annualized</b> realized returns vs. bottom decile <b>3.0%</b> — a 6.75-point spread
          out of sample.</li>
      <li><b>The most surprising winner:</b> climate exposure (FEMA disaster history + NRI wildfire/flood
          scores) ranks in the top 10 predictors for <i>every</i> horizon. Underweighted in standard
          real-estate analysis.</li>
      <li><b>The expected winners:</b> real interest rates dominate medium and long horizons;
          M2 money supply and lumber prices dominate short.</li>
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
     net (50+ candidates across 10 thematic classes) and let out-of-sample model performance and
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
  <p>A future improvement is spatial cross-validation (hold out entire metros instead of just future
     months) and walk-forward CV (rolling refit). The current splits are honest but a single
     experiment each.</p>

  <h3>What's in the feature pool right now</h3>
  <p>The current feature store covers 14+ data sources organized into 10 thematic classes,
     producing ~50 modeling features after coverage filtering. Hover over any feature name in the
     SHAP charts below for a one-line description.</p>
  <ul>
    <li><b>Macro / rates:</b> 30Y mortgage, 10Y TIPS, M2, Case-Shiller, lumber, US unemployment +
        their 1/3/12-month deltas (FRED)</li>
    <li><b>Inventory:</b> months-of-supply, days-on-market, price cuts, active/new/pending
        listings per zip (Realtor.com)</li>
    <li><b>Demographics:</b> population by age cohort, household income, education, home value,
        rent, owner-occupancy at ZCTA (Census ACS, 10 vintages 2013-2022)</li>
    <li><b>Migration:</b> county-to-county AGI flow (IRS SOI)</li>
    <li><b>Supply:</b> building permits at MSA (Census BPS); business establishments + employment
        per zip (Census ZBP)</li>
    <li><b>Jobs:</b> wages + YoY growth at county (BLS QCEW); monthly unemployment rate at county
        (BLS LAUS)</li>
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
     NOAA VIIRS satellite nightlights, Foursquare POIs (Whole Foods / coffee / breweries), BEA
     per-capita income, USAspending federal $ flows.</p>

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

  <h2>1 · Decile spread, every model × horizon × split</h2>
  <p>Each bar is one model on one horizon on one test window. Positive values mean the model's top-decile
     picks beat its bottom-decile picks; negative values mean it anti-ranked. The left panel is the
     pre-2008 split (smaller and noisier); the right is post-2012 (the main result).</p>
  {_div(decile, include_js=True)}
  <p>LightGBM wins decisively on the medium and long horizons in the post-2012 split. On the 1-year
     horizon, the demographics-only hedonic baseline (Ridge regression on ACS variables) actually edges
     LightGBM — a useful sanity check that fancy methods aren't always required.</p>

  <h2>2 · What did the model learn? — SHAP feature importance</h2>
  <p>For each horizon, we compute mean absolute SHAP value per feature on the training sample. SHAP
     decomposes each prediction into per-feature contributions (positive = pushed the prediction up,
     negative = down). The bars below show the 15 features that the LightGBM model used most heavily.</p>
  <h3>Short horizon (1 year)</h3>
  {_div(shap_figs[0], include_js=False)}
  <h3>Medium horizon (3 years)</h3>
  {_div(shap_figs[1], include_js=False)}
  <h3>Long horizon (5 years)</h3>
  {_div(shap_figs[2], include_js=False)}

  <p><b>Two findings worth flagging:</b></p>
  <ul>
    <li><b>Climate features punch above their weight.</b> FEMA disaster declarations (fire, hurricane,
        severe storm, total count) and NRI wildfire / flood risk scores appear in the top 10 for every
        horizon. The original study design hypothesized that climate was an underweighted bucket
        compared to the textbook macro + demographic predictors — the data agrees.</li>
    <li><b>Real rates separate medium / long horizons from short.</b> M2 money supply and lumber prices
        dominate the 1-year picture; 10-year real rates (TIPS) take #1 for both 3-year and 5-year
        horizons. This is exactly the kind of structural separation a horizon-comparative study is
        meant to surface.</li>
  </ul>

  <h2>3 · Does the signal survive price stratification?</h2>
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

  <h2>4 · Where might the model want to buy today?</h2>
  <p>Using the post-2012 LightGBM trained on 3-year forward returns, we scored every zip in the
     <b>top-100</b> US metros at the most recent month with enough feature coverage to make a confident
     prediction. The table below is rolled up to <b>one row per metro (CBSA)</b> — earlier versions
     keyed off the raw (city, state) of each ZIP, which split a single metro into dozens of tiny
     hamlets (e.g. 10+ Pittsburgh-CBSA suburbs would all appear separately). Each row now shows the
     metro's principal (largest-population) city, the best-scoring ZIP inside that metro, the mean
     predicted return across the metro's top-5 ZIPs, and how many of its ZIPs land in the top 10%
     of all predictions nationally. <b>Hover over any feature</b> to see its description.</p>
  {leaderboard_html}

  <h3>4a · ZIP-level detail (top 20)</h3>
  <p>The same model expanded back to per-ZIP rows for users who want to drill into specific neighborhoods
     rather than cities.</p>
  {zip_leaderboard_html}

  <div class='caveat'>
    <b>Important caveats on the leaderboard:</b>
    <ul style='margin-top: .4em; margin-bottom: 0;'>
      <li>Predictions are compressed to a narrow +2.7% to +3.8% band. Most of the model's signal is
          time-driven (real rates, M2) which is uniform across zips at any single point in time.
          Spatial discrimination is currently weak.</li>
      <li>The negative SHAP contribution from <code>fred.real_rate_10y</code> appears for every zip —
          it's not a zip-specific signal, it's just "rates are high right now." Useful temporal context,
          not actionable for picking.</li>
      <li>This is a model output, not investment advice. The model is unaware of zoning, school
          districts, walkability, specific listings, or your personal constraints. It has a positive
          backtest signal, not a guarantee.</li>
    </ul>
  </div>

  <h2>Caveats and what's not in this run</h2>
  <ul>
    <li><b>Four data sources still pending.</b> HMDA mortgage records, FCC broadband, NOAA satellite
        nightlights, and Foursquare points-of-interest are coded but need manual downloads or paid
        credentials. Adding them is the next concrete expansion.</li>
    <li><b>Single experiment per split.</b> Walk-forward CV with rolling refit would give honest
        confidence bands.</li>
    <li><b>No spatial cross-validation.</b> Holding out whole metros (not just months) would catch
        any leakage from neighboring zips inside the same CBSA being on both sides.</li>
    <li><b>Returns are price-only.</b> Imputed rent yield (using Zillow ZORI) is on the roadmap for
        a "total-return" target.</li>
    <li><b>ACS age-bin codes drift pre-2017</b> (the <code>age_28_38_share</code> feature shows 0%
        for older vintages); known bug, fixable.</li>
    <li><b>Realtor inventory only goes back to 2016-07.</b> Pre-2016 observations have no inventory
        features.</li>
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
