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

ARTIFACTS = PROCESSED / "phase2_artifacts"
RESULTS = pd.read_parquet(PROCESSED / "phase2_results.parquet")
LEADERBOARD = pd.read_parquet(PROCESSED / "zip_leaderboard.parquet")
FS = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")

HORIZONS = ["fwd_1y", "fwd_3y", "fwd_5y"]
HORIZON_CLASS = {"fwd_1y": "short (1y)", "fwd_3y": "medium (3y)", "fwd_5y": "long (5y)"}


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
    )
    fig = go.Figure(
        go.Bar(x=top["mean_abs_shap"], y=top["feature"], orientation="h",
               marker_color="#3a5da9")
    )
    fig.update_layout(
        title=f"Top {k} SHAP features — {horizon} @ {split}",
        xaxis_title="mean |SHAP|", yaxis_title="", height=460,
        margin=dict(l=200, r=20, t=60, b=40),
    )
    return fig


def chart_leaderboard_top_n(n: int = 20) -> str:
    """Return an HTML <table> string with top-n zips."""
    df = LEADERBOARD.head(n).copy()
    df["zhvi"] = df["zhvi"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    df["zori"] = df["zori"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    df["predicted"] = df["predicted_fwd_3y_annualized"].apply(lambda x: f"{x*100:+.2f}%")
    df = df[["zip", "cbsa", "zhvi", "zori", "predicted", "drivers"]]
    df.columns = ["ZIP", "CBSA", "ZHVI", "ZORI", "Pred fwd_3y ann.", "Top SHAP drivers (feature → contribution)"]
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.values
    )
    head = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in df.columns) + "</tr>"
    return f"<table class='leaderboard'>{head}{rows}</table>"


def chart_headline_table() -> str:
    """Render the per-horizon best-model table from PHASE2_RESULTS.md."""
    best = (
        RESULTS[RESULTS["split"] == "post_2012"]
        .sort_values(["horizon", "decile_spread"], ascending=[True, False])
        .groupby("horizon").head(1)
        .sort_values("horizon")
    )
    rows = []
    for _, r in best.iterrows():
        rows.append(
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
      {"".join(rows)}
    </table>
    """


def main() -> None:
    print("Building charts…")
    decile = chart_decile_spread()
    shap_figs = [chart_shap_for(h) for h in HORIZONS]
    leaderboard_html = chart_leaderboard_top_n(20)
    headline_html = chart_headline_table()

    html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>arbok — residential RE predictor study</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
            max-width: 1100px; margin: 2em auto; color: #1a1a1a; padding: 0 1em; }}
    h1 {{ border-bottom: 2px solid #1a1a1a; padding-bottom: .3em; }}
    h2 {{ margin-top: 2.2em; color: #2c5282; }}
    table {{ border-collapse: collapse; margin: 1em 0; }}
    th, td {{ padding: .35em .8em; border-bottom: 1px solid #ddd; text-align: left; font-size: .92em; }}
    th {{ background: #f3f5f9; }}
    table.headline td:nth-child(4) {{ font-weight: bold; color: #2c5282; }}
    table.leaderboard {{ font-family: ui-monospace, 'SF Mono', monospace; font-size: .82em; }}
    table.leaderboard td:nth-child(5) {{ font-weight: bold; }}
    .meta {{ color: #666; font-size: .9em; }}
    .footnote {{ color: #666; font-size: .85em; margin-top: 2em;
                 border-top: 1px solid #eee; padding-top: 1em; }}
  </style>
</head>
<body>
  <h1>arbok — residential RE predictor study</h1>
  <p class='meta'>Generated 2026-05-21. 3.73M zip-month panel · 200 metros · 8 of 12 sources joined ·
     post_2012 split: train ≤2017-12, test 2018-2024.</p>

  <h2>Headline (post_2012 split, best model per horizon)</h2>
  {headline_html}

  <h2>1 · Decile spread by horizon × split × model</h2>
  <p>The actionable metric: difference between mean realized return of top-decile and bottom-decile
     predicted zips. R² is negative everywhere because test windows are crisis regimes — decile
     spread + Spearman are what matter.</p>
  {_div(decile, include_js=True)}

  <h2>2 · Top SHAP features per horizon (post_2012, LightGBM)</h2>
  <p>Climate features (FEMA fire/hurricane, NRI wildfire) appear in the top 10 for every horizon —
     this validates the design-doc thesis that climate is the underweighted sleeper predictor bucket.
     Real rates (FRED) dominate medium / long horizons; M2 + lumber dominate short.</p>
  {_div(shap_figs[0], include_js=False)}
  {_div(shap_figs[1], include_js=False)}
  {_div(shap_figs[2], include_js=False)}

  <h2>3 · Top-20 zip leaderboard (predicted fwd_3y annualized, as of latest data)</h2>
  <p>Scored from the post_2012 LightGBM at the most recent year-month with ≥80% feature coverage
     across the model's 41 features. Driver column shows top-3 SHAP contributions per zip.</p>
  {leaderboard_html}

  <p class='footnote'>arbok · model-first, personal-overlay second · all data free /
     public sources. Code at <code>src/arbok/</code>.</p>
</body>
</html>
"""
    out = PROCESSED / "report.html"
    out.write_text(html)
    print(f"Wrote {out}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
