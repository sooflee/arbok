"""Backtest the model's top-decile picks against realized returns.

For each month T in the post-2012 test window (2018-01 .. 2021-12, capped so
t+3y is observable in the panel which extends through 2026-04):
  1. Predict fwd_3y for every zip using the LightGBM model.
  2. Take top-decile / bottom-decile baskets by predicted return.
  3. Look up *realized* fwd_3y for each zip in `targets_zip_month.parquet`.
  4. Aggregate basket means and compare to the universe mean.

Outputs
-------
- ``data/processed/backtest_topdecile.parquet``         (price-only fwd_3y)
- ``data/processed/backtest_topdecile_total.parquet``   (fwd_3y_total)

Both files share the schema:
    score_month, n_zips_top10, n_zips_bot10,
    mean_realized_top10, mean_realized_bot10, mean_realized_universe, spread

Run::

    python scripts/backtest_portfolio.py
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from arbok.config import PROCESSED


# Backtest window. Need t+3y observable; targets panel ends at 2026-04, so the
# last possible score_month is 2023-04. Task asks for 2018-01..2021-12 which is
# safely inside that envelope.
SCORE_START = "2018-01-01"
SCORE_END = "2021-12-01"

TOP_PCT = 0.10
BOT_PCT = 0.10
MIN_COVERAGE = 0.80  # min fraction of features non-null to keep a zip


def backtest_one(
    horizon: str,
    model_path: Path,
    features_path: Path,
    feature_store: pd.DataFrame,
    targets: pd.DataFrame,
    out_path: Path,
    label: str,
) -> pd.DataFrame:
    """Run the backtest for one horizon/model and save the per-month table.

    Parameters
    ----------
    horizon
        Target column name in `targets` (e.g. "fwd_3y" or "fwd_3y_total").
    model_path, features_path
        LightGBM booster + accompanying feature list.
    feature_store
        Wide zip-month feature frame.
    targets
        Long zip-month frame with realized forward returns.
    out_path
        Where to write the resulting per-month parquet.
    label
        Short string for stdout logging.
    """
    t0 = time.time()
    booster = lgb.Booster(model_file=str(model_path))
    features = features_path.read_text().strip().split("\n")
    print(f"\n[{label}] model={model_path.name}  features={len(features)}")

    # Targets keyed by (zip, year_month) -> realized horizon.
    tg = targets[["zip", "year_month", horizon]].dropna(subset=[horizon]).copy()
    tg["year_month"] = tg["year_month"].dt.to_timestamp()  # period[M] -> ts
    tg = tg.rename(columns={horizon: "realized"})

    months = pd.date_range(SCORE_START, SCORE_END, freq="MS")
    print(f"[{label}] scoring {len(months)} months  ({months[0].date()} .. {months[-1].date()})")

    rows = []
    skipped = []
    for m in months:
        snap = feature_store[feature_store["year_month"] == m]
        if len(snap) == 0:
            skipped.append((m, "no feature rows"))
            continue

        X = snap[features].copy()
        # Drop zips whose feature coverage is too thin to score reliably.
        cov = X.notna().sum(axis=1) / len(features)
        keep = cov >= MIN_COVERAGE
        if keep.sum() < 1000:
            skipped.append((m, f"only {keep.sum()} zips above {MIN_COVERAGE:.0%} coverage"))
            continue
        X = X.loc[keep]
        zips = snap.loc[keep, "zip"].values

        # Impute remaining NaNs with column medians (matches score_zips.py).
        X = X.fillna(X.median(numeric_only=True))
        pred = booster.predict(X)

        scored = pd.DataFrame({"zip": zips, "pred": pred})

        # Join realized returns for THIS score_month.
        m_targets = tg[tg["year_month"] == m]
        merged = scored.merge(m_targets[["zip", "realized"]], on="zip", how="inner")
        if merged.empty:
            skipped.append((m, "no realized targets"))
            continue

        merged = merged.sort_values("pred", ascending=False).reset_index(drop=True)
        n = len(merged)
        k_top = max(1, int(np.floor(n * TOP_PCT)))
        k_bot = max(1, int(np.floor(n * BOT_PCT)))
        top = merged.head(k_top)
        bot = merged.tail(k_bot)

        rows.append({
            "score_month": m,
            "n_zips_universe": n,
            "n_zips_top10": k_top,
            "n_zips_bot10": k_bot,
            "mean_realized_top10": float(top["realized"].mean()),
            "mean_realized_bot10": float(bot["realized"].mean()),
            "mean_realized_universe": float(merged["realized"].mean()),
            "spread": float(top["realized"].mean() - merged["realized"].mean()),
        })

    df = pd.DataFrame(rows).sort_values("score_month").reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"[{label}] {len(df)} months scored, {len(skipped)} skipped  ({elapsed:.1f}s)")
    if skipped:
        print(f"[{label}] skipped: {skipped[:5]}{'…' if len(skipped) > 5 else ''}")
    print(f"[{label}] -> {out_path}")
    return df


def print_summary(df: pd.DataFrame, label: str) -> None:
    """Per-model summary block printed to stdout."""
    if df.empty:
        print(f"\n=== {label}: no rows ===")
        return
    n_months = len(df)
    excess = df["mean_realized_top10"] - df["mean_realized_universe"]
    bot_excess = df["mean_realized_bot10"] - df["mean_realized_universe"]
    top_vs_bot = df["mean_realized_top10"] - df["mean_realized_bot10"]
    hit_rate = (excess > 0).mean()
    hit_rate_top_vs_bot = (top_vs_bot > 0).mean()

    print(f"\n=== {label} summary ===")
    print(f"  months covered:                  {n_months}  ({df['score_month'].min().date()} .. {df['score_month'].max().date()})")
    print(f"  mean top10 realized (annualized):     {df['mean_realized_top10'].mean()*100:+.2f}%")
    print(f"  mean bottom10 realized (annualized):  {df['mean_realized_bot10'].mean()*100:+.2f}%")
    print(f"  mean universe realized (annualized):  {df['mean_realized_universe'].mean()*100:+.2f}%")
    print(f"  avg top10 excess vs universe:    {excess.mean()*100:+.2f} pp/yr   (median {excess.median()*100:+.2f}, std {excess.std()*100:.2f})")
    print(f"  avg bot10 deficit vs universe:   {bot_excess.mean()*100:+.2f} pp/yr")
    print(f"  avg top10 - bot10 spread:        {top_vs_bot.mean()*100:+.2f} pp/yr")
    print(f"  hit rate (top10 > universe):     {hit_rate:.1%}  ({int(hit_rate*n_months)}/{n_months} months)")
    print(f"  hit rate (top10 > bot10):        {hit_rate_top_vs_bot:.1%}")

    worst = df.nsmallest(3, "spread")[["score_month", "mean_realized_top10", "mean_realized_universe", "spread"]]
    best = df.nlargest(3, "spread")[["score_month", "mean_realized_top10", "mean_realized_universe", "spread"]]
    print(f"  worst 3 months (top10 vs universe):")
    for _, r in worst.iterrows():
        print(f"    {r['score_month'].date()}  top10={r['mean_realized_top10']*100:+.2f}%  univ={r['mean_realized_universe']*100:+.2f}%  spread={r['spread']*100:+.2f}pp")
    print(f"  best 3 months (top10 vs universe):")
    for _, r in best.iterrows():
        print(f"    {r['score_month'].date()}  top10={r['mean_realized_top10']*100:+.2f}%  univ={r['mean_realized_universe']*100:+.2f}%  spread={r['spread']*100:+.2f}pp")


def main() -> None:
    wall0 = time.time()

    print("Loading feature store + targets …")
    fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
    tg = pd.read_parquet(PROCESSED / "targets_zip_month.parquet")
    print(f"  feature_store: {fs.shape}")
    print(f"  targets:       {tg.shape}")

    # Price-only headline model.
    df_price = backtest_one(
        horizon="fwd_3y",
        model_path=PROCESSED / "phase2_artifacts" / "fwd_3y__post_2012__lgbm.txt",
        features_path=PROCESSED / "phase2_artifacts" / "fwd_3y__post_2012__lgbm.features.txt",
        feature_store=fs,
        targets=tg,
        out_path=PROCESSED / "backtest_topdecile.parquet",
        label="fwd_3y (price-only)",
    )

    # Total-return variant.
    df_total = backtest_one(
        horizon="fwd_3y_total",
        model_path=PROCESSED / "phase2_artifacts_total" / "fwd_3y_total__post_2012__lgbm.txt",
        features_path=PROCESSED / "phase2_artifacts_total" / "fwd_3y_total__post_2012__lgbm.features.txt",
        feature_store=fs,
        targets=tg,
        out_path=PROCESSED / "backtest_topdecile_total.parquet",
        label="fwd_3y_total (price + rent yield)",
    )

    print_summary(df_price, "fwd_3y (price-only)")
    print_summary(df_total, "fwd_3y_total (price + rent yield)")

    print(f"\nTotal wall time: {time.time() - wall0:.1f}s")


if __name__ == "__main__":
    main()
