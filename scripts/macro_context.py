"""Macro context for the report.

Produces two artifacts that feed into the new "5b. Macro context" section
of the report:

  1. ``data/processed/timing_universe_mean.parquet``
       Per-month, the universe-mean *predicted* fwd_3y annualized return
       (post_2012 LightGBM scored over every top-100-metro ZIP using
       median-imputed features that month) and the *realized* fwd_3y
       universe mean from ``targets_zip_month.parquet``.

  2. ``data/processed/hedge_comparison.parquet``
       Long-format monthly series of nominal + real returns for a set of
       inflation-hedge candidates (S&P 500, broad US equity, gold, REITs,
       BTC, 10Y nominal yield, 10Y TIPS yield) and US residential RE
       proxied by the equal-weight Zillow ZHVI top-100-metro index.

Designed to be safe to re-run — network calls are cached on disk (FRED via
the existing ``arbok.sources.fred`` cache, yfinance via Pandas pickles
under ``data/raw/yfinance/``). If a series isn't available it's skipped
and the gap is noted at the end of the run.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from arbok.config import PROCESSED, RAW
from arbok.sources.fred import fetch_series as fred_fetch

warnings.filterwarnings("ignore", category=FutureWarning)

ARTIFACTS = PROCESSED / "phase2_artifacts"
HORIZON = "fwd_3y"
SPLIT = "post_2012"
MODEL_PATH = ARTIFACTS / f"{HORIZON}__{SPLIT}__lgbm.txt"
FEATURES_PATH = ARTIFACTS / f"{HORIZON}__{SPLIT}__lgbm.features.txt"

YF_CACHE = RAW / "yfinance"
YF_CACHE.mkdir(parents=True, exist_ok=True)

TIMING_OUT = PROCESSED / "timing_universe_mean.parquet"
HEDGE_OUT = PROCESSED / "hedge_comparison.parquet"


# ---------------------------------------------------------------------------
# Analysis A: timing — universe-mean predicted vs realized fwd_3y per month.
# ---------------------------------------------------------------------------

def _load_model_and_features() -> tuple[lgb.Booster, list[str]]:
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    features = FEATURES_PATH.read_text().strip().split("\n")
    return booster, features


def build_timing_panel(
    feature_store: pd.DataFrame,
    booster: lgb.Booster,
    features: list[str],
    start_month: str = "2013-01-01",
) -> pd.DataFrame:
    """Score the universe at every month, return predicted vs realized means.

    The post_2012 LightGBM was trained on data ≤ 2017-12 and is being
    used here as a deployed model — at every month from 2013-01 onward
    we score the available universe (top-100-metro ZIPs) with that
    month's features and take the mean. We pair with the realized
    fwd_3y universe mean from the same panel (NaN once t+3y falls
    beyond the panel's end).

    Imputation strategy: prefer the within-month column median, and fall
    back to the **panel-wide column median** when a column is entirely
    missing in that month (which happens for recent months on ACS-class
    features that publish with multi-year lag — score_zips.py does the
    same thing implicitly because it only scores the most-recent month
    after coverage filtering).
    """
    fs = feature_store
    fs = fs[fs["year_month"] >= pd.Timestamp(start_month)].copy()

    # Panel-wide medians (the safety net for fully-missing months).
    panel_median = fs[features].median(numeric_only=True)

    # Score every month with ≥500 ZIPs. Don't require feature coverage at the
    # row level — LightGBM handles NaNs natively; we just want to avoid the
    # situation where a column is entirely missing (which would degenerate to
    # NaN imputation -> bias). Use within-month median when available, panel
    # median otherwise.
    months = sorted(fs["year_month"].unique())
    print(f"  Scoring {len(months)} months "
          f"({months[0].date()} → {months[-1].date()})")

    rows = []
    for m in months:
        snap = fs[fs["year_month"] == m]
        if len(snap) < 500:
            continue
        X = snap[features].copy()
        within_med = X.median(numeric_only=True)
        fill = within_med.where(within_med.notna(), panel_median)
        X = X.fillna(fill)
        # If any column still has NaN (panel median itself is NaN), drop it.
        X = X.fillna(0.0)
        pred = booster.predict(X)
        rows.append({
            "year_month": m,
            "n_zips": int(len(snap)),
            "predicted_universe_mean": float(np.mean(pred)),
            "predicted_universe_median": float(np.median(pred)),
            "predicted_universe_p10": float(np.quantile(pred, 0.10)),
            "predicted_universe_p90": float(np.quantile(pred, 0.90)),
        })
    pred_df = pd.DataFrame(rows)

    # Realized universe-mean fwd_3y — same universe (top-100 metros only,
    # which the feature store is already restricted to).
    realized = (
        fs.groupby("year_month")["fwd_3y"]
          .mean()
          .rename("realized_universe_mean")
          .reset_index()
    )

    out = pred_df.merge(realized, on="year_month", how="left")
    return out


# ---------------------------------------------------------------------------
# Analysis B: hedge comparison.
# ---------------------------------------------------------------------------

def _yf_monthly(symbol: str, label: str) -> pd.Series | None:
    """Download monthly close from Yahoo (auto-adjusted), cache on disk.

    Returns a Series indexed by month-start Timestamps, named ``label``.
    Returns None if the download fails (caller can skip the asset).
    """
    cache = YF_CACHE / f"{symbol.replace('=','_').replace('^','_')}.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            if not df.empty:
                return df["close"].rename(label)
        except Exception:
            pass
    try:
        import yfinance as yf
    except ImportError:
        print(f"    yfinance not installed; cannot fetch {symbol}")
        return None
    try:
        df = yf.download(
            symbol, start="2000-01-01",
            interval="1mo", progress=False, auto_adjust=True,
        )
        if df.empty:
            print(f"    {symbol}: empty download")
            return None
        close = df["Close"].squeeze()
        close.index = close.index.to_period("M").to_timestamp()
        close = close.dropna()
        out = pd.DataFrame({"close": close})
        out.to_parquet(cache)
        return out["close"].rename(label)
    except Exception as e:
        print(f"    {symbol}: failed -> {e}")
        return None


def _fred_monthly(series_id: str, label: str) -> pd.Series | None:
    """Fetch a FRED series, downsample to month-start last-of-month."""
    try:
        df = fred_fetch(series_id)
    except Exception as e:
        print(f"    FRED {series_id}: failed -> {e}")
        return None
    s = df.set_index("date")["value"].sort_index()
    s = s.resample("ME").last()
    s.index = s.index.to_period("M").to_timestamp()
    return s.rename(label)


def _zhvi_national_series() -> pd.Series:
    """Equal-weighted ZHVI across the top-100-metro panel as our RE price index.

    Uses the feature-store ZHVI (already top-100-metro scoped) so the
    series is consistent with how the model views the universe.
    """
    fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet",
                         columns=["year_month", "zhvi"])
    s = (
        fs.groupby("year_month")["zhvi"].mean()
          .dropna().sort_index()
    )
    return s.rename("RE_ZHVI_top100")


def build_hedge_panel() -> tuple[pd.DataFrame, list[str]]:
    """Return long-format asset × month panel of nominal + real returns.

    Columns: asset, year_month, level, nominal_return, real_return,
             cum_nominal, cum_real, ann_3y_nominal, ann_3y_real,
             ann_5y_nominal, ann_5y_real.
    """
    gaps: list[str] = []

    print("  Pulling CPI from FRED (CPIAUCSL)…")
    cpi = _fred_monthly("CPIAUCSL", "CPI")
    if cpi is None:
        raise RuntimeError("CPI is required for real-return computation.")

    series: dict[str, pd.Series] = {}

    # FRED rates (yields, kept as context — not used as return series).
    print("  Pulling 10Y nominal + TIPS yields from FRED…")
    for sid, label in (("DGS10", "Treasury_10Y_yield"),
                       ("DFII10", "TIPS_10Y_yield")):
        s = _fred_monthly(sid, label)
        if s is None:
            gaps.append(label)
        else:
            series[label] = s

    # Equity & gold & REIT via yfinance.
    print("  Pulling equity / gold / REIT / BTC from Yahoo Finance…")
    yf_map = [
        ("^GSPC", "SP500"),                # S&P 500 price index, 2000+
        ("^SP500TR", "SP500_TR"),          # S&P 500 total-return, 2000+
        ("VTI", "VTI_broad_market"),       # broad US equity proxy (post-Wilshire-FRED-delisting)
        ("GC=F", "Gold_futures"),          # gold front-month future
        ("IYR", "REIT_iShares_IYR"),       # iShares US RE ETF — long history
        ("VNQ", "REIT_Vanguard_VNQ"),      # alt REIT TR proxy, 2004+
        ("BTC-USD", "Bitcoin"),
    ]
    for sym, label in yf_map:
        s = _yf_monthly(sym, label)
        if s is None or s.empty:
            gaps.append(label)
            continue
        series[label] = s

    # Zillow ZHVI top-100 mean as the residential-RE price proxy.
    print("  Building RE proxy from Zillow ZHVI top-100 mean…")
    zhvi = _zhvi_national_series()
    series["RE_ZHVI_top100"] = zhvi

    # Build wide frame.
    wide = pd.DataFrame(series)
    wide = wide.dropna(how="all")

    # CPI as multiplier for real-return computation.
    cpi_aligned = cpi.reindex(wide.index).ffill()

    # Per-asset compute returns + rolling annualized + real.
    asset_frames: list[pd.DataFrame] = []
    # Asset list: include yields (DGS10, DFII10) as context but no return calc.
    return_assets = [c for c in wide.columns
                     if c not in {"Treasury_10Y_yield", "TIPS_10Y_yield"}]
    for asset in return_assets:
        level = wide[asset].dropna()
        if level.empty:
            continue
        ret = level.pct_change()
        # Real monthly return: (1 + nom)/(1 + cpi growth) - 1
        cpi_ret = cpi_aligned.reindex(level.index).pct_change()
        real_ret = (1 + ret) / (1 + cpi_ret) - 1

        # Rolling 3y / 5y annualized geometric mean.
        def _rolling_ann(s: pd.Series, n: int) -> pd.Series:
            # geometric annualization: (prod(1+r))^(12/n) - 1 over n months.
            log_r = np.log1p(s)
            roll = log_r.rolling(n).sum()
            ann = np.expm1(roll * (12.0 / n))
            return ann

        ann_3y = _rolling_ann(ret, 36)
        ann_5y = _rolling_ann(ret, 60)
        ann_3y_real = _rolling_ann(real_ret, 36)
        ann_5y_real = _rolling_ann(real_ret, 60)

        # Cumulative real index (normalised so the first non-null = 100).
        cum_nom = (1 + ret.fillna(0)).cumprod() * 100
        cum_real = (1 + real_ret.fillna(0)).cumprod() * 100

        df = pd.DataFrame({
            "asset": asset,
            "year_month": level.index,
            "level": level.values,
            "nominal_return": ret.values,
            "real_return": real_ret.values,
            "cum_nominal": cum_nom.values,
            "cum_real": cum_real.values,
            "ann_3y_nominal": ann_3y.values,
            "ann_3y_real": ann_3y_real.values,
            "ann_5y_nominal": ann_5y.values,
            "ann_5y_real": ann_5y_real.values,
        })
        asset_frames.append(df)

    long_df = pd.concat(asset_frames, ignore_index=True)
    return long_df, gaps


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Analysis A: Is now a good time to buy? ===")
    booster, features = _load_model_and_features()
    print(f"  Loaded model: {MODEL_PATH.name}, {len(features)} features")
    fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
    timing = build_timing_panel(fs, booster, features)
    timing.to_parquet(TIMING_OUT, index=False)
    print(f"  → {TIMING_OUT}  ({len(timing)} rows)")
    if not timing.empty:
        latest = timing.iloc[-1]
        print(f"  latest month: {latest['year_month'].date()}  "
              f"predicted={latest['predicted_universe_mean']:+.4f}  "
              f"realized={latest['realized_universe_mean'] if pd.notna(latest['realized_universe_mean']) else 'NA'}")

    print()
    print("=== Analysis B: Real estate vs other inflation hedges ===")
    hedge, gaps = build_hedge_panel()
    hedge.to_parquet(HEDGE_OUT, index=False)
    print(f"  → {HEDGE_OUT}  ({len(hedge):,} rows × {len(hedge['asset'].unique())} assets)")
    if gaps:
        print(f"  Missing series (skipped): {gaps}")

    # Headline numbers for log.
    print()
    print("  Median 3y real return (rolling, since 2000):")
    summary = (
        hedge.dropna(subset=["ann_3y_real"])
             .groupby("asset")["ann_3y_real"].median()
             .sort_values(ascending=False)
    )
    for asset, val in summary.items():
        print(f"    {asset:30s}  {val*100:+.2f}%")


if __name__ == "__main__":
    main()
