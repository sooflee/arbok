"""Phase 2 (total-return variant): LightGBM headline models on fwd_Xy_total.

Mirrors `scripts/run_phase2.py` but targets the TOTAL-RETURN columns
(`fwd_1y_total`, `fwd_3y_total`, `fwd_5y_total`) which combine price appreciation
with ZORI-derived rent-yield carry. ZORI coverage is sparse (~11% of zip-months)
and starts ~2015, so:

  - Only the `post_2012` split is trained (pre_2008 has no ZORI overlap).
  - Training sets per horizon are ~40K rows; test sets are 80K-245K.
  - Headline grid only (LightGBM point fits) — no walk-forward / spatial CV
    / quantile fits. Total runtime budget ~10 min.

LEAKAGE GUARDS:
  prep.NON_FEATURE_EXACT only excludes the price-only fwd_* targets. We add
  the current total-return target to it for the duration of the build_xy
  call AND drop the other two total-return columns from the working frame,
  so neither the same-horizon nor cross-horizon total target can leak in as
  a feature. (See `_train_one_horizon`.)

  NOTE: fwd_Xy_total = fwd_Xy + rent_yield_t by construction, and
  rent_yield_t = 12 * zori_t / zhvi_t IS still a feature. This is fine —
  the rent yield is observable at decision time, so it's a legitimate input
  to a forward-return predictor. It does mean the total-return Spearman
  partly reflects the model copying the carry term; the decile spread and
  cross-horizon comparison are the more honest signals.

Outputs (parallel to the price-only run, distinct paths so the in-flight phase2
process is never touched):
  - data/processed/phase2_artifacts_total/fwd_Xy__post_2012__lgbm.{txt,features.txt}
  - data/processed/phase2_results_total.parquet
  - data/processed/zip_leaderboard_total.parquet         (top-50 by fwd_3y_total)

Run end-to-end: `python scripts/run_phase2_total.py`
The script trains the three horizons, scores the latest available snapshot for
fwd_3y_total, and prints a comparison vs the existing price-only top-50.
"""
from dotenv import load_dotenv
load_dotenv()

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from arbok.config import PROCESSED, TOP_N_METROS
from arbok.models.eval import evaluate
from arbok.models.lgbm import fit_lightgbm
from arbok.models.prep import SPLITS_AVAIL, build_xy
from arbok.sources.census_metros import load_top_metros

HORIZONS_TOTAL = ("fwd_1y_total", "fwd_3y_total", "fwd_5y_total")

# Train only on the post_2012 split — pre_2008 has no ZORI overlap.
SPLIT_NAME = "post_2012"

# Per task brief: lighter bagging in case 215K rows + 140 features stress memory.
LGBM_PARAMS_TOTAL: dict = {
    "bagging_fraction": 0.5,
}

# Columns the feature pipeline must NEVER see as features. prep.feature_columns
# already drops the price-only fwd_* cols + zhvi/zori. We additionally need to
# drop the total-return targets so the *other* horizon's target doesn't leak in
# (e.g. when training fwd_3y_total, fwd_1y_total and fwd_5y_total must not be
# features). build_xy calls feature_columns on the full frame, so we drop them
# from the working copy below instead of mutating shared module state.
ALL_TOTAL_TARGETS = list(HORIZONS_TOTAL)

ARTIFACTS_DIR = PROCESSED / "phase2_artifacts_total"
RESULTS_PATH = PROCESSED / "phase2_results_total.parquet"

# Scoring step: mirror scripts/score_zips.py contract but for the fwd_3y_total
# headline model. Writes a separate leaderboard so the price-only leaderboard
# (zip_leaderboard.parquet) is never overwritten.
LEADERBOARD_PATH = PROCESSED / "zip_leaderboard_total.parquet"
LEADERBOARD_TOP_N = 50
SCORE_HORIZON = "fwd_3y_total"


def _load_panel() -> pd.DataFrame:
    fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
    print(f"feature store loaded: {fs.shape}")
    top_metros = load_top_metros().head(TOP_N_METROS)["cbsa"].tolist()
    fs = fs[fs["cbsa"].isin(top_metros)].copy()
    print(f"after top-{TOP_N_METROS}-metro filter: {fs.shape}  ({fs['cbsa'].nunique()} cbsas)")
    return fs


def _train_one_horizon(fs: pd.DataFrame, target: str) -> tuple[dict, "object"]:
    """Train LightGBM for one total-return horizon on the post_2012 split."""
    print(f"\n=== {target} @ {SPLIT_NAME} ===")
    split = next(s for s in SPLITS_AVAIL if s.name == SPLIT_NAME)

    # Leakage guard: prep.NON_FEATURE_EXACT excludes the price-only fwd_*
    # targets but NOT the fwd_*_total columns. We must remove BOTH the
    # current target and the other total-return cols from the feature pool
    # before build_xy is called. Strategy: copy the frame, rename the target
    # so build_xy still finds it, drop the original (and the other totals)
    # so feature_columns can't pick them up, then translate column lists
    # back to the canonical target name in the artifacts.
    #
    # We use a sentinel name "__target_total__" that's guaranteed not to
    # collide with any real feature.
    SENTINEL = "__target_total__"
    drop_total = [c for c in ALL_TOTAL_TARGETS if c in fs.columns]
    work = fs.drop(columns=drop_total, errors="ignore").copy()
    work[SENTINEL] = fs[target].values

    # Inject SENTINEL into NON_FEATURE_EXACT for the duration of this call so
    # feature_columns excludes it. (Other phase2 processes run in separate
    # OS processes — no shared memory — so this mutation is process-local.)
    from arbok.models.prep import NON_FEATURE_EXACT
    NON_FEATURE_EXACT.add(SENTINEL)
    try:
        # Construct a temp Split-like object isn't needed; build_xy takes
        # `target` by name. We feed the sentinel name in and translate back.
        X_train, y_train, X_test, y_test, feats = build_xy(
            work, target=SENTINEL, split=split, min_coverage=0.10,
        )
    finally:
        NON_FEATURE_EXACT.discard(SENTINEL)
    print(f"  train: {len(X_train):,} rows, {len(feats)} features")
    print(f"  test:  {len(X_test):,} rows")
    if len(X_test) < 100:
        print("  SKIP (test too small)")
        return {}, None

    # Hold out the tail of train as the val set for early stopping
    # (matches the convention in run.py).
    cutoff = max(1, int(len(X_train) * 0.9))
    fit = fit_lightgbm(
        X_train.iloc[:cutoff],
        y_train.iloc[:cutoff],
        X_train.iloc[cutoff:],
        y_train.iloc[cutoff:],
        params=LGBM_PARAMS_TOTAL,
    )

    y_pred = fit.predict(X_test)
    metrics = evaluate(y_test.values, y_pred, name="lightgbm")
    print(
        f"  spearman={metrics['spearman']:.4f}  "
        f"decile_spread={metrics['decile_spread']:.4f}  "
        f"r2={metrics['r2']:.4f}  rmse={metrics['rmse']:.4f}  n_test={metrics['n']:,}"
    )
    return metrics, fit


def _score_top_zips(fs: pd.DataFrame) -> pd.DataFrame:
    """Score the latest snapshot with the fwd_3y_total booster; return top-N.

    Mirrors scripts/score_zips.py's structural choices: pick the latest month
    where >=1000 zips have >=80% feature coverage, restrict to top-N metros
    (already done in fs), impute by column-median, predict, take top-N by
    predicted fwd_3y_total annualized. We DO NOT have quantile boosters here
    (out of runtime budget) — the leaderboard parquet exposes the point
    prediction only, plus the same SHAP-driver string the price-only run uses.
    """
    model_path = ARTIFACTS_DIR / f"{SCORE_HORIZON}__{SPLIT_NAME}__lgbm.txt"
    feats_path = ARTIFACTS_DIR / f"{SCORE_HORIZON}__{SPLIT_NAME}__lgbm.features.txt"
    booster = lgb.Booster(model_file=str(model_path))
    features = feats_path.read_text().strip().split("\n")
    print(f"\n--- Scoring {SCORE_HORIZON} leaderboard ---")
    print(f"Loaded model: {model_path.name} ({len(features)} features)")

    # Latest month where >=80% of features are populated for >=1000 zips
    counts = (
        fs.groupby("year_month")
        .apply(lambda g: ((g[features].notna().sum(axis=1) >= len(features) * 0.8).sum()))
        .sort_index()
    )
    ok_months = counts[counts >= 1000]
    if ok_months.empty:
        # Fall back to most-populated month if no month clears the threshold
        latest = counts.idxmax()
        print(f"  (warn) no month has 1000+ zips at 80% coverage; using {latest}")
    else:
        latest = ok_months.index.max()
    print(f"Scoring as of: {latest}  ({counts.loc[latest]:,} zips with >=80% feature coverage)")

    snapshot = fs[fs["year_month"] == latest].copy()
    print(f"Snapshot rows: {len(snapshot):,}")

    X = snapshot[features].copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    snapshot["predicted_fwd_3y_total_annualized"] = booster.predict(X)

    ranked = (
        snapshot.sort_values("predicted_fwd_3y_total_annualized", ascending=False)
        .head(LEADERBOARD_TOP_N)
        .reset_index(drop=True)
    )

    # SHAP per-row drivers (top-3 by |contribution|) for parity with score_zips.py.
    print(f"Computing SHAP for top {LEADERBOARD_TOP_N}...")
    import shap
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(ranked[features])
    if isinstance(sv, list):
        sv = sv[0]

    def _top_drivers(row_shap, k: int = 3) -> str:
        idx = np.argsort(-np.abs(row_shap))[:k]
        return ", ".join(f"{features[i].replace('__', '.')}({row_shap[i]:+.3f})" for i in idx)

    ranked["drivers"] = [_top_drivers(sv[i]) for i in range(len(ranked))]

    cols = [
        "zip", "cbsa", "zhvi", "zori", "rent_yield_t",
        "predicted_fwd_3y_total_annualized", "drivers",
    ]
    out = ranked[cols].copy()
    out.to_parquet(LEADERBOARD_PATH, index=False)
    print(f"Saved leaderboard -> {LEADERBOARD_PATH}")
    return out


def _compare_leaderboards(total_top: pd.DataFrame) -> None:
    """Print overlap stats between the price-only top-50 and the total-return top-50."""
    price_lb_path = PROCESSED / "zip_leaderboard.parquet"
    if not price_lb_path.exists():
        print(f"\n(skip) price-only leaderboard not found at {price_lb_path}")
        return
    price_top = pd.read_parquet(price_lb_path)
    print("\n" + "=" * 70)
    print(f"COMPARISON: price-only top-{len(price_top)} vs total-return top-{len(total_top)}")
    print("=" * 70)

    price_zips = set(price_top["zip"].astype(str))
    total_zips = set(total_top["zip"].astype(str))
    overlap = price_zips & total_zips
    price_only = price_zips - total_zips
    total_only = total_zips - price_zips
    print(f"  in BOTH:                {len(overlap):3d}  ({sorted(overlap)})")
    print(f"  price-only ranking:     {len(price_only):3d}  ({sorted(price_only)})")
    print(f"  total-return ranking:   {len(total_only):3d}  ({sorted(total_only)})")

    # Quick characterization: ZHVI / ZORI / rent-yield distributions of each side.
    # The price-only leaderboard doesn't carry rent_yield_t, so compute on the fly
    # from its own zhvi/zori columns.
    def _row_stats(df: pd.DataFrame, label: str) -> None:
        zh = df["zhvi"].dropna()
        zo = df["zori"].dropna() if "zori" in df.columns else pd.Series(dtype=float)
        ry = (df["zori"] * 12 / df["zhvi"]).dropna() if "zori" in df.columns else pd.Series(dtype=float)
        print(
            f"  [{label:18s}] zhvi median={zh.median():>10,.0f}  "
            f"zori median={(zo.median() if len(zo) else float('nan')):>6,.0f}  "
            f"rent_yield median={(ry.median() if len(ry) else float('nan')):>6.4f}  "
            f"n={len(df)}"
        )

    _row_stats(price_top, "price-only top")
    _row_stats(total_top.rename(columns={"predicted_fwd_3y_total_annualized": "predicted_fwd_3y_annualized"}), "total-return top")

    # Show side-by-side metro composition (top-5 CBSAs each side).
    print("\n  Top-5 CBSAs (price-only):")
    print(price_top["cbsa"].value_counts().head(5).to_string())
    print("\n  Top-5 CBSAs (total-return):")
    print(total_top["cbsa"].value_counts().head(5).to_string())


def main() -> None:
    overall_t0 = time.time()
    fs = _load_panel()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in HORIZONS_TOTAL:
        t0 = time.time()
        metrics, fit = _train_one_horizon(fs, target)
        if not metrics or fit is None:
            continue

        # Save artifacts (mirror phase2_artifacts/ naming convention).
        # We keep the target name as-is so the leaderboard step can resolve it.
        slug = f"{target}__{SPLIT_NAME}"
        fit.model.save_model(str(ARTIFACTS_DIR / f"{slug}__lgbm.txt"))
        (ARTIFACTS_DIR / f"{slug}__lgbm.features.txt").write_text(
            "\n".join(fit.feature_names)
        )

        row = {
            "horizon": target,
            "split": SPLIT_NAME,
            "demean": False,
            **metrics,
        }
        rows.append(row)
        print(f"  saved artifacts -> {ARTIFACTS_DIR}/{slug}__lgbm.* ({time.time() - t0:.1f}s)")

    if rows:
        out = pd.DataFrame(rows)
        out.to_parquet(RESULTS_PATH, index=False)
        print(f"\nSaved results -> {RESULTS_PATH}  ({len(out)} rows)")
        print(out.round(4).to_string(index=False))
    else:
        print("\nNo successful runs — nothing written.")

    # Scoring step + comparison vs the existing price-only leaderboard.
    if (ARTIFACTS_DIR / f"{SCORE_HORIZON}__{SPLIT_NAME}__lgbm.txt").exists():
        total_top = _score_top_zips(fs)
        _compare_leaderboards(total_top)
    else:
        print(f"\n(skip) no {SCORE_HORIZON} booster — scoring skipped")

    print(f"\n=== TOTAL wall-time: {(time.time() - overall_t0) / 60:.2f} min ===")


if __name__ == "__main__":
    main()
