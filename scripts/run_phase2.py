"""Phase 2 entry: train baselines + ElasticNet + LightGBM per horizon x split.

Runs TWO passes: raw features (macro/time + spatial mixed) and CBSA-month-demeaned
features (pure spatial signal). Both are saved separately to data/processed/.

Modeling scope: TOP_100_METROS — restricts both training AND test data to zips in
the top 100 US metros by population. Smaller, denser, more relevant feature space.

Wave-2 additions (on top of the existing headline point-prediction grid):
- Walk-forward (rolling-origin) CV for the post_2012 split, all three horizons.
  Cutoffs: 2018-12-31 ... 2022-12-31, 12-month test windows. LightGBM only.
  Saved to data/processed/phase2_walkforward.parquet.
- Spatial CV (hold-out-CBSA k=5) for all three horizons, training data <= 2017-12
  so the test is still temporally honest. LightGBM only.
  Saved to data/processed/phase2_spatialcv.parquet.
- Quantile LightGBM boosters (alphas 0.1/0.5/0.9) for the headline
  fwd_3y x post_2012 cell. Saved into phase2_artifacts{,_demeaned}/ as
  fwd_3y__post_2012__lgbm_q{10,50,90}.txt + features file.
"""
from dotenv import load_dotenv
load_dotenv()

import time
from pathlib import Path

import numpy as np
import pandas as pd

from arbok.config import PROCESSED
from arbok.models.lgbm import fit_lgbm_quantiles, fit_lightgbm
from arbok.models.prep import (
    SPLITS_AVAIL,
    Split,
    build_xy,
    feature_columns,
    spatial_demean_features,
    spatial_kfold,
)
from arbok.models.run import build_walk_forward_splits, run_all, walk_forward_cv
from arbok.sources.census_metros import load_top_metros
from arbok.config import TOP_N_METROS

HORIZONS = ("fwd_1y", "fwd_3y", "fwd_5y")

# Walk-forward cutoffs (post_2012 split only). Each cutoff defines a fold
# whose test window is the *next* 12 months.
WF_CUTOFFS = [
    "2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31",
]
WF_TEST_MONTHS = 12

# Spatial-CV temporal cap: train+test both restricted to <= this date so the
# CBSA holdout is the only stress, not "future".
SPATIAL_CV_TIME_CAP = pd.Timestamp("2017-12-31")
SPATIAL_CV_N_SPLITS = 5

# LightGBM defaults for the new CV passes. Bagging at 0.5 to keep memory
# footprint manageable on the 3.7M-row panel (per task instructions).
LGBM_PARAMS_CV: dict = {
    "bagging_fraction": 0.5,
    "feature_fraction": 0.8,
    "min_data_in_leaf": 200,
}


# ---------------------------------------------------------------------------
# Step 1: existing point-prediction grid (untouched contract: phase2_results
# parquet schema + phase2_artifacts/ directory layout).
# ---------------------------------------------------------------------------


def _run_headline_grid(fs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the existing point-prediction grid for both raw + demeaned passes."""
    print("\n" + "=" * 70)
    print("PASS 1: raw features (macro + spatial mixed)")
    print("=" * 70)
    raw_results, _ = run_all(fs, spatial_demean=False)

    print("\n" + "=" * 70)
    print("PASS 2: CBSA-month-demeaned features (pure within-metro spatial signal)")
    print("=" * 70)
    demean_results, _ = run_all(fs, spatial_demean=True)
    return raw_results, demean_results


# ---------------------------------------------------------------------------
# Step 2: walk-forward CV (post_2012 only, all horizons, LightGBM).
# ---------------------------------------------------------------------------


def _lgbm_model_fn(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame) -> np.ndarray:
    """model_fn for walk_forward_cv: fit LightGBM, predict test fold."""
    # Hold out the tail of train as a val set for early stopping.
    cutoff = max(1, int(len(X_tr) * 0.9))
    fit = fit_lightgbm(
        X_tr.iloc[:cutoff],
        y_tr.iloc[:cutoff],
        X_tr.iloc[cutoff:],
        y_tr.iloc[cutoff:],
        params=LGBM_PARAMS_CV,
    )
    return fit.predict(X_te)


def _run_walk_forward(fs: pd.DataFrame, demean: bool) -> pd.DataFrame:
    """Walk-forward CV per horizon. Returns long-form DataFrame of per-fold metrics."""
    label = "demeaned" if demean else "raw"
    print(f"\n--- Walk-forward CV ({label}, post_2012 only) ---")

    work = spatial_demean_features(fs, target_cols=list(HORIZONS)) if demean else fs

    # Walk-forward operates on positional indices; reset so iloc-by-position lines up.
    work = work.reset_index(drop=True)

    rows = []
    for horizon in HORIZONS:
        # Restrict to rows where the target is observable (we never want fold
        # boundaries to cross the unobservable tail of the panel).
        target_avail = work[horizon].notna()
        sub = work.loc[target_avail].reset_index(drop=True)
        splits = build_walk_forward_splits(
            sub, time_col="year_month",
            train_until_dates=WF_CUTOFFS, test_window_months=WF_TEST_MONTHS,
        )
        if not splits:
            print(f"  {horizon}: no usable folds, skipping")
            continue

        # Pre-compute the feature column set on the *first-fold train slice*
        # (mimics the build_xy >=10% coverage filter; downstream walk_forward_cv
        # also drops zero-variance / all-NaN columns per fold).
        first_train_idx = splits[0][0]
        feats = feature_columns(sub.iloc[first_train_idx], min_coverage=0.10)
        # Drop residual target leakage just in case (paranoia).
        feats = [f for f in feats if f not in HORIZONS]

        print(f"  {horizon}: {len(splits)} folds, {len(feats)} feature cols")
        per_fold, _ = walk_forward_cv(
            sub,
            horizon=horizon,
            splits=splits,
            model_fn=_lgbm_model_fn,
            feature_cols=feats,
            impute="median",
        )
        if per_fold.empty:
            print(f"  {horizon}: walk_forward_cv returned empty")
            continue

        for rec in per_fold.reset_index().to_dict("records"):
            row = {
                "horizon": horizon,
                "split": "post_2012",
                "model": "lightgbm",
                "demean": demean,
                **rec,
            }
            # Rename eval-table metric keys to *_test for clarity vs train metrics.
            row["spearman_test"] = row.pop("spearman", float("nan"))
            row["rmse_test"] = row.pop("rmse", float("nan"))
            row["r2_test"] = row.pop("r2", float("nan"))
            row["decile_spread_test"] = row.pop("decile_spread", float("nan"))
            row["top_decile_mean_test"] = row.pop("top_decile_mean", float("nan"))
            row["bot_decile_mean_test"] = row.pop("bot_decile_mean", float("nan"))
            rows.append(row)

        # Per-horizon quick summary
        print(
            f"    spearman: mean={per_fold['spearman'].mean():.4f}, "
            f"std={per_fold['spearman'].std():.4f}, "
            f"decile_spread mean={per_fold['decile_spread'].mean():.4f}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3: spatial CV (hold-out-CBSA, train+test both <= 2017-12, LightGBM).
# ---------------------------------------------------------------------------


def _run_spatial_cv(fs: pd.DataFrame, demean: bool) -> pd.DataFrame:
    """Hold-out-CBSA k=5 CV, train+test <= SPATIAL_CV_TIME_CAP, LightGBM only."""
    label = "demeaned" if demean else "raw"
    print(f"\n--- Spatial CV ({label}, k={SPATIAL_CV_N_SPLITS}, cutoff <= {SPATIAL_CV_TIME_CAP.date()}) ---")

    work = spatial_demean_features(fs, target_cols=list(HORIZONS)) if demean else fs
    work = work[work["year_month"] <= SPATIAL_CV_TIME_CAP].copy()
    work = work.reset_index(drop=True)
    print(f"  panel <= {SPATIAL_CV_TIME_CAP.date()}: {len(work):,} rows, {work['cbsa'].nunique()} cbsas")

    if work["cbsa"].nunique() < SPATIAL_CV_N_SPLITS:
        print("  not enough cbsas, skipping")
        return pd.DataFrame()

    rows = []

    for horizon in HORIZONS:
        target_avail = work[horizon].notna()
        sub = work.loc[target_avail].reset_index(drop=True)
        if sub.empty:
            print(f"  {horizon}: no rows after target-NaN filter, skipping")
            continue

        # Recompute folds against the subset so positional indices are aligned.
        sub_folds = list(spatial_kfold(sub, n_splits=SPATIAL_CV_N_SPLITS))

        feats = feature_columns(sub.iloc[sub_folds[0][0]], min_coverage=0.10)
        feats = [f for f in feats if f not in HORIZONS]
        print(f"  {horizon}: {len(sub_folds)} folds, {len(feats)} features")

        per_fold, _ = walk_forward_cv(
            sub,
            horizon=horizon,
            splits=sub_folds,
            model_fn=_lgbm_model_fn,
            feature_cols=feats,
            impute="median",
        )
        if per_fold.empty:
            print(f"  {horizon}: spatial_kfold CV returned empty")
            continue

        for rec in per_fold.reset_index().to_dict("records"):
            row = {
                "horizon": horizon,
                "model": "lightgbm",
                "demean": demean,
                **rec,
            }
            row["spearman_test"] = row.pop("spearman", float("nan"))
            row["rmse_test"] = row.pop("rmse", float("nan"))
            row["r2_test"] = row.pop("r2", float("nan"))
            row["decile_spread_test"] = row.pop("decile_spread", float("nan"))
            row["top_decile_mean_test"] = row.pop("top_decile_mean", float("nan"))
            row["bot_decile_mean_test"] = row.pop("bot_decile_mean", float("nan"))
            rows.append(row)
        print(
            f"    spearman: mean={per_fold['spearman'].mean():.4f}, "
            f"std={per_fold['spearman'].std():.4f}, "
            f"decile_spread mean={per_fold['decile_spread'].mean():.4f}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: quantile LightGBM for the headline horizon x split (fwd_3y x post_2012).
# ---------------------------------------------------------------------------


def _fit_and_save_quantile_lgbm(fs: pd.DataFrame, demean: bool) -> None:
    """Fit alpha={0.1, 0.5, 0.9} LGBM boosters for fwd_3y x post_2012; save next to point models."""
    suffix = "_demeaned" if demean else ""
    art_dir = PROCESSED / f"phase2_artifacts{suffix}"
    art_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Quantile LightGBM (fwd_3y, post_2012, {'demeaned' if demean else 'raw'}) ---")
    work = spatial_demean_features(fs, target_cols=list(HORIZONS)) if demean else fs

    post_2012_split = next(s for s in SPLITS_AVAIL if s.name == "post_2012")
    X_train, y_train, X_test, y_test, feats = build_xy(
        work, target="fwd_3y", split=post_2012_split, min_coverage=0.10,
    )
    print(f"  train: {len(X_train):,} rows, {len(feats)} features")
    if len(X_train) < 100:
        print("  too few rows, skipping")
        return

    # Tail of train as validation set for early stopping.
    cutoff = max(1, int(len(X_train) * 0.9))
    qmodels = fit_lgbm_quantiles(
        X_train.iloc[:cutoff],
        y_train.iloc[:cutoff],
        alphas=(0.1, 0.5, 0.9),
        X_val=X_train.iloc[cutoff:],
        y_val=y_train.iloc[cutoff:],
        params=LGBM_PARAMS_CV,
    )

    for alpha, fit in qmodels.items():
        q = int(round(alpha * 100))
        path = art_dir / f"fwd_3y__post_2012__lgbm_q{q}.txt"
        fit.model.save_model(str(path))
        (art_dir / f"fwd_3y__post_2012__lgbm_q{q}.features.txt").write_text(
            "\n".join(fit.feature_names)
        )
        print(f"    saved q{q} -> {path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    overall_t0 = time.time()

    fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
    print(f"feature store loaded: {fs.shape}")
    top_metros = load_top_metros().head(TOP_N_METROS)["cbsa"].tolist()
    fs = fs[fs["cbsa"].isin(top_metros)].copy()
    print(f"after top-{TOP_N_METROS}-metro filter: {fs.shape}  ({fs['cbsa'].nunique()} cbsas)")

    timings: dict[str, float] = {}

    # --- existing point-prediction grid (raw + demeaned) ---
    step_t0 = time.time()
    raw_results, demean_results = _run_headline_grid(fs)
    timings["point_grid"] = time.time() - step_t0

    print("\n\n=== SIDE-BY-SIDE: raw vs demeaned (decile_spread, post_2012 LightGBM) ===")
    both = pd.concat([raw_results, demean_results])
    key = both[(both["split"] == "post_2012") & (both["model"] == "lightgbm")]
    print(key.pivot_table(index="horizon", columns="demean", values=["spearman", "decile_spread"]).round(4))

    # --- walk-forward CV (post_2012 only, all horizons) ---
    step_t0 = time.time()
    wf_raw = _run_walk_forward(fs, demean=False)
    wf_dem = _run_walk_forward(fs, demean=True)
    wf = pd.concat([wf_raw, wf_dem], ignore_index=True)
    if not wf.empty:
        wf_path = PROCESSED / "phase2_walkforward.parquet"
        wf.to_parquet(wf_path, index=False)
        print(f"\nSaved walk-forward -> {wf_path}  ({len(wf)} rows)")
    timings["walk_forward"] = time.time() - step_t0

    # --- spatial CV (k=5, train+test <= 2017-12) ---
    step_t0 = time.time()
    sp_raw = _run_spatial_cv(fs, demean=False)
    sp_dem = _run_spatial_cv(fs, demean=True)
    sp = pd.concat([sp_raw, sp_dem], ignore_index=True)
    if not sp.empty:
        sp_path = PROCESSED / "phase2_spatialcv.parquet"
        sp.to_parquet(sp_path, index=False)
        print(f"\nSaved spatial-CV -> {sp_path}  ({len(sp)} rows)")
    timings["spatial_cv"] = time.time() - step_t0

    # --- quantile boosters for fwd_3y x post_2012 (raw + demeaned) ---
    step_t0 = time.time()
    _fit_and_save_quantile_lgbm(fs, demean=False)
    _fit_and_save_quantile_lgbm(fs, demean=True)
    timings["quantile_lgbm"] = time.time() - step_t0

    print("\n\n=== Phase 2 wall-time ===")
    for k, v in timings.items():
        print(f"  {k:18s}  {v / 60:6.2f} min")
    print(f"  {'TOTAL':18s}  {(time.time() - overall_t0) / 60:6.2f} min")


if __name__ == "__main__":
    main()
