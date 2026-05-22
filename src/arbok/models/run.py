"""Phase 2 orchestrator: train baselines + ElasticNet + LightGBM per horizon, both splits."""
from __future__ import annotations

import pickle
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from arbok.config import PROCESSED

ARTIFACTS = PROCESSED / "phase2_artifacts"
from arbok.models.baselines import hedonic_predictor, metro_mean_predictor, overall_mean_predictor
from arbok.models.elastic import fit_elasticnet
from arbok.models.eval import evaluate, results_table
from arbok.models.lgbm import fit_lightgbm
from arbok.models.prep import SPLITS_AVAIL, Split, build_xy, spatial_demean_features

# fwd_1y = short, fwd_3y = medium, fwd_5y = long
DEFAULT_HORIZONS = ("fwd_1y", "fwd_3y", "fwd_5y")


def run_one(
    fs: pd.DataFrame,
    target: str,
    split: Split,
    min_coverage: float = 0.10,
) -> tuple[pd.DataFrame, dict]:
    """Train the full slate of models for one (target, split) combination.

    Returns a results-table DataFrame + an artifacts dict (with fitted models and
    feature-importance tables for downstream inspection).
    """
    print(f"\n=== {target} @ {split.name} ===")
    X_train, y_train, X_test, y_test, feats = build_xy(fs, target, split, min_coverage)
    train = fs.loc[X_train.index]
    test = fs.loc[X_test.index]
    print(f"  train: {len(X_train):,} rows, {len(feats)} features")
    print(f"  test:  {len(X_test):,} rows")
    if len(X_test) < 100:
        print("  SKIP (test too small)")
        return pd.DataFrame(), {}

    results = []
    artifacts: dict = {"feats": feats}

    # Baselines
    results.append(evaluate(y_test.values, overall_mean_predictor(train, test, target).y_pred, "overall_mean"))
    results.append(evaluate(y_test.values, metro_mean_predictor(train, test, target).y_pred, "metro_mean"))
    results.append(evaluate(y_test.values, hedonic_predictor(X_train, y_train, X_test).y_pred, "hedonic_acs"))

    # ElasticNet
    elastic_fit = fit_elasticnet(X_train, y_train)
    artifacts["elastic"] = elastic_fit
    results.append(evaluate(y_test.values, elastic_fit.predict(X_test), "elasticnet"))

    # LightGBM (with a held-out tail for early stopping)
    cutoff = int(len(X_train) * 0.9)
    lgbm_fit = fit_lightgbm(X_train.iloc[:cutoff], y_train.iloc[:cutoff], X_train.iloc[cutoff:], y_train.iloc[cutoff:])
    artifacts["lgbm"] = lgbm_fit
    results.append(evaluate(y_test.values, lgbm_fit.predict(X_test), "lightgbm"))

    tbl = results_table(results)
    print(tbl.to_string())
    return tbl, artifacts


def run_all(
    fs: pd.DataFrame,
    horizons: tuple[str, ...] = DEFAULT_HORIZONS,
    splits: tuple[Split, ...] = SPLITS_AVAIL,
    min_coverage: float = 0.10,
    save: bool = True,
    spatial_demean: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the full grid; return a long-form results frame keyed by (horizon, split).

    When `spatial_demean=True`, both features and targets are CBSA-month-demeaned BEFORE
    training. Targets become "excess return vs metro" and features become "this zip's
    deviation from its metro neighbors." Useful for picking BETWEEN zips inside a metro;
    NOT useful for picking BETWEEN metros (the macro signal is zeroed out by construction).
    """
    if spatial_demean:
        print("\n*** SPATIAL DEMEAN ENABLED: targets and features demeaned by (cbsa, year_month) ***")
        fs = spatial_demean_features(fs, target_cols=list(horizons))

    rows = []
    artifacts: dict[tuple[str, str], dict] = {}
    for horizon in horizons:
        for split in splits:
            tbl, arts = run_one(fs, horizon, split, min_coverage)
            if not tbl.empty:
                for row in tbl.reset_index().to_dict("records"):
                    rows.append({"horizon": horizon, "split": split.name,
                                 "demean": spatial_demean, **row})
                artifacts[(horizon, split.name)] = arts
    out = pd.DataFrame(rows)
    if save:
        suffix = "_demeaned" if spatial_demean else ""
        path = PROCESSED / f"phase2_results{suffix}.parquet"
        out.to_parquet(path, index=False)
        print(f"\nSaved results -> {path}")
        art_dir = PROCESSED / f"phase2_artifacts{suffix}"
        art_dir.mkdir(parents=True, exist_ok=True)
        for (horizon, split_name), arts in artifacts.items():
            slug = f"{horizon}__{split_name}"
            if "lgbm" in arts:
                arts["lgbm"].model.save_model(str(art_dir / f"{slug}__lgbm.txt"))
                (art_dir / f"{slug}__lgbm.features.txt").write_text(
                    "\n".join(arts["lgbm"].feature_names)
                )
            if "elastic" in arts:
                with open(art_dir / f"{slug}__elastic.pkl", "wb") as f:
                    pickle.dump(arts["elastic"], f)
        print(f"Saved trained models -> {art_dir}/")
    return out, artifacts


# ---------------------------------------------------------------------------
# Walk-forward (rolling-origin) cross-validation
# ---------------------------------------------------------------------------


SplitPair = tuple[np.ndarray, np.ndarray]


def build_walk_forward_splits(
    df: pd.DataFrame,
    time_col: str,
    train_until_dates: Sequence[str | pd.Timestamp],
    test_window_months: int = 12,
) -> list[SplitPair]:
    """Construct rolling-origin (train_idx, test_idx) splits from a panel.

    For each cutoff `t` in `train_until_dates`:
      - train_idx = positional indices where df[time_col] <= t
      - test_idx  = positional indices where t < df[time_col] <= t + test_window_months

    Indices are POSITIONAL (0..len(df)-1) — usable with `.iloc` and with
    DataFrames that have been reset / are otherwise positionally addressed.
    The dataframe is NOT mutated.

    Example (1-year test windows):
        cutoffs = ["2018-12-31", "2019-12-31", "2020-12-31",
                   "2021-12-31", "2022-12-31"]
        splits  = build_walk_forward_splits(df, "year_month", cutoffs, 12)
    """
    if time_col not in df.columns:
        raise ValueError(f"{time_col!r} not in df")
    times = pd.to_datetime(df[time_col])
    pos = np.arange(len(df))
    splits: list[SplitPair] = []
    for cutoff in train_until_dates:
        t = pd.Timestamp(cutoff)
        test_end = t + pd.DateOffset(months=int(test_window_months))
        train_mask = times <= t
        test_mask = (times > t) & (times <= test_end)
        train_idx = pos[train_mask.values]
        test_idx = pos[test_mask.values]
        if len(train_idx) == 0 or len(test_idx) == 0:
            # Skip empty folds (caller can inspect length to detect)
            continue
        splits.append((train_idx, test_idx))
    return splits


def walk_forward_cv(
    df: pd.DataFrame,
    horizon: str,
    splits: Sequence[SplitPair],
    model_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    feature_cols: list[str] | None = None,
    impute: str | None = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin temporal CV.

    Parameters
    ----------
    df : panel including features + horizon target column
    horizon : target column name (e.g. "fwd_1y")
    splits : sequence of (train_idx, test_idx) positional-index tuples; use
        `build_walk_forward_splits` to construct
    model_fn : callable (X_train, y_train, X_test) -> y_pred. Models can pick
        their own params. Receives imputed, numeric-only X.
    metric_fns : optional dict of {name: fn(y_true, y_pred) -> float}. If
        None, defaults to the standard suite from `arbok.models.eval.evaluate`
        and the per-fold frame will contain its full set of columns.
    feature_cols : optional explicit feature column list. If None, all
        numeric non-target columns minus the standard NON_FEATURE_* set are
        used (see `arbok.models.prep.feature_columns`).
    impute : "median" (default) or None. Median imputation uses train-fold
        statistics only (no leakage).

    Returns
    -------
    per_fold : DataFrame indexed by fold with one row per fold and one column
        per metric.
    summary : DataFrame indexed by metric with columns
        [mean, std, q025, q975, n_folds].
    """
    from arbok.models.prep import feature_columns  # local import to avoid cycle

    if horizon not in df.columns:
        raise ValueError(f"{horizon!r} not in df")

    rows = []
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        sub_train = df.iloc[tr_idx]
        sub_test = df.iloc[te_idx]

        # Drop rows where target is missing on each side
        tr_mask = sub_train[horizon].notna()
        te_mask = sub_test[horizon].notna()
        sub_train = sub_train.loc[tr_mask]
        sub_test = sub_test.loc[te_mask]
        if len(sub_train) == 0 or len(sub_test) == 0:
            continue

        feats = list(feature_cols) if feature_cols is not None else feature_columns(sub_train)
        feats = [f for f in feats if f != horizon and f in sub_test.columns]

        X_tr = sub_train[feats]
        y_tr = sub_train[horizon].astype(float)
        X_te = sub_test[feats]
        y_te = sub_test[horizon].astype(float).values

        if impute == "median":
            med = X_tr.median(numeric_only=True)
            X_tr = X_tr.fillna(med)
            X_te = X_te.fillna(med)
            still_na = X_tr.columns[X_tr.isna().any()].tolist()
            if still_na:
                X_tr = X_tr.drop(columns=still_na)
                X_te = X_te.drop(columns=still_na, errors="ignore")
            nunique = X_tr.nunique(dropna=False)
            zero_var = nunique[nunique <= 1].index.tolist()
            if zero_var:
                X_tr = X_tr.drop(columns=zero_var)
                X_te = X_te.drop(columns=zero_var, errors="ignore")

        y_pred = np.asarray(model_fn(X_tr, y_tr, X_te), dtype=float)
        y_true = np.asarray(y_te, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

        if metric_fns is None:
            metrics = evaluate(y_true, y_pred, name=f"fold_{fold_i}")
            metrics.pop("model", None)
        else:
            metrics = {name: float(fn(y_true, y_pred)) for name, fn in metric_fns.items()}
            metrics["n"] = int(len(y_true))

        metrics["fold"] = fold_i
        metrics["n_train"] = int(len(sub_train))
        metrics["n_test"] = int(len(sub_test))
        rows.append(metrics)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    per_fold = pd.DataFrame(rows).set_index("fold")

    # Summary stats across folds (numeric cols only)
    numeric = per_fold.select_dtypes(include=[np.number])
    summary = pd.DataFrame(
        {
            "mean": numeric.mean(),
            "std": numeric.std(ddof=1),
            "q025": numeric.quantile(0.025),
            "q975": numeric.quantile(0.975),
            "n_folds": numeric.notna().sum().astype(int),
        }
    )
    return per_fold, summary
