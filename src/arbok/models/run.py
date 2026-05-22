"""Phase 2 orchestrator: train baselines + ElasticNet + LightGBM per horizon, both splits."""
from __future__ import annotations

import pickle

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
