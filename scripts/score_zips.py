"""Score every zip in the latest available month, rank by predicted fwd_3y.

Uses the post_2012 LightGBM model (the strongest in Phase 2 results). For each
top zip, surfaces the three features that contributed most to its prediction
(per-row SHAP).
"""
from dotenv import load_dotenv
load_dotenv()

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from arbok.config import PROCESSED

HORIZON = "fwd_3y"
SPLIT = "post_2012"
TOP_N = 50
TOP_METROS_FILTER = 100  # restrict leaderboard to zips in the top-N metros by population

ARTIFACTS = PROCESSED / "phase2_artifacts"
MODEL_PATH = ARTIFACTS / f"{HORIZON}__{SPLIT}__lgbm.txt"
FEATURES_PATH = ARTIFACTS / f"{HORIZON}__{SPLIT}__lgbm.features.txt"

# Load model + feature list
booster = lgb.Booster(model_file=str(MODEL_PATH))
features = FEATURES_PATH.read_text().strip().split("\n")
print(f"Loaded model: {MODEL_PATH.name}, {len(features)} features")

# Load feature store
fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
print(f"Feature store: {fs.shape}")

# Find the latest year_month where >=80% of the model's features are populated for >=1000 zips
counts = (
    fs.groupby("year_month")
    .apply(lambda g: ((g[features].notna().sum(axis=1) >= len(features) * 0.8).sum()))
    .sort_index()
)
ok_months = counts[counts >= 1000]
latest = ok_months.index.max()
print(f"Scoring as of: {latest}  ({counts.loc[latest]:,} zips with >=80% feature coverage)")

snapshot = fs[fs["year_month"] == latest].copy()
print(f"Snapshot rows: {len(snapshot):,}")

# Restrict to top-N metros so leaderboard surfaces zips in larger cities.
from arbok.sources.census_metros import load_top200
top_metros = load_top200().head(TOP_METROS_FILTER)
snapshot = snapshot[snapshot["cbsa"].isin(top_metros["cbsa"])].copy()
print(f"After top-{TOP_METROS_FILTER}-metro filter: {len(snapshot):,} rows")

# Impute features with column medians (no leakage — this is just for scoring)
X = snapshot[features].copy()
med = X.median(numeric_only=True)
X = X.fillna(med)

# Predict
snapshot["predicted_fwd_3y_annualized"] = booster.predict(X)

# Per-row SHAP for top-N
ranked = snapshot.sort_values("predicted_fwd_3y_annualized", ascending=False).head(TOP_N).reset_index(drop=True)
print(f"\nComputing SHAP for top {TOP_N}...")
explainer = shap.TreeExplainer(booster)
sv = explainer.shap_values(ranked[features])
if isinstance(sv, list):
    sv = sv[0]

# Top-3 drivers per zip (signed contribution; positive = pushed prediction up)
def top_drivers(row_shap, k=3):
    idx = np.argsort(-np.abs(row_shap))[:k]
    return [(features[i], float(row_shap[i])) for i in idx]

drivers = [top_drivers(sv[i]) for i in range(len(ranked))]
ranked["drivers"] = [", ".join(f"{f.replace('__','.')}({v:+.3f})" for f, v in d) for d in drivers]

# Save + print
out = ranked[["zip", "cbsa", "zhvi", "zori", "predicted_fwd_3y_annualized", "drivers"]]
out.to_parquet(PROCESSED / "zip_leaderboard.parquet", index=False)
print(f"\nSaved leaderboard -> {PROCESSED / 'zip_leaderboard.parquet'}")

print(f"\n=== Top {min(20, TOP_N)} zips by predicted {HORIZON} annualized return (as of {latest}) ===")
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 220)
display = out.head(20).copy()
display["zhvi"] = display["zhvi"].apply(lambda x: f"${x:>10,.0f}" if pd.notna(x) else "n/a")
display["zori"] = display["zori"].apply(lambda x: f"${x:>5,.0f}" if pd.notna(x) else "n/a")
display["predicted_fwd_3y_annualized"] = display["predicted_fwd_3y_annualized"].apply(lambda x: f"{x*100:+.2f}%")
print(display.to_string(index=False))
