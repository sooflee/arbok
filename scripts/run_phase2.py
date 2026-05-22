"""Phase 2 entry: train baselines + ElasticNet + LightGBM per horizon x split.

Runs TWO passes: raw features (macro/time + spatial mixed) and CBSA-month-demeaned
features (pure spatial signal). Both are saved separately to data/processed/.

Modeling scope: TOP_100_METROS — restricts both training AND test data to zips in
the top 100 US metros by population. Smaller, denser, more relevant feature space.
"""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from arbok.config import PROCESSED
from arbok.models.run import run_all
from arbok.sources.census_metros import load_top200

TRAIN_TOP_N_METROS = 100

fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
print(f"feature store loaded: {fs.shape}")
top_metros = load_top200().head(TRAIN_TOP_N_METROS)["cbsa"].tolist()
fs = fs[fs["cbsa"].isin(top_metros)].copy()
print(f"after top-{TRAIN_TOP_N_METROS}-metro filter: {fs.shape}  ({fs['cbsa'].nunique()} cbsas)")

print("\n" + "=" * 70)
print("PASS 1: raw features (macro + spatial mixed)")
print("=" * 70)
raw_results, _ = run_all(fs, spatial_demean=False)

print("\n" + "=" * 70)
print("PASS 2: CBSA-month-demeaned features (pure within-metro spatial signal)")
print("=" * 70)
demean_results, _ = run_all(fs, spatial_demean=True)

print("\n\n=== SIDE-BY-SIDE: raw vs demeaned (decile_spread, post_2012 LightGBM) ===")
both = pd.concat([raw_results, demean_results])
key = both[(both["split"] == "post_2012") & (both["model"] == "lightgbm")]
print(key.pivot_table(index="horizon", columns="demean", values=["spearman", "decile_spread"]).round(4))
