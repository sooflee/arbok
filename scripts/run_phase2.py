"""Phase 2 entry: train baselines + ElasticNet + LightGBM per horizon x split."""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from arbok.config import PROCESSED
from arbok.models.run import run_all

fs = pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
print(f"feature store loaded: {fs.shape}")

results, artifacts = run_all(fs)
print("\n\n=== SUMMARY (sorted by decile_spread within each horizon x split) ===")
print(results.sort_values(["horizon", "split", "decile_spread"], ascending=[True, True, False]).to_string(index=False))

# Print top SHAP features per LGBM model
print("\n\n=== Top SHAP features per horizon x split (LightGBM) ===")
for (horizon, split_name), arts in artifacts.items():
    lgbm = arts.get("lgbm")
    if lgbm is None:
        continue
    print(f"\n--- {horizon} @ {split_name} ---")
    fs_train_idx = fs[fs[f"is_train_{split_name}"].fillna(False) & fs[horizon].notna()].index
    X = fs.loc[fs_train_idx, arts["feats"]].fillna(fs.loc[fs_train_idx, arts["feats"]].median(numeric_only=True))
    shap_top = lgbm.shap_top_k(X, k=15)
    print(shap_top.to_string(index=False))
