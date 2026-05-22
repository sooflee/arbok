# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # 20 — Feature store assembly
#
# Builds the wide `(zip, year_month, ...features..., ...targets...)` parquet
# by joining every registered source onto the Zillow panel. Run after notebooks
# `00_panel_targets.py` and `10_metros.py`, and after each source's
# `build_and_save()` has produced its processed parquet.

# %%
import pandas as pd

from arbok.config import PROCESSED, SPLITS
from arbok.features.assembler import (
    SOURCE_SPECS,
    VINTAGE_LAGS,
    build_and_save,
    load_feature_store,
)
from arbok.features.crosswalks import load_zip_county, load_zip_tract
from arbok.panel import build_panel
from arbok.sources.census_metros import load_top200
from arbok.sources.hud_crosswalk import dominant_cbsa_per_zip, load_zip_cbsa_crosswalk
from arbok.sources.zillow import fetch_zhvi, fetch_zori
from arbok.targets import add_split_label, compute_forward_returns

# %% [markdown]
# ## 1. Inputs

# %%
top200 = load_top200()
zhvi = fetch_zhvi()
zori = fetch_zori()

XWALK_YEAR = 2024
XWALK_QUARTER = 4
zip_cbsa = load_zip_cbsa_crosswalk(XWALK_YEAR, XWALK_QUARTER)
zip_tract = load_zip_tract(XWALK_YEAR, XWALK_QUARTER)
zip_county = load_zip_county(XWALK_YEAR, XWALK_QUARTER)
print(f"top200: {top200.shape}, ZHVI: {zhvi.shape}, ZORI: {zori.shape}")

# %% [markdown]
# ## 2. Panel + targets

# %%
panel = build_panel(zhvi, zori, dominant_cbsa_per_zip(zip_cbsa), top200)
targets = compute_forward_returns(panel)
targets = add_split_label(targets)
print(f"panel: {panel.shape}, targets: {targets.shape}")

# %% [markdown]
# ## 3. Inspect the vintage-lag config

# %%
pd.DataFrame([
    {"source": s.name, "level": s.level, "grain": s.grain, "lag_months": s.lag_months, "namespace": s.namespace}
    for s in SOURCE_SPECS
])

# %% [markdown]
# ## 4. Build the feature store
#
# Skips any source whose processed parquet isn't on disk yet. Run each source's
# `build_and_save()` once before this cell for full coverage.

# %%
fs = build_and_save(panel, targets, zip_tract, zip_county)
print(f"feature store: {fs.shape[0]:,} rows x {fs.shape[1]} cols")

# %% [markdown]
# ## 5. Sanity checks

# %%
fs = load_feature_store()
print("\nDtypes:")
print(fs.dtypes.value_counts())
print("\nNon-null share per feature column (top 20 most-populated):")
print(fs.notna().mean().sort_values(ascending=False).head(20))

print("\nSplit row counts:")
for split_name in SPLITS:
    for kind in ("train", "test"):
        col = f"is_{kind}_{split_name}"
        if col in fs.columns:
            print(f"  {col}: {int(fs[col].sum()):,}")
