# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
# ---

# %% [markdown]
# # Phase 0b/0c — Zip-month panel + forward returns
#
# 1. Fetch Zillow ZHVI (SFR, smoothed SA) + ZORI (all homes + multifamily, smoothed SA) at zip.
# 2. Load HUD-USPS ZIP-CBSA crosswalk (manual download required — see docstring).
# 3. Load the top-200 metros table produced by `notebooks/10_metros.py`.
# 4. Build the zip-month panel filtered to those metros.
# 5. Compute 6mo / 1y / 3y / 5y / 10y forward ZHVI returns and split labels.
# 6. Sanity check: per-metro means over time.

# %%
from __future__ import annotations

import logging

import pandas as pd

from arbok.config import HORIZONS_MONTHS, SPLITS
from arbok.panel import build_panel
from arbok.sources.census_metros import load_top200
from arbok.sources.hud_crosswalk import (
    load_zip_cbsa_crosswalk,
    load_zip_cbsa_crosswalk_static,
)
from arbok.sources.zillow import fetch_zhvi, fetch_zori
from arbok.targets import add_split_label, compute_forward_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# %% [markdown]
# ## 1. Zillow ZHVI + ZORI

# %%
zhvi = fetch_zhvi()
zori = fetch_zori()
print(f"ZHVI: {len(zhvi):,} rows, {zhvi['zip'].nunique():,} zips, "
      f"{zhvi['year_month'].min()} -> {zhvi['year_month'].max()}")
print(f"ZORI: {len(zori):,} rows, {zori['zip'].nunique():,} zips, "
      f"{zori['year_month'].min()} -> {zori['year_month'].max()}")
zhvi.head()

# %% [markdown]
# ## 2. HUD ZIP-CBSA crosswalk
#
# Manual download required. See `arbok.sources.hud_crosswalk` docstring.
# If the workbook isn't present we fall back to the live API (requires
# `HUD_USPS_TOKEN` in env). Edit `(year, quarter)` to whatever you downloaded.

# %%
try:
    crosswalk = load_zip_cbsa_crosswalk(year=2024, quarter=4)
    print(f"Loaded HUD crosswalk from local workbook: {len(crosswalk):,} rows")
except FileNotFoundError as e:
    print(e)
    crosswalk = load_zip_cbsa_crosswalk_static()
    print(f"Loaded HUD crosswalk from API: {len(crosswalk):,} rows")
crosswalk.head()

# %% [markdown]
# ## 3. Top-200 metros (produced by `notebooks/10_metros.py`)

# %%
top200 = load_top200()
print(f"top200: {len(top200):,} metros")
top200.head()

# %% [markdown]
# ## 4. Build the panel

# %%
panel = build_panel(zhvi=zhvi, zori=zori, crosswalk=crosswalk, top200=top200)
print(
    f"Panel: {len(panel):,} rows, {panel['zip'].nunique():,} zips, "
    f"{panel['cbsa'].nunique():,} CBSAs, "
    f"{panel['year_month'].min()} -> {panel['year_month'].max()}"
)
panel.head()

# %% [markdown]
# ## 5. Forward returns + split labels

# %%
targets = compute_forward_returns(panel)
targets = add_split_label(targets)
fwd_cols = list(HORIZONS_MONTHS.keys())
print("Non-null target counts:")
print(targets[fwd_cols].notna().sum())
print("\nSplit summary (coarse):")
print(targets["split"].value_counts(dropna=False))
print("\nPer-scheme membership:")
for key in SPLITS:
    print(
        f"  {key}: train={targets[f'is_train_{key}'].sum():,} "
        f"test={targets[f'is_test_{key}'].sum():,}"
    )
targets.head()

# %% [markdown]
# ## 6. Sanity check — average forward return by metro for a few well-known CBSAs
#
# 35620 = New York-Newark-Jersey City, 31080 = Los Angeles-Long Beach-Anaheim,
# 16980 = Chicago-Naperville-Elgin, 41860 = San Francisco-Oakland-Berkeley,
# 19100 = Dallas-Fort Worth-Arlington.

# %%
sample_cbsas = ["35620", "31080", "16980", "41860", "19100"]
sample = targets[targets["cbsa"].isin(sample_cbsas)]
metro_means = (
    sample.groupby("cbsa")[fwd_cols]
    .mean()
    .reindex(sample_cbsas)
)
metro_means

# %% [markdown]
# ### Average ZHVI by year for the same metros

# %%
sample = sample.assign(year=sample["year_month"].dt.year)
zhvi_by_year = (
    sample.groupby(["cbsa", "year"])["zhvi"]
    .mean()
    .unstack("cbsa")
    .reindex(columns=sample_cbsas)
)
zhvi_by_year.tail(15)

# %% [markdown]
# ### Quick plot — median ZHVI over time across all top-200 metros

# %%
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("matplotlib not installed; skipping plot.")

if plt is not None:
    ts = targets.groupby("year_month")["zhvi"].median()
    fig, ax = plt.subplots(figsize=(10, 4))
    ts.index = ts.index.to_timestamp()
    ax.plot(ts.index, ts.values)
    ax.set_title("Median ZHVI across top-200-metro zips")
    ax.set_ylabel("USD")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
