# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Phase 0a — Top 200 US metros by population
#
# Pulls CBSA-level population estimates from the Census PEP API (vintage 2023),
# filters to metropolitan statistical areas (drops micros), keeps the top 200,
# and saves to `data/processed/top200_metros.csv`.
#
# Requires `CENSUS_API_KEY` in env (or the manually-staged CSV fallback —
# see docstring in `arbok.sources.census_metros`).

# %%
from arbok.sources.census_metros import build_and_save, load_top200, top_n_metros, fetch_cbsa_population

# %% [markdown]
# ## Build (one-shot)

# %%
top = build_and_save(vintage=2023)
top.head(10)

# %% [markdown]
# ## Reload from disk

# %%
df = load_top200()
print(df.shape)
df.head()

# %% [markdown]
# ## Sanity checks

# %%
# Top 5 by population
df.nlargest(5, "population")[["rank", "core_name", "state", "population"]]

# %%
# State coverage (counts of metros per primary state grouping)
df["state"].value_counts().head(15)

# %%
# Smallest metro in the top-200 — defines the population threshold for inclusion
df.tail(3)[["rank", "core_name", "state", "population"]]

# %% [markdown]
# ## Optional: include micropolitan areas for comparison
#
# Just to confirm the metro-only filter is doing what we expect.

# %%
all_cbsas = fetch_cbsa_population(vintage=2023)
metros = all_cbsas["name"].str.endswith("Metro Area").sum()
micros = all_cbsas["name"].str.endswith("Micro Area").sum()
print(f"Metros: {metros}  Micros: {micros}  Total: {len(all_cbsas)}")
