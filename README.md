# arbok

Predictor study for US residential real estate at zip-code level. Top ~200 metros by population, three time horizons (short / medium / long), free data sources only.

## Quick start

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv + install deps
uv sync --all-extras

# Run a notebook
uv run jupyter lab notebooks/
```

## Layout

```
src/arbok/         # Importable, testable Python package
  config.py        # Paths, horizons, backtest splits
  sources/         # One module per data source
  panel.py         # Zip-month panel construction
  targets.py       # Forward-return computation
notebooks/         # Percent-format .py (open as notebooks in VS Code; jupytext converts to .ipynb)
data/
  raw/             # Downloaded files (gitignored)
  interim/         # In-progress processing (gitignored)
  processed/       # Final feature store + small reference tables
docs/
  DESIGN.md        # Study design, methodology, backtest splits
  PREDICTORS.md    # Catalog of ~50 predictors with sources & confidence
```

## Notebook format

Notebooks live as percent-format `.py` (`# %%` cell markers) for git-friendly diffs. VS Code and PyCharm open them as notebooks. Convert to `.ipynb` with `jupytext --to ipynb notebooks/00_panel_targets.py`.

## Scope

- **Geography:** top 200 US metros by population (CBSAs), zip-code level
- **Targets:** 6mo, 1yr, 3yr, 5yr, 10yr forward returns
- **Horizon classes:** short (<1y), medium (1–3y), long (5–10y) — compared head-to-head
- **Backtest splits:**
  - **pre-2008:** train ≤ 2007-12, test 2008–2013 (GFC)
  - **post-2012:** train ≤ 2017-12, test 2018–2024 (rate shock + COVID)
- **Data:** free / public sources only — see `docs/DESIGN.md`
