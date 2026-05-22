# Study design

## Goal
Identify the best predictors of US residential real-estate returns at zip-code level, separated by time horizon. Two outputs:
1. A predictive model per horizon, with SHAP-based feature importance.
2. A personal investment shortlist of zip codes (constrained by buyer-specific filters).

Model first, personal overlay second.

## Scope (locked 2026-05-21)
- **Geography:** top 200 US metros by population (CBSAs), zip-code level (~22K zips).
- **Horizons:** short (6mo, 1y), medium (3y), long (5y, 10y), studied comparatively.
- **Data:** free / public sources only.
- **Deliverable:** this repo + notebooks.
- **Backtest splits:** pre-2008 AND post-2012, both reported.

## Targets

Forward total appreciation, annualized for horizons ≥ 1y:

| Column   | Class  | Definition                            |
|----------|--------|---------------------------------------|
| fwd_6m   | short  | ZHVI(t+6) / ZHVI(t) − 1               |
| fwd_1y   | short  | ZHVI(t+12) / ZHVI(t) − 1              |
| fwd_3y   | medium | (ZHVI(t+36) / ZHVI(t))^(1/3) − 1      |
| fwd_5y   | long   | (ZHVI(t+60) / ZHVI(t))^(1/5) − 1      |
| fwd_10y  | long   | (ZHVI(t+120) / ZHVI(t))^(1/10) − 1    |

Phase 0 uses price appreciation only. Phase 2 augments with imputed rent (ZORI) and a carry estimate for total return.

## Backtest splits

Both reported for every model.

| Split     | Train     | Test         | Captures                          |
|-----------|-----------|--------------|-----------------------------------|
| pre_2008  | ≤ 2007-12 | 2008–2013    | GFC bust + recovery               |
| post_2012 | ≤ 2017-12 | 2018–2024    | Rate shock, COVID, 2022 reset     |

## Validation
- **Spatial** holdout: hold out entire metros (CBSAs) so adjacent zips never leak from train to test.
- **Temporal** holdout within each split (rolling-origin where useful).
- Baselines to beat: naive AR(1) on target, metro-mean forecast, hedonic regression on ACS basics.

## Modeling
- Per-horizon ElasticNet (interpretable) + LightGBM (performance), both reported.
- Interpretation via SHAP. Coefficients only for sanity-checking.
- No deep learning in Phase 2; revisit if structured-tabular ceiling proven.

## Build sequence

| Phase | Deliverable                                                   |
|-------|---------------------------------------------------------------|
| 0     | Top-200 metros, Zillow ZHVI/ZORI panel, forward returns       |
| 1     | Ingest 15 starter-pack predictors                             |
| 2     | Baselines + three horizon models on starter pack; spatial CV  |
| 3     | Expand to full ~50-predictor catalog; SHAP comparison         |
| 4     | Personal overlay (filter top zips by buyer constraints)       |

## Methodological traps (don't relitigate)
- **Spatial autocorrelation:** spatial CV, never random k-fold.
- **ACS smoothing:** 5-year ACS estimates are centered on year 2 of the window; never treat as monthly.
- **Vintaging:** use ALFRED for FRED revisions; record data-as-of in feature names where it matters.
- **Whole Foods endogeneity:** alpha is in lease *announcements* 18mo out, not openings. Track announcements where possible.
- **Zip-code stability:** USPS occasionally redraws zips. Use HUD zip↔tract crosswalk; track changes year-over-year.
- **Thin-sales zips:** apply minimum monthly sales-volume filter (≥10 sales/mo trailing 12mo); ZHVI drops sparse zips itself but the filter is belt-and-suspenders.

## Output artifacts

Two final artifacts:
1. **Research output:** feature-importance comparison across horizons + a "novel predictor" callout for any speculative entry that meaningfully beats the well-established baselines.
2. **Decision output:** zip leaderboard with confidence bands and top-3 SHAP drivers per zip, filterable by buyer constraints.
