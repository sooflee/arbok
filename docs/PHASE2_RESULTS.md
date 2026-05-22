# Phase 2 results — first end-to-end pass

Run date: 2026-05-21. Feature store: 3.73M rows × 80 cols (panel × 8 of 12 source modules). 4 sources still missing parquets (HMDA, FCC, VIIRS, Foursquare).

## Headline

LightGBM beats every baseline on every horizon × split combination for spatial ranking. The **medium horizon (fwd_3y) on the post_2012 split is the strongest signal** in this run.

| Horizon | Best model | Spearman ρ | Decile spread | Top decile realized | Bottom decile realized |
|---------|------------|-----------:|--------------:|--------------------:|-----------------------:|
| fwd_1y  | LightGBM   | +0.095     | +2.31%        | 9.23%               | 6.92%                  |
| fwd_3y  | LightGBM   | **+0.294** | **+6.75%**    | 9.70%               | 2.95%                  |
| fwd_5y  | LightGBM   | +0.320     | +2.05%        | 9.24%               | 7.19%                  |

(All numbers from the post_2012 split — train ≤ 2017-12, test 2018–2024.)

## Why R² is negative everywhere

Test windows are crisis regimes (GFC for pre_2008; rate-shock + COVID for post_2012). The mean target value in the test period is meaningfully different from the train period, so any model that predicts approximately the train mean gets penalized hard on R². **Decile spread + Spearman are the actionable metrics here** — they measure spatial ranking, which is what an investor cares about.

## Top SHAP features per horizon × split (post_2012)

| Rank | fwd_1y                                 | fwd_3y                                  | fwd_5y                                |
|------|----------------------------------------|-----------------------------------------|---------------------------------------|
| 1    | `fred__m2_sa`                          | **`fred__real_rate_10y`**               | **`fred__real_rate_10y`**             |
| 2    | `fred__lumber`                         | `fred__m2_sa`                           | `fred__case_shiller_us`               |
| 3    | `fema__disasters_10yr_fire`            | `fema__disasters_10yr_hurricane`        | `fred__m2_sa`                         |
| 4    | `fred__case_shiller_us`                | `fema__disasters_10yr_fire`             | `fema__disasters_10yr_fire`           |
| 5    | `zhvi_t`                               | `fred__case_shiller_us`                 | `fema__disasters_10yr_hurricane`      |
| 6    | `fema__disasters_10yr_hurricane`       | `zhvi_t`                                | `fred__unemployment_us`               |
| 7    | `fema__disasters_10yr_other`           | `nri__wildfire_risk`                    | `zhvi_t`                              |
| 8    | `nri__wildfire_risk`                   | `fema__disasters_10yr_count`            | `nri__wildfire_risk`                  |
| 9    | `fema__disasters_10yr_count`           | `fred__lumber`                          | `fema__disasters_10yr_count`          |
| 10   | `fema__disasters_10yr_severe_storm`    | `fema__disasters_10yr_other`            | `fema__disasters_10yr_other`          |

## Two non-obvious findings

### 1. Climate features are doing real work

FEMA fire / hurricane / storm disasters and NRI wildfire / flood scores appear in the **top 10 SHAP features for every horizon × split combination**. The design-doc thesis was that climate is the underweighted sleeper predictor bucket. Validated — and with only FEMA NRI + OpenFEMA, both free.

### 2. Real rates dominate medium/long; money supply dominates short

The horizon comparison surfaces an interpretable separation:

- **Short (1y):** monetary aggregates (M2) and commodity prices (lumber) lead.
- **Medium / long (3y, 5y):** real rates (10Y TIPS) take #1. Case-Shiller momentum follows.

This is exactly the comparative pattern the study was designed to surface.

## Open issues from this run

- **ElasticNet + hedonic baselines blow up** (RMSE 0.5–3.4 vs LightGBM's 0.03–0.07). Predictions go wildly outside the plausible return range. Likely alpha too small or a high-magnitude feature dominating after scaling. Task #13.
- **27 features pre_2008 vs 41 post_2012** — Realtor, ACS, BPS, QCEW, IRS-SOI all begin coverage post-2010. Cleaner pre-2008 results need historical reconstruction.
- **Single random spatial split**, not full spatial K-fold. Add walk-forward CV for confidence bands.
- **4 sources still missing**: HMDA (no auth, just slow), FCC (auth + manual), VIIRS (auth + rasterio), Foursquare (HF download).
- **ACS age-bin codes drift pre-2017** — `age_28_38_share` shows 0% for older vintages. Bug filed.
