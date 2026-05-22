"""Human-readable labels + descriptions for every modeling feature.

Each entry maps the internal feature column name (e.g. ``fred__mortgage_30y_d12m``)
to a (display, description) tuple. Used in the HTML report to make the SHAP
charts and leaderboard intelligible to a non-technical reader.

When you add a feature in src/arbok/features/assembler.py or add a new source,
add the corresponding label here too. Unknown features fall back to a humanized
version of the column name + a generic description noting the source namespace.
"""
from __future__ import annotations

# (display, description) per feature
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    # ───────────────────────────────────── FRED macro
    "fred__mortgage_30y":          ("30Y mortgage rate",        "Average 30-year fixed-rate mortgage from Freddie Mac (FRED: MORTGAGE30US). Weekly, % per year."),
    "fred__mortgage_30y_d1m":      ("30Y rate, 1mo Δ",         "1-month change in the 30Y mortgage rate (percentage points)."),
    "fred__mortgage_30y_d3m":      ("30Y rate, 3mo Δ",         "3-month change in the 30Y mortgage rate."),
    "fred__mortgage_30y_d12m":     ("30Y rate, 12mo Δ",        "12-month change in the 30Y mortgage rate. Big positive values = rate-shock regime."),
    "fred__real_rate_10y":         ("10Y real rate (TIPS)",     "Yield on the 10-year Treasury Inflation-Protected Security (FRED: DFII10). Real cost of capital."),
    "fred__real_rate_10y_d1m":     ("10Y real rate, 1mo Δ",    "1-month change in the 10Y TIPS yield."),
    "fred__real_rate_10y_d3m":     ("10Y real rate, 3mo Δ",    "3-month change in the 10Y TIPS yield."),
    "fred__real_rate_10y_d12m":    ("10Y real rate, 12mo Δ",   "12-month change in the 10Y TIPS yield."),
    "fred__treasury_10y":          ("10Y Treasury yield",       "Nominal 10-year US Treasury constant-maturity yield (FRED: DGS10)."),
    "fred__treasury_10y_d1m":      ("10Y Treasury, 1mo Δ",     "1-month change in the 10Y Treasury yield."),
    "fred__treasury_10y_d3m":      ("10Y Treasury, 3mo Δ",     "3-month change in the 10Y Treasury yield."),
    "fred__treasury_10y_d12m":     ("10Y Treasury, 12mo Δ",    "12-month change in the 10Y Treasury yield."),
    "fred__m2_sa":                 ("M2 money supply",          "Seasonally-adjusted M2 money supply in billions of $ (FRED: M2SL). Liquidity proxy."),
    "fred__housing_affordability": ("Housing affordability",    "NAR Housing Affordability Index — combines home price, mortgage rate, median income (FRED: FIXHAI)."),
    "fred__case_shiller_us":       ("Case-Shiller US",          "S&P Case-Shiller US national home-price index (FRED: CSUSHPISA). Big-picture price trend."),
    "fred__lumber":                ("Lumber PPI",               "Producer Price Index for lumber & wood products (FRED: WPU081). Construction-cost proxy."),
    "fred__unemployment_us":       ("US unemployment rate",     "Headline US unemployment rate (FRED: UNRATE), %."),

    # ───────────────────────────────────── Realtor.com inventory
    "rdc__median_dom":               ("Median days on market",  "Median days a listing sits before going under contract, per zip (Realtor.com Research)."),
    "rdc__median_listing_price":     ("Median listing price",   "Median asking price of active listings in the zip."),
    "rdc__median_listing_price_per_sqft": ("Median list $/sqft", "Median listing price per square foot."),
    "rdc__active_listing_count":     ("Active listings",        "Count of active for-sale listings in the zip."),
    "rdc__new_listing_count":        ("New listings",           "Count of newly-listed homes in the month."),
    "rdc__price_reduced_count":      ("Listings with price cut", "Count of listings with at least one price reduction. Demand-weakness proxy."),
    "rdc__pending_listing_count":    ("Pending listings",       "Count of listings currently under contract but not yet closed."),

    # ───────────────────────────────────── Census BPS (building permits)
    "bps__permits_1unit":            ("Permits: single-family",  "Building permits authorized for 1-unit structures in the metro that month."),
    "bps__permits_2units":           ("Permits: 2-unit",         "Building permits for 2-unit (duplex) structures."),
    "bps__permits_3to4units":        ("Permits: 3-4 unit",       "Building permits for small multi-family (3-4 units)."),
    "bps__permits_5plus_units":      ("Permits: 5+ unit",        "Building permits for larger multi-family (5+ units)."),
    "bps__permits_total":            ("Permits: total",          "Total new-residential-construction permits authorized in the metro this month."),
    "bps__valuation_total_usd":      ("Permit valuation $",       "Total declared construction value of permitted structures (USD)."),

    # ───────────────────────────────────── BLS QCEW (wages)
    "qcew__avg_weekly_wage":         ("Avg weekly wage",         "Average weekly wage across all industries in the county, quarterly (BLS QCEW)."),
    "qcew__avg_weekly_wage_yoy_pct": ("Wage YoY %",             "Year-over-year % change in county avg weekly wage. Income-growth proxy."),

    # ───────────────────────────────────── IRS SOI county-to-county migration
    "irs__net_returns":               ("Net tax-return inflow",  "Net inflow of tax returns to the county (inflow − outflow). Population-flow signal (IRS SOI)."),
    "irs__net_agi_thousands":         ("Net AGI inflow $K",      "Net Adjusted Gross Income inflow to the county, $1k. Income-weighted migration."),

    # ───────────────────────────────────── ACS demographics
    "acs__total_pop":                 ("Total population",        "Total population of the ZCTA from ACS 5-year estimates."),
    "acs__median_household_income":   ("Median HH income",        "Median household income (ACS, $ per year)."),
    "acs__median_home_value":         ("Median home value",       "ACS-reported median owner-occupied home value (self-reported survey)."),
    "acs__median_gross_rent":         ("Median gross rent",       "Median monthly gross rent paid by renters in the ZCTA."),
    "acs__owner_occupied_units":      ("Owner-occupied units",    "Count of owner-occupied housing units."),
    "acs__occupied_units_total":      ("Total occupied units",    "Total occupied housing units (owner + renter)."),
    "acs__pop_25_29":                 ("Pop 25-29",               "Population aged 25-29 in the ZCTA."),
    "acs__pop_30_34":                 ("Pop 30-34",               "Population aged 30-34. Prime first-time-homebuyer cohort."),
    "acs__pop_35_39":                 ("Pop 35-39",               "Population aged 35-39."),
    "acs__pop_40_44":                 ("Pop 40-44",               "Population aged 40-44."),
    "acs__pop_65_plus":               ("Pop 65+",                 "Population aged 65 and over."),
    "acs__bachelors_or_higher_pct":   ("% Bachelor's+",          "Share of adults aged 25+ with a bachelor's degree or higher."),
    "acs__owner_occupied_pct":        ("% owner-occupied",        "Share of occupied housing units that are owner-occupied."),
    "acs__age_28_38_share":           ("Age 28-38 share",         "Share of population aged 28-38 — approximated from S0101 5-yr age bins (Census ACS)."),

    # ───────────────────────────────────── HMDA mortgage
    "hmda__count_originated":         ("HMDA originations",       "Count of originated home-purchase loans in the tract that year (CFPB HMDA)."),
    "hmda__mean_loan_amount":         ("HMDA mean loan $",        "Mean loan amount on originated mortgages in the tract."),
    "hmda__institutional_share":      ("HMDA institutional %",    "Share of originated loans linked to institutional / non-owner-occupied buyers."),

    # ───────────────────────────────────── FEMA disasters
    "fema__disasters_10yr_count":     ("FEMA disasters (10y)",    "Count of federally-declared disasters touching the county in the trailing 10 years."),
    "fema__disasters_10yr_fire":      ("FEMA fires (10y)",         "Wildfire-category FEMA disasters in the trailing 10 years."),
    "fema__disasters_10yr_flood":     ("FEMA floods (10y)",        "Flood-category FEMA disasters in the trailing 10 years."),
    "fema__disasters_10yr_hurricane": ("FEMA hurricanes (10y)",    "Hurricane-category FEMA disasters in the trailing 10 years."),
    "fema__disasters_10yr_other":     ("FEMA other disasters",     "Other-category FEMA disasters in the trailing 10 years."),
    "fema__disasters_10yr_severe_storm": ("FEMA severe storms",    "Severe-storm FEMA disasters in the trailing 10 years."),
    "fema__disasters_10yr_tornado":   ("FEMA tornadoes (10y)",     "Tornado FEMA disasters in the trailing 10 years."),

    # ───────────────────────────────────── FEMA NRI (climate risk)
    "nri__flood_risk_combined":       ("NRI flood risk",           "FEMA National Risk Index — max of riverine + coastal flood risk score (0-100). Tract-level."),
    "nri__wildfire_risk":             ("NRI wildfire risk",        "FEMA NRI wildfire risk score (0-100). Tract-level."),
    "nri__composite_risk":            ("NRI composite risk",       "FEMA NRI overall composite natural-hazard risk score (0-100)."),

    # ───────────────────────────────────── Derived (panel transforms)
    "derived__yield_gross":           ("Gross rental yield",       "12 × ZORI / ZHVI: annualized gross rental yield per zip (cap-rate proxy)."),
    "derived__vol_24m":               ("ZHVI 24m volatility",      "Rolling 24-month standard deviation of monthly ZHVI returns. Higher = more price-risky."),
    "derived__drawdown":              ("ZHVI drawdown",            "Current ZHVI as fraction of trailing-60-month max (negative = below recent peak)."),

    # ───────────────────────────────────── DOE AFDC EV chargers
    "afdc__station_count":            ("EV stations",              "Count of public EV charging stations in the zip (DOE AFDC, snapshot)."),
    "afdc__l2_ports":                 ("Level-2 EV ports",         "Total Level-2 EV charging ports in the zip."),
    "afdc__dcfc_ports":               ("DC fast-charge ports",     "Total DC fast-charge ports in the zip. Sparse but growing fast."),

    # ───────────────────────────────────── Census ZBP (business establishments)
    "zbp__estab":                     ("Business establishments",  "Total business establishments in the zip (Census Zip Business Patterns, 2018 vintage)."),
    "zbp__emp":                       ("Business employment",      "Total employees of businesses in the zip."),

    # ───────────────────────────────────── Property tax (derived from ACS)
    "ptax__effective_property_tax_rate": ("Effective property tax rate",  "Median real-estate taxes paid / median home value per ZCTA (ACS B25103/B25077). NJ ≈ 2.2%, HI ≈ 0.25%."),
    "ptax__median_real_estate_tax_paid": ("Median property tax paid",     "Median real-estate taxes paid by ZCTA owners with a mortgage (ACS B25103, $/yr)."),
    "ptax__median_home_value":           ("ACS median home value",         "ACS-reported median owner-occupied home value (self-reported survey, B25077)."),

    # ───────────────────────────────────── BEA per-capita income
    "bea__per_capita_income_usd":        ("BEA per-capita income $",      "BEA per-capita personal income per county (annual). More current than ACS."),
    "bea__total_personal_income_thousands": ("BEA total personal income $K", "Total county personal income from BEA, in $1k (annual)."),
    "bea__population":                   ("BEA population",                "BEA estimated population per county (annual; differs from Census PEP marginally)."),

    # ───────────────────────────────────── CMS Hospital Compare
    "cms__hospital_count":                ("Hospitals in county",          "Count of Medicare-certified hospitals in the county (CMS Provider Data)."),
    "cms__mean_star_rating":              ("Mean hospital star rating",    "Mean overall 1-5 CMS star rating across rated hospitals in the county."),
    "cms__total_beds":                    ("Hospital bed count",            "Total Medicare-certified hospital beds in the county (may be NaN — POS extract needed)."),
    "cms__pct_with_4plus_stars":          ("% hospitals 4+ stars",         "Share of rated hospitals in the county scoring 4 or 5 CMS overall stars."),

    # ───────────────────────────────────── MIT Election Lab
    "mit__dem_vote_share":                ("Dem vote share (county)",      "County Democratic vote share at the prior presidential election (MIT Election Lab)."),
    "mit__rep_vote_share":                ("Rep vote share (county)",      "County Republican vote share at the prior presidential election (MIT Election Lab)."),
    "mit__margin":                        ("Vote margin (Dem-Rep)",        "Signed vote margin: + = Dem-leaning county, - = Rep-leaning."),
    "mit__winner_party":                  ("Winning party (county)",        "Party that won the most recent presidential vote in this county."),

    # ───────────────────────────────────── County Health Rankings
    "health__health_outcomes_rank":       ("Health outcomes rank",          "National percentile rank derived from premature death rate (lower = healthier)."),
    "health__health_factors_rank":        ("Health factors rank",           "National percentile composite of smoking, obesity, uninsured, poverty (lower = better)."),
    "health__pct_smokers":                ("% adult smokers",               "Share of adults who currently smoke (County Health Rankings v009)."),
    "health__pct_obese":                  ("% adult obese",                 "Share of adults with BMI ≥ 30 (CHR)."),
    "health__pct_uninsured":              ("% uninsured",                   "Share of population under 65 without health insurance (CHR)."),
    "health__pct_in_poverty":             ("% in poverty",                  "Share of population below federal poverty line (CHR)."),
    "health__avg_daily_pm25":             ("Avg daily PM2.5",                "County average daily PM2.5 concentration (µg/m³) per CHR — duplicates EPA AQS at lower resolution."),

    # ───────────────────────────────────── HUD Fair Market Rent
    "fmr__fmr_efficiency":                ("FMR studio",                    "HUD Fair Market Rent for an efficiency unit (Small Area FMR for ZIPs in 24 designated metros)."),
    "fmr__fmr_1br":                       ("FMR 1-bedroom",                 "HUD Fair Market Rent for a 1-bedroom unit ($/mo)."),
    "fmr__fmr_2br":                       ("FMR 2-bedroom",                 "HUD Fair Market Rent for a 2-bedroom unit ($/mo). The Section-8 reference."),
    "fmr__fmr_3br":                       ("FMR 3-bedroom",                 "HUD Fair Market Rent for a 3-bedroom unit ($/mo)."),
    "fmr__fmr_4br":                       ("FMR 4-bedroom",                 "HUD Fair Market Rent for a 4-bedroom unit ($/mo)."),

    # ───────────────────────────────────── EIA energy
    "elec__elec_residential_cents_per_kwh": ("Residential electricity ¢/kWh", "Monthly average residential electricity price per state from EIA (¢/kWh)."),

    # ───────────────────────────────────── Behavioral
    "wiki__wiki_pageviews":               ("Wikipedia pageviews",           "Monthly pageviews of the metro's primary Wikipedia article — search-interest proxy."),

    # ───────────────────────────────────── Pseudo-feature: zhvi at time t
    "zhvi_t":                         ("ZHVI at time t",            "Current Zillow Home Value Index for the zip — the price level we're predicting the return on."),
}


def label(feature_name: str) -> tuple[str, str]:
    """Look up (display, description) for a feature; humanize gracefully on miss."""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    # Fall back to a humanized version
    if "__" in feature_name:
        ns, rest = feature_name.split("__", 1)
        display = rest.replace("_", " ")
        return (f"{display} ({ns})",
                f"Feature {feature_name} from source namespace `{ns}`. No detailed label registered.")
    return (feature_name, f"Feature {feature_name}. No detailed label registered.")
