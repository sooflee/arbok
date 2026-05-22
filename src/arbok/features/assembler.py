"""Top-level feature-store builder.

Joins every source onto the (zip, year_month) panel spine. Each source is
declared once in `SOURCE_SPECS` with its level (zip / zcta / tract / county /
cbsa / national), its time grain, its vintage lag, and a small per-source
adapter function that returns a wide zip-level (or panel-level) DataFrame.

Vintage lag rules: the value of a feature OBSERVED at time T is only available
to predict targets at time T + lag_months (the publication delay). The assembler
shifts each feature forward by its lag so that no row contains future
information. If you ever see negative R^2 disappear, suspect a missing lag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from arbok.config import PROCESSED
from arbok.features.crosswalks import (
    aggregate_tract_to_zip,
    broadcast_county_to_zip,
    load_zip_county,
    load_zip_tract,
)


# Vintage publication lags in months — how long after measurement until the value
# becomes publicly known. Conservative; tune per series as needed.
VINTAGE_LAGS: dict[str, int] = {
    "zillow": 1,        # ZHVI releases ~3 weeks after month-end
    "fred": 1,          # most FRED rate series same-day or next-day; round up
    "realtor": 1,       # Realtor.com monthly inventory ~2 weeks after month-end
    "census_bps": 2,    # building permits ~6 weeks after reference month
    "bls_qcew": 7,      # QCEW released ~6.5 months after quarter end
    "hud_usps": 4,      # USPS vacancy quarterly with ~3-4 month lag
    "irs_soi": 18,      # IRS SOI ~18-24 month lag
    "acs": 12,          # ACS 5-yr released Dec of year+1
    "hmda": 9,          # HMDA modified LAR ~9 months; aggregations ~3 months
    "fcc_broadband": 6, # BDC ~6 months after reference period
    "fema_disasters": 1,
    "fema_nri": 12,     # NRI is annual; treat as 1-year lag for safety
    "noaa_viirs": 6,    # VIIRS annual composite ~3-6 months
    "foursquare": 3,    # Snapshot; treat as 3-month lag for new POIs to surface
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    level: str       # zip | zcta | tract | county | cbsa | national
    grain: str       # monthly | quarterly | annual | static
    lag_months: int
    adapter: Callable[[], pd.DataFrame]
    namespace: str   # prefix prepended to feature columns


def _ns(df: pd.DataFrame, prefix: str, keys: tuple[str, ...] = ("zip", "year_month")) -> pd.DataFrame:
    """Rename feature columns to `<prefix>__<col>`, leaving join keys untouched."""
    return df.rename(columns={c: f"{prefix}__{c}" for c in df.columns if c not in keys})


def _shift_year_month(df: pd.DataFrame, lag_months: int, col: str = "year_month") -> pd.DataFrame:
    """Push observation times forward by lag_months so they're only joined to
    targets at time T >= obs_time + lag."""
    if lag_months == 0:
        return df
    out = df.copy()
    out[col] = out[col] + pd.DateOffset(months=lag_months)
    return out


def _broadcast_annual_to_monthly(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """Replicate annual rows across 12 months. Lag shift later moves the
    publication date forward so no row leaks future info."""
    out = df.copy()
    out[year_col] = out[year_col].astype(int)
    pieces = []
    for month in range(1, 13):
        chunk = out.copy()
        chunk["year_month"] = pd.to_datetime(
            chunk[year_col].astype(str) + "-" + str(month).zfill(2) + "-01"
        )
        pieces.append(chunk)
    rep = pd.concat(pieces, ignore_index=True)
    return rep.drop(columns=[year_col])


def _broadcast_quarterly_to_monthly(df: pd.DataFrame, year_quarter_col: str = "year_quarter") -> pd.DataFrame:
    """Replicate quarterly rows across 3 months of that quarter."""
    out = df.copy()
    yq = out[year_quarter_col].astype(str).str.split("Q", expand=True)
    year = yq[0].astype(int)
    quarter = yq[1].astype(int)
    pieces = []
    for offset in range(3):
        chunk = out.copy()
        month = (quarter - 1) * 3 + 1 + offset  # Q1 -> 1,2,3; Q2 -> 4,5,6; etc.
        chunk["year_month"] = pd.to_datetime(
            year.astype(str) + "-" + month.astype(str).str.zfill(2) + "-01"
        )
        pieces.append(chunk)
    rep = pd.concat(pieces, ignore_index=True)
    return rep.drop(columns=[year_quarter_col])


# ---------------------------------------------------------------------------
# Per-source adapters. Each returns a DataFrame with `zip` (+ `year_month` if
# temporal) and feature columns. None do crosswalking or lag-shifting themselves
# — those are applied uniformly in `_apply_spec`.
# ---------------------------------------------------------------------------

def _adapter_zillow_panel() -> pd.DataFrame:
    """ZHVI + ZORI are the spine; load them via panel.build_panel separately.
    This adapter is a no-op; the assembler treats Zillow as the spine."""
    from arbok.panel import build_panel  # noqa
    raise NotImplementedError("zillow is the spine, not joined as a side source")


def _adapter_fred() -> pd.DataFrame:
    from arbok.sources.fred import load_macro_monthly
    df = load_macro_monthly()
    df = df.reset_index().rename(columns={df.index.name or "year_month": "year_month"})
    return df  # national-level (no zip)


def _adapter_realtor() -> pd.DataFrame:
    from arbok.sources.realtor import load_realtor_inventory
    return load_realtor_inventory()


def _adapter_census_bps_msa() -> pd.DataFrame:
    from arbok.sources.census_bps import load_bps
    df = load_bps(geo="msa").rename(columns={"cbsa": "cbsa"})
    return df  # cbsa + year_month


def _adapter_bls_qcew() -> pd.DataFrame:
    from arbok.sources.bls_qcew import load_qcew, aggregate_to_yoy_wage_growth
    yoy = aggregate_to_yoy_wage_growth(load_qcew())
    yoy = yoy.rename(columns={"area_fips": "county"})
    return yoy[["county", "year_quarter", "avg_weekly_wage", "avg_weekly_wage_yoy_pct"]]


def _adapter_irs_soi() -> pd.DataFrame:
    from arbok.sources.irs_soi import build_and_save  # noqa
    # Expects the prebuilt parquet
    df = pd.read_parquet(PROCESSED / "irs_migration_county_year.parquet")
    df["county"] = df["state_fips"].astype(str).str.zfill(2) + df["county_fips"].astype(str).str.zfill(3)
    return df  # county + year


def _adapter_acs() -> pd.DataFrame:
    from arbok.sources.acs import load_acs_zcta, derive_age_cohort_share
    df = load_acs_zcta()
    df = derive_age_cohort_share(df)
    return df.rename(columns={"zcta": "zip"})  # zcta -> approximate zip + year


def _adapter_hmda() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "hmda_tract_year.parquet")
    return df  # tract + year


def _adapter_fcc_broadband() -> pd.DataFrame:
    from arbok.sources.fcc_broadband import load_fcc_broadband
    return load_fcc_broadband()  # tract + version/effective_date


def _adapter_fema_disasters() -> pd.DataFrame:
    from arbok.sources.fema_disasters import load_fema_disasters
    df = load_fema_disasters()
    df["county"] = df["fips_state_code"].astype(str).str.zfill(2) + df["fips_county_code"].astype(str).str.zfill(3)
    return df  # county + year


def _adapter_fema_nri() -> pd.DataFrame:
    from arbok.sources.first_street import load_fema_nri, derive_starter_risk_features
    tract = load_fema_nri(level="tract")
    df = derive_starter_risk_features(tract)
    return df.rename(columns={"tract_geoid": "tract"})  # tract, static


def _adapter_noaa_viirs() -> pd.DataFrame:
    from arbok.sources.noaa_viirs import load_viirs_nightlights, derive_yoy_delta
    df = load_viirs_nightlights()
    df = derive_yoy_delta(df)
    return df.rename(columns={"zcta": "zip"})  # zcta -> approx zip + year


def _adapter_foursquare() -> pd.DataFrame:
    from arbok.sources.foursquare_places import load_foursquare_pois
    return load_foursquare_pois()  # zip, static


def _adapter_derived() -> pd.DataFrame:
    """Panel-derived features (ZORI yield, rolling volatility, drawdown)."""
    from arbok.features.derived import load_derived
    return load_derived()  # zip + year_month


def _adapter_afdc_ev() -> pd.DataFrame:
    """DOE AFDC EV-station counts per zip (snapshot)."""
    from arbok.sources.doe_afdc import load_afdc
    return load_afdc()  # zip, static


def _adapter_zbp() -> pd.DataFrame:
    """Census Zip Business Patterns — establishments + employment per zip."""
    from arbok.sources.census_zbp import load_zbp
    return load_zbp()  # zip + year (snapshot, last avail 2018)


def _adapter_epa_aqs() -> pd.DataFrame:
    """EPA AQS air quality — PM2.5 + ozone annual at county."""
    from arbok.sources.epa_aqs import load_epa_aqs
    df = load_epa_aqs()
    df["county"] = df["state_fips"].astype(str).str.zfill(2) + df["county_fips"].astype(str).str.zfill(3)
    return df  # county + year


def _adapter_wikipedia() -> pd.DataFrame:
    """Wikipedia monthly pageviews per metro Wikipedia article — search-interest proxy."""
    from arbok.sources.wikipedia_pageviews import load_pageviews
    df = load_pageviews()
    df = df.rename(columns={"views": "wiki_pageviews"})
    return df  # cbsa + year_month


def _adapter_bls_laus() -> pd.DataFrame:
    """BLS Local Area Unemployment Statistics — county-monthly unemployment rate."""
    from arbok.sources.bls_laus import load_laus
    df = load_laus()
    if "area_fips" in df.columns:
        df = df.rename(columns={"area_fips": "county"})
    return df  # county + year_month


def _adapter_bea_income() -> pd.DataFrame:
    """BEA county per-capita personal income (annual)."""
    from arbok.sources.bea_income import load_bea_income
    df = load_bea_income()
    if "area_fips" in df.columns:
        df = df.rename(columns={"area_fips": "county"})
    return df  # county + year


def _adapter_usaspending() -> pd.DataFrame:
    """USAspending — federal contract + grant $ flowing into the zip annually."""
    from arbok.sources.usaspending import load_usaspending
    df = load_usaspending()
    if "fiscal_year" in df.columns:
        df = df.rename(columns={"fiscal_year": "year"})
    return df  # zip + year


SOURCE_SPECS: list[SourceSpec] = [
    SourceSpec("fred",            "national", "monthly",   VINTAGE_LAGS["fred"],          _adapter_fred,           "fred"),
    SourceSpec("realtor",         "zip",      "monthly",   VINTAGE_LAGS["realtor"],       _adapter_realtor,        "rdc"),
    SourceSpec("census_bps",      "cbsa",     "monthly",   VINTAGE_LAGS["census_bps"],    _adapter_census_bps_msa, "bps"),
    SourceSpec("bls_qcew",        "county",   "quarterly", VINTAGE_LAGS["bls_qcew"],      _adapter_bls_qcew,       "qcew"),
    SourceSpec("irs_soi",         "county",   "annual",    VINTAGE_LAGS["irs_soi"],       _adapter_irs_soi,        "irs"),
    SourceSpec("acs",             "zcta",     "annual",    VINTAGE_LAGS["acs"],           _adapter_acs,            "acs"),
    SourceSpec("hmda",            "tract",    "annual",    VINTAGE_LAGS["hmda"],          _adapter_hmda,           "hmda"),
    SourceSpec("fcc_broadband",   "tract",    "annual",    VINTAGE_LAGS["fcc_broadband"], _adapter_fcc_broadband,  "fcc"),
    SourceSpec("fema_disasters",  "county",   "annual",    VINTAGE_LAGS["fema_disasters"], _adapter_fema_disasters,"fema"),
    SourceSpec("fema_nri",        "tract",    "static",    VINTAGE_LAGS["fema_nri"],      _adapter_fema_nri,       "nri"),
    SourceSpec("noaa_viirs",      "zcta",     "annual",    VINTAGE_LAGS["noaa_viirs"],    _adapter_noaa_viirs,     "viirs"),
    SourceSpec("foursquare",      "zip",      "static",    VINTAGE_LAGS["foursquare"],    _adapter_foursquare,     "fsq"),
    SourceSpec("derived",         "zip",      "monthly",   1,                              _adapter_derived,        "derived"),
    SourceSpec("afdc_ev",         "zip",      "static",    3,                              _adapter_afdc_ev,        "afdc"),
    SourceSpec("zbp",             "zip",      "annual",    12,                             _adapter_zbp,            "zbp"),
    SourceSpec("epa_aqs",         "county",   "annual",    12,                             _adapter_epa_aqs,        "aqs"),
    SourceSpec("wikipedia",       "cbsa",     "monthly",   1,                              _adapter_wikipedia,      "wiki"),
    SourceSpec("bls_laus",        "county",   "monthly",   2,                              _adapter_bls_laus,       "laus"),
    SourceSpec("bea_income",      "county",   "annual",    12,                             _adapter_bea_income,     "bea"),
    SourceSpec("usaspending",     "zip",      "annual",    6,                              _adapter_usaspending,    "usaspending"),
]


_NON_FEATURE_COLS = {
    "zip", "zcta", "tract", "county",
    "state_fips", "county_fips", "fips_state_code", "fips_county_code",
    "cbsa", "cbsa_name", "year", "year_month", "year_quarter",
    "version", "effective_date",
}


def _apply_spec(spec: SourceSpec, panel: pd.DataFrame, zip_tract: pd.DataFrame, zip_county: pd.DataFrame) -> pd.DataFrame:
    """Produce a (zip, year_month, feature*) frame for one source, ready to left-join onto panel.

    Pipeline: geography -> time -> lag -> namespace.
    """
    df = spec.adapter()
    value_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    # 1. Geography alignment: collapse to zip-keyed (or CBSA-keyed, or unkeyed for national).
    if spec.level in ("zip", "zcta"):
        if "zip" not in df.columns and "zcta" in df.columns:
            df = df.rename(columns={"zcta": "zip"})
        df["zip"] = df["zip"].astype(str).str.zfill(5)
    elif spec.level == "tract":
        time_col = next((c for c in ("year", "year_month") if c in df.columns), None)
        if time_col:
            pieces = []
            for k, g in df.groupby(time_col):
                z = aggregate_tract_to_zip(g, zip_tract, value_cols)
                z[time_col] = k
                pieces.append(z)
            df = pd.concat(pieces, ignore_index=True)
        else:
            df = aggregate_tract_to_zip(df, zip_tract, value_cols)
    elif spec.level == "county":
        time_col = next((c for c in ("year", "year_quarter", "year_month") if c in df.columns), None)
        if time_col:
            pieces = []
            for k, g in df.groupby(time_col):
                z = broadcast_county_to_zip(g, zip_county, value_cols)
                z[time_col] = k
                pieces.append(z)
            df = pd.concat(pieces, ignore_index=True)
        else:
            df = broadcast_county_to_zip(df, zip_county, value_cols)
    elif spec.level == "cbsa":
        df["cbsa"] = df["cbsa"].astype(str).str.zfill(5)
        # Don't merge to zip yet; defer until after time alignment.
    elif spec.level == "national":
        pass
    else:
        raise ValueError(f"unknown level: {spec.level}")

    # 2. Time alignment: normalize to year_month.
    if spec.grain == "monthly" and "year_month" in df.columns:
        df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M").dt.to_timestamp()
    elif spec.grain == "quarterly":
        df = _broadcast_quarterly_to_monthly(df)
    elif spec.grain == "annual":
        df = _broadcast_annual_to_monthly(df)
    elif spec.grain == "static":
        panel_months = panel[["year_month"]].drop_duplicates()
        df = df.merge(panel_months, how="cross")

    # 3. CBSA -> zip merge happens AFTER time alignment, so both sides share year_month.
    if spec.level == "cbsa":
        panel_zc = panel[["zip", "cbsa", "year_month"]].drop_duplicates()
        merge_keys = ["cbsa", "year_month"] if "year_month" in df.columns else ["cbsa"]
        df = panel_zc.merge(df, on=merge_keys, how="left").drop(columns=["cbsa"])

    # 4. Vintage lag shift.
    df = _shift_year_month(df, spec.lag_months)

    # 5. Namespace columns.
    keys = ("zip", "year_month") if spec.level != "national" else ("year_month",)
    df = _ns(df, spec.namespace, keys=keys)
    return df


def build_feature_store(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    zip_tract: pd.DataFrame,
    zip_county: pd.DataFrame,
    sources: list[SourceSpec] | None = None,
) -> pd.DataFrame:
    """Assemble the wide (zip, year_month) feature store.

    Parameters
    ----------
    panel : output of arbok.panel.build_panel
    targets : output of arbok.targets.compute_forward_returns (already labeled with splits)
    zip_tract, zip_county : HUD crosswalks (one recent quarter is sufficient)
    sources : optional override of SOURCE_SPECS
    """
    if sources is None:
        sources = SOURCE_SPECS

    # Normalize year_month to month-start Timestamp on both sides; Period[M]
    # round-trips from parquet but doesn't merge against datetime64 adapter output.
    def _norm_ym(df: pd.DataFrame) -> pd.DataFrame:
        if "year_month" in df.columns and str(df["year_month"].dtype).startswith("period"):
            df = df.copy()
            df["year_month"] = df["year_month"].dt.to_timestamp()
        return df

    panel = _norm_ym(panel)
    targets = _norm_ym(targets)
    fs = panel.merge(targets, on=["zip", "cbsa", "year_month"], how="left", suffixes=("", "_t"))
    for spec in sources:
        try:
            piece = _apply_spec(spec, panel, zip_tract, zip_county)
        except FileNotFoundError as e:
            print(f"[skip] {spec.name}: parquet missing — {e}")
            continue
        except NotImplementedError:
            continue
        except Exception as e:  # adapter schema mismatch, KeyError, etc.
            print(f"[skip] {spec.name}: adapter error ({type(e).__name__}: {e})")
            continue
        keys = ["zip", "year_month"] if spec.level != "national" else ["year_month"]
        fs = fs.merge(piece, on=keys, how="left")
        print(f"[joined] {spec.name}: +{piece.shape[1] - len(keys)} cols, {piece.shape[0]} rows")
    return fs


def build_and_save(panel: pd.DataFrame, targets: pd.DataFrame, zip_tract: pd.DataFrame, zip_county: pd.DataFrame) -> pd.DataFrame:
    fs = build_feature_store(panel, targets, zip_tract, zip_county)
    out = PROCESSED / "feature_store_zip_month.parquet"
    fs.to_parquet(out, index=False)
    print(f"Wrote feature store: {out} ({fs.shape[0]:,} rows x {fs.shape[1]} cols)")
    return fs


def load_feature_store() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "feature_store_zip_month.parquet")
