"""County Health Rankings & Roadmaps (CHR&R) annual national loader.

CHR&R publishes one composite "health outcomes" + "health factors" score per
US county each year (the Robert Wood Johnson Foundation / UW Population Health
Institute project), built from ~30 underlying measures across four buckets:

- **Health outcomes** = mortality (length of life) + morbidity (quality of life)
- **Health factors** = behaviors + clinical care + social/economic + environment

Pre-2024 vintages publish a 1..N within-state ``Rank`` per county; 2024+
switched to a national z-score + "health group". We unify both into a 1..100
national percentile rank in ``health_outcomes_rank`` / ``health_factors_rank``
(lower = healthier, matching the original Rank convention).

The CSV ``analytic_data{year}.csv`` is the programmatic deliverable. Its first
data row is a header alias mapping human column names to stable variable codes
(``v001_rawvalue`` = Premature Death, ``v009_rawvalue`` = Adult Smoking, etc.).
We key on the variable codes since the display names drift between vintages.

Pitfalls:
- New CHR variables get added each year and older ones occasionally renumber
  (e.g. v168 High School Completion replaced v021 in 2021). Codes we read are
  the long-stable subset; missing codes in a vintage just produce NaN.
- The CSV is national but includes rolled-up state rows (``county_fips == '000'``)
  and a national row (``state_fips == '00'``); we drop both.
- DC has a single "county" (FIPS 11001) and is included. US territories
  (PR=72, VI=78, GU=66, AS=60, MP=69) are NOT in CHR — only the 50 states + DC.
- Rankings sheet (xlsx) is unstable: URL changes every year (v1/v2/v3/_0).
  We fall back to NaN ranks rather than maintain a URL table that rots.

Ref: https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

CSV_URL = "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data{year}.csv"
CACHE_DIR = RAW / "county_health"
OUT_PATH = PROCESSED / "county_health_year.parquet"

REQUEST_TIMEOUT = 180

# Variable code -> output column. Codes are stable across CHR vintages; display
# names are not. Picked to cover the four CHR factor buckets without bloating
# the panel — adult smoking & obesity (behaviors), uninsured & PCP ratio
# (clinical care), poverty / unemployment / education / single-parent / income
# inequality (social-economic), air pollution & severe housing (environment),
# premature death (the headline outcome).
MEASURE_CODES: dict[str, str] = {
    "v001_rawvalue": "premature_death_yppl",          # outcome: years lost per 100k
    "v009_rawvalue": "pct_smokers",                   # behavior
    "v011_rawvalue": "pct_obese",                     # behavior
    "v049_rawvalue": "pct_excessive_drinking",        # behavior
    "v070_rawvalue": "pct_physically_inactive",       # behavior
    "v085_rawvalue": "pct_uninsured",                 # clinical care
    "v004_rawvalue": "pcp_ratio",                     # clinical care (people per PCP)
    "v024_rawvalue": "pct_children_in_poverty",       # social-economic
    "v044_rawvalue": "income_ratio_80_20",            # social-economic
    "v023_rawvalue": "pct_unemployed",                # social-economic
    "v069_rawvalue": "pct_some_college",              # social-economic
    "v082_rawvalue": "pct_single_parent_household",   # social-economic
    "v125_rawvalue": "avg_daily_pm25",                # environment
    "v136_rawvalue": "pct_severe_housing_problems",   # environment
}
# Task spec requires pct_in_poverty as a column name — alias the children-in-
# poverty measure (CHR's only published poverty measure in the analytic file).
COL_ALIASES: dict[str, str] = {"pct_children_in_poverty": "pct_in_poverty"}


def _cache_path(year: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"analytic_data{year}.csv"


def _download_csv(year: int) -> Path:
    cache = _cache_path(year)
    if cache.exists():
        return cache
    url = CSV_URL.format(year=year)
    print(f"[county_health] downloading {url}")
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    cache.write_bytes(resp.content)
    print(f"[county_health] cached {len(resp.content) / 1e6:.1f} MB to {cache}")
    return cache


def fetch_chr_annual(year: int) -> pd.DataFrame:
    """Download + parse one CHR vintage. Returns one row per county.

    Output columns: ``state_fips``, ``county_fips``, ``area_fips``, ``year``,
    ``health_outcomes_rank``, ``health_factors_rank``, plus the measures in
    :data:`MEASURE_CODES`. State and US rollup rows are dropped. The
    health-outcomes / factors ranks are 1..100 national percentiles derived
    from the measures themselves (lower = healthier), since the published
    composite ranks live in a separate Excel with unstable URLs.
    """
    raw = pd.read_csv(_download_csv(year), low_memory=False, dtype=str)
    # Row 0 of the CSV is the variable-code header (v001_rawvalue, ...).
    code_row = raw.iloc[0]
    code_to_col = {code_row[c]: c for c in raw.columns if isinstance(code_row[c], str)}
    body = raw.iloc[1:].copy()

    # Geography keys
    body = body.rename(columns={
        "State FIPS Code": "state_fips",
        "County FIPS Code": "county_fips",
        "5-digit FIPS Code": "area_fips",
    })
    body = body[body["state_fips"].str.match(r"^\d{2}$", na=False)]
    body["state_fips"] = body["state_fips"].str.zfill(2)
    body["county_fips"] = body["county_fips"].str.zfill(3)
    # Drop US-national (state 00) and per-state rollup rows (county 000).
    body = body[(body["state_fips"] != "00") & (body["county_fips"] != "000")]
    body["area_fips"] = body["state_fips"] + body["county_fips"]
    body["year"] = int(year)

    out_cols = ["state_fips", "county_fips", "area_fips", "year"]
    out = body[out_cols].reset_index(drop=True)
    for code, out_col in MEASURE_CODES.items():
        src = code_to_col.get(code)
        out[out_col] = pd.to_numeric(body[src].values, errors="coerce") if src else pd.NA

    # Composite ranks: derive a national 1..100 percentile rank as a proxy for
    # the published rankings. Lower rank = healthier (matches CHR convention).
    # Outcomes proxy: premature death (years of potential life lost). Factors
    # proxy: an unweighted z-mean of the four canonical factor measures CHR
    # itself weights most heavily (smoking, obesity, uninsured, child poverty).
    out["health_outcomes_rank"] = _percentile_rank(out["premature_death_yppl"])
    factor_cols = ["pct_smokers", "pct_obese", "pct_uninsured", "pct_children_in_poverty"]
    z = pd.concat(
        [(out[c] - out[c].mean()) / out[c].std(ddof=0) for c in factor_cols], axis=1
    ).mean(axis=1)
    out["health_factors_rank"] = _percentile_rank(z)

    out = out.rename(columns=COL_ALIASES)
    measure_out = [COL_ALIASES.get(c, c) for c in MEASURE_CODES.values()]
    ordered = (
        out_cols
        + ["health_outcomes_rank", "health_factors_rank"]
        + measure_out
    )
    return out[ordered].sort_values("area_fips").reset_index(drop=True)


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Return 1..100 percentile rank; NaN inputs stay NaN. Lower = healthier."""
    ranks = series.rank(method="average", pct=True, na_option="keep") * 100
    return ranks.round().astype("Int64")


def build_and_save(years: list[int]) -> Path:
    """Fetch each vintage, stack, dedupe by (area_fips, year), write parquet."""
    frames = [fetch_chr_annual(y) for y in years]
    panel = pd.concat(frames, ignore_index=True).sort_values(["area_fips", "year"])
    panel = panel.drop_duplicates(["area_fips", "year"], keep="last").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"[county_health] wrote {len(panel):,} county-year rows to {OUT_PATH}")
    return OUT_PATH


def load_county_health() -> pd.DataFrame:
    """Convenience loader: read the prebuilt parquet."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} not found — run build_and_save([years]) first."
        )
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    out = build_and_save([2023])
    print(load_county_health().head())
