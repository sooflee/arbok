"""FBI Crime Data Explorer (CDE) county-level crime rates loader.

NIBRS (National Incident-Based Reporting System) is the post-2021 successor
to legacy UCR Summary Reporting. Agencies submit incident-level records with
offense type, location type, weapon, and victim-offender relationship. For
per-county RE features we want annual *rates per 100K residents* by category
(violent, property, person, drug) — not the incident detail.
Per `docs/PREDICTORS.md` section 10, NIBRS is a county-level signal.

API:
    Base: https://api.usa.gov/crime/fbi/cde/
    Auth: free api.data.gov key required (https://api.data.gov/signup/);
          export `CDE_API_KEY=...` before calling. DEMO_KEY works for ~30
          calls/hr and is enough to smoke-test one small state-year.
    Docs landing: https://crime-data-explorer.app.cloud.gov/pages/docs
    Endpoints (confirmed 2026-05 via DEMO_KEY probe):
        GET /agencies/byStateAbbr/{abbr}
            -> dict[county_upper] -> list of {ori, counties, is_nibrs,
               agency_name, agency_type_name, nibrs_start_date, ...}.
        GET /summarized/agency/{ori}/{offense}?from=MM-YYYY&to=MM-YYYY
            -> monthly offense actuals + rates + participated_population
               for that agency. `offense` slugs: violent-crime,
               property-crime, homicide, aggravated-assault, burglary,
               motor-vehicle-theft (also larceny, robbery, rape, arson).

No direct county summarized endpoint exists on CDE. We roll up agency
actuals via ORI -> county mapping, then divide by sum(participated_pop) *
1e5. This mirrors the rollup the CDE webapp performs client-side.

Pitfalls:
- **Agency reporting completeness.** Coverage varies wildly by state and
  year. We surface `agencies_reporting` and `population_covered` so
  downstream code can filter sparse county-years.
- **NIBRS transition 2021.** FBI sunset legacy SRS end of 2020. In 2021
  ~40% of agencies (NYPD, LAPD, FL state) had not migrated, producing a
  coverage cliff — national totals for 2021 are not comparable to before
  or after. By 2023 coverage is back >90%. Treat 2021 as a structural
  break, not a real drop.
- **ORI -> county.** Some ORIs span multiple counties (state police,
  university/transit PDs); we assign the whole agency to the first listed
  county. Filter `agency_type_name in {"City","County"}` to drop state-PD
  inflation.
- **County name -> FIPS.** CDE returns uppercased county names with
  non-standard punctuation ("DE KALB" vs "DEKALB", "ST CLAIR" vs
  "ST. CLAIR"). We strip punctuation and join against a cached Census
  county gazetteer.
- **Rate units.** CDE per-agency rate uses the agency's jurisdictional
  pop. For county rollup we recompute sum(actuals)/sum(pop)*1e5 so unequal
  agency sizes are weighted correctly.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

API_BASE = "https://api.usa.gov/crime/fbi/cde"

NIBRS_DIR = RAW / "fbi_nibrs"
NIBRS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED / "fbi_nibrs_county_year.parquet"

# Offense slug -> output column suffix (rate per 100K).
OFFENSES: dict[str, str] = {
    "violent-crime": "violent_crime",
    "property-crime": "property_crime",
    "homicide": "homicide",
    "aggravated-assault": "aggravated_assault",
    "burglary": "burglary",
    "motor-vehicle-theft": "motor_vehicle_theft",
}

# 50 states + DC. Territories (PR/GU/VI) excluded: patchy NIBRS coverage, out of scope.
STATE_ABBRS: tuple[str, ...] = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY")

# Polite rate limiting. api.data.gov default is 1000 req/hr = ~3.6s between
# requests. We pace at 0.2s and rely on caching to keep total volume low.
_REQUEST_DELAY_SEC = 0.2


def _api_key() -> str:
    key = os.environ.get("CDE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CDE_API_KEY not set. Get a free key at https://api.data.gov/signup/ "
            "and export `CDE_API_KEY=...`. DEMO_KEY also works but is ~30 req/hr."
        )
    return key


def _get_json(path: str, params: dict | None = None) -> dict:
    """GET {API_BASE}/{path} with API_KEY appended, parsed JSON."""
    params = dict(params or {})
    params["API_KEY"] = _api_key()
    resp = requests.get(f"{API_BASE}/{path.lstrip('/')}", params=params, timeout=60)
    resp.raise_for_status()
    time.sleep(_REQUEST_DELAY_SEC)
    return resp.json()


def fetch_state_agencies(state_abbr: str) -> pd.DataFrame:
    """ORI -> agency mapping for one state. Cached JSON in data/raw/fbi_nibrs/.

    Columns: ori, county, state_abbr, agency_name, agency_type_name,
    is_nibrs, nibrs_start_date.
    """
    import json
    cache = NIBRS_DIR / f"agencies_{state_abbr}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        payload = _get_json(f"agencies/byStateAbbr/{state_abbr}")
        cache.write_text(json.dumps(payload))
    rows = [
        {"ori": ag.get("ori"), "county": county,
         "state_abbr": ag.get("state_abbr", state_abbr),
         "agency_name": ag.get("agency_name"),
         "agency_type_name": ag.get("agency_type_name"),
         "is_nibrs": ag.get("is_nibrs"),
         "nibrs_start_date": ag.get("nibrs_start_date")}
        for county, agencies in payload.items() for ag in agencies
    ]
    return pd.DataFrame(rows)


def fetch_agency_offense_year(ori: str, offense: str, year: int) -> dict:
    """Return {actuals_sum, pop_mean, months_reporting} for one
    (ori, offense, year). Pop is mean of monthly participated_population —
    matches CDE's rate-per-100K denominator."""
    payload = _get_json(
        f"summarized/agency/{ori}/{offense}",
        params={"from": f"01-{year}", "to": f"12-{year}"},
    )
    offenses_block = payload.get("offenses", {}).get("actuals", {})
    actuals = next((v for k, v in offenses_block.items()
                    if k.endswith("Offenses")), {})
    pop_block = payload.get("populations", {}).get("participated_population", {})
    pops = next((v for v in pop_block.values()), {})
    pop_vals = [float(v) for v in pops.values() if v]
    return {
        "actuals_sum": sum(int(v or 0) for v in actuals.values()),
        "pop_mean": (sum(pop_vals) / len(pop_vals)) if pop_vals else 0.0,
        "months_reporting": sum(1 for v in actuals.values() if v is not None),
    }


_FIPS_URL = "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt"
_FIPS_SUFFIXES = (" County", " Parish", " Borough", " Census Area",
                  " Municipality", " Municipio", " City and Borough", " city")


def _county_name_to_fips() -> pd.DataFrame:
    """Census county gazetteer: state_abbr, state_fips, county_fips, county_key.
    `county_key` is the uppercased name with suffixes and "." stripped, so it
    joins the CDE response (which returns names only)."""
    path = NIBRS_DIR / "census_counties.txt"
    if not path.exists():
        print(f"[fbi_nibrs] downloading county FIPS gazetteer -> {path}")
        resp = requests.get(_FIPS_URL, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    gaz = pd.read_csv(path, sep="|", dtype=str).rename(columns={
        "STATE": "state_abbr", "STATEFP": "state_fips",
        "COUNTYFP": "county_fips", "COUNTYNAME": "county_name_raw",
    })
    name = gaz["county_name_raw"]
    for s in _FIPS_SUFFIXES:
        name = name.str.removesuffix(s)
    gaz["county_key"] = name.str.upper().str.replace(".", "", regex=False)
    return gaz[["state_abbr", "state_fips", "county_fips", "county_key"]]


def _norm_county_key(s: pd.Series) -> pd.Series:
    return (s.fillna("").str.split(",").str[0]
             .str.strip().str.upper().str.replace(".", "", regex=False))


def fetch_county_crime(year: int, states: tuple[str, ...] = STATE_ABBRS) -> pd.DataFrame:
    """County-level crime rates per 100K for one year across `states`.

    Columns: state_fips, county_fips, area_fips, year, violent_crime_per_100k,
    property_crime_per_100k, homicide_per_100k, aggravated_assault_per_100k,
    burglary_per_100k, motor_vehicle_theft_per_100k, agencies_reporting,
    population_covered.
    """
    fips = _county_name_to_fips()
    rows: list[dict] = []
    for state in states:
        agencies = fetch_state_agencies(state)
        agencies["county_key"] = _norm_county_key(agencies["county"])
        for ori, sub in agencies.groupby("ori"):
            rec = {"ori": ori, "state_abbr": state,
                   "county_key": sub.iloc[0]["county_key"]}
            for offense in OFFENSES:
                try:
                    info = fetch_agency_offense_year(ori, offense, year)
                except requests.HTTPError:
                    info = {"actuals_sum": 0, "pop_mean": 0.0, "months_reporting": 0}
                rec[f"{offense}__actuals"] = info["actuals_sum"]
                rec[f"{offense}__pop"] = info["pop_mean"]
            rows.append(rec)

    per_agency = pd.DataFrame(rows)
    # Population-weighted rollup: sum(actuals)/sum(pop)*1e5.
    agg_spec = {"agencies_reporting": ("ori", "count")}
    for slug in OFFENSES:
        agg_spec[f"{slug}__a"] = (f"{slug}__actuals", "sum")
        agg_spec[f"{slug}__p"] = (f"{slug}__pop", "sum")
    agg = per_agency.groupby(["state_abbr", "county_key"], as_index=False).agg(**agg_spec)
    agg["population_covered"] = agg["violent-crime__p"].astype(float)
    for slug, col in OFFENSES.items():
        agg[f"{col}_per_100k"] = (
            agg[f"{slug}__a"] / agg[f"{slug}__p"].replace(0, pd.NA) * 100_000
        ).astype("Float64")
    out = agg.merge(fips, on=["state_abbr", "county_key"], how="left")
    out = out.dropna(subset=["state_fips", "county_fips"])
    out["area_fips"] = out["state_fips"] + out["county_fips"]
    out["year"] = year
    rate_cols = [f"{c}_per_100k" for c in OFFENSES.values()]
    return out[["state_fips", "county_fips", "area_fips", "year",
                *rate_cols, "agencies_reporting", "population_covered"]
              ].reset_index(drop=True)


def build_and_save(years: list[int]) -> Path:
    """Build the county-year crime-rates panel and persist as parquet."""
    frames = [fetch_county_crime(y) for y in years]
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"[fbi_nibrs] wrote {len(panel):,} rows for years={years} to {OUTPUT_PATH}")
    return OUTPUT_PATH


def load_nibrs() -> pd.DataFrame:
    """Convenience loader: read the processed parquet. Raises if not built."""
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUTPUT_PATH} not found. Run `fbi_nibrs.build_and_save([...])` first."
        )
    return pd.read_parquet(OUTPUT_PATH)
