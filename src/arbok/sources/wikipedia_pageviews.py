"""Wikipedia monthly pageviews per metro — behavioural / search-interest predictor.

Pulls per-article monthly pageviews from the Wikimedia Pageviews REST API
(free, no auth, but requires a non-empty User-Agent). For each CBSA in the
top-200 list, we resolve a Wikipedia article slug (e.g. "Austin,_Texas") and
ask the API for monthly totals across the project history.

Verified endpoint shape (2026-05-21): GET ``.../per-article/en.wikipedia/
all-access/all-agents/<ARTICLE>/monthly/<YYYYMMDD>/<YYYYMMDD>`` returns
``{"items": [{"article", "timestamp" (YYYYMMDDHH), "views", ...}, ...]}``.

Expected runtime for the full top-200: ~25-30s (200 calls x ~0.1s sleep
plus latency). Smoke test (5 metros) runs in ~2s.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED
from arbok.sources.census_metros import load_top200

WIKIPEDIA_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{article}/monthly/{start}/{end}"
)

# Wikimedia rejects empty / generic UAs; identify the project + contact.
USER_AGENT = "arbok-research/0.1 (https://github.com/; contact via repo)"
REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.1
OUT_PATH = PROCESSED / "wikipedia_pageviews_cbsa_month.parquet"

# Two-letter state -> full name, for assembling article slugs like
# "Austin,_Texas". Covers the 50 states + DC (PR not in top-200 metros).
_STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def metro_to_article(metro_name: str) -> str:
    """Convert a CBSA core name (with state suffix) to a Wikipedia article slug.

    Examples
    --------
    >>> metro_to_article("Austin-Round Rock-Georgetown, TX")
    'Austin,_Texas'
    >>> metro_to_article("New York-Newark-Jersey City, NY-NJ")
    'New_York,_New_York'

    Heuristic: take the first city before the first dash, append the first
    state from the multi-state suffix. Works for the vast majority of CBSAs;
    edge cases (e.g. "Washington" -> "Washington,_D.C.") may need manual
    overrides in a follow-up if pageview counts come back zero.
    """
    if "," in metro_name:
        core, state_blob = metro_name.rsplit(",", 1)
    else:
        core, state_blob = metro_name, ""
    primary_city = core.split("-", 1)[0].strip()
    primary_state_code = state_blob.strip().split("-", 1)[0].strip()
    state_full = _STATE_NAMES.get(primary_state_code, primary_state_code)
    slug_city = primary_city.replace(" ", "_")
    slug_state = state_full.replace(" ", "_")
    return f"{slug_city},_{slug_state}"


def fetch_article_pageviews(
    article: str,
    start: str = "20100101",
    end: str = "20241201",
) -> pd.DataFrame:
    """Pull monthly pageviews for a single article.

    Returns a DataFrame with columns ``year_month`` (Timestamp on month start),
    ``views`` (int), ``article`` (str). Returns an empty frame on 404 (the
    article doesn't exist) so the outer loop can keep going.
    """
    url = WIKIPEDIA_URL.format(article=requests.utils.quote(article, safe=",_"),
                               start=start, end=end)
    resp = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
    )
    if resp.status_code == 404:
        return pd.DataFrame(columns=["year_month", "views", "article"])
    resp.raise_for_status()
    items = resp.json().get("items", [])
    if not items:
        return pd.DataFrame(columns=["year_month", "views", "article"])
    df = pd.DataFrame(items)
    # timestamp is "YYYYMMDDHH" — slice to month-start.
    df["year_month"] = pd.to_datetime(df["timestamp"].str[:6], format="%Y%m")
    df["views"] = pd.to_numeric(df["views"]).astype("int64")
    df["article"] = article
    return df[["year_month", "views", "article"]]


def fetch_for_metros(top200_df: pd.DataFrame) -> pd.DataFrame:
    """Loop the top-200 list, fetch each article's monthly pageviews, concat.

    Adds the source ``cbsa`` so the frame can be joined back into the panel.
    Articles with no Wikipedia page (404) are skipped silently — the caller
    can detect them via missing CBSAs in the output.
    """
    frames: list[pd.DataFrame] = []
    for row in top200_df.itertuples(index=False):
        metro_name = f"{row.core_name}, {row.state}"
        article = metro_to_article(metro_name)
        try:
            df = fetch_article_pageviews(article)
        except requests.RequestException as e:
            print(f"  skip {row.cbsa} ({article}): {e}")
            time.sleep(SLEEP_SECONDS)
            continue
        if df.empty:
            print(f"  no data {row.cbsa} ({article})")
        else:
            df = df.assign(cbsa=row.cbsa)
            frames.append(df)
        time.sleep(SLEEP_SECONDS)
    if not frames:
        return pd.DataFrame(columns=["year_month", "views", "article", "cbsa"])
    return pd.concat(frames, ignore_index=True)


def build_and_save() -> Path:
    """Fetch pageviews for every top-200 metro and persist to parquet.

    Expected runtime ~25-30s for 200 metros (1 request each + 0.1s sleep).
    """
    top = load_top200()
    panel = fetch_for_metros(top)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH)
    return OUT_PATH


def load_pageviews() -> pd.DataFrame:
    """Read the persisted monthly Wikipedia pageviews panel."""
    if not OUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUT_PATH} not found — run build_and_save() first."
        )
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    # Smoke test: pull 5 known metros and print head/tail.
    top = load_top200()
    smoke_cbsas = {
        "35620": "New York",
        "31080": "Los Angeles",
        "16980": "Chicago",
        "12420": "Austin",
        "14260": "Boise",
    }
    subset = top[top["cbsa"].astype(str).isin(smoke_cbsas)]
    print(f"smoke: fetching {len(subset)} metros")
    out = fetch_for_metros(subset)
    print(out.groupby("cbsa")["views"].agg(["count", "mean", "max"]))
    print(out.head())
