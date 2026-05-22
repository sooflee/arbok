"""HMDA (Home Mortgage Disclosure Act) mortgage records.

Source: CFPB Data Browser, https://ffiec.cfpb.gov/data-browser/data/
The Loan/Application Register (LAR) is reported by lenders annually and
is published at **census-tract granularity**. Tract IDs are 11-digit
(2-digit state + 3-digit county + 6-digit tract).

Key LAR fields: loan_amount (dollars), income (thousands; applicant
income), action_taken (1=originated), loan_purpose (1=purchase,
2=improvement, 31/32=refi/cash-out, 4=other, 5=N/A), occupancy_type
(1=principal, 2=second, 3=investment), derived_dwelling_category, lei
(20-char lender ID), census_tract (11-digit FIPS).

WARNING — file sizes: the full nationwide LAR is multi-GB per year
(~20M+ rows). Single states can be hundreds of MB (CA, TX, FL, NY).
Always cache to disk; `fetch_hmda_lar` writes to data/raw/hmda/ and
re-reads from cache on subsequent calls. For tract-level summaries,
download the LAR once and aggregate locally — the CFPB aggregations
endpoint does NOT support a `tracts` parameter (only state/county/MSA).
CFPB does not publish rate limits but throttles aggressively on parallel
queries; keep requests sequential with backoff.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from arbok.config import RAW, PROCESSED

DATA_BROWSER_BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view"
HMDA_RAW = RAW / "hmda"
HMDA_RAW.mkdir(parents=True, exist_ok=True)

# Known large institutional / SFR-operator LEIs. These are 20-char Legal
# Entity Identifiers; expand as you find them. Used in
# derive_institutional_buyer_share() as an OR with the asset-size proxy.
# Sources: SEC filings, HMDA panel disclosures, news.
KNOWN_INSTITUTIONAL_LEIS: set[str] = {
    # Wells Fargo Bank NA, Rocket Mortgage, etc. — populate as research progresses.
    # Leaving the seed set empty intentionally rather than guessing wrong LEIs.
}

# Loan-amount proxy for "institutional" lenders when LEI lookup is missing.
# Per-loan size doesn't identify institutional buyers directly, but a
# lender filing >5,000 originations in a single year is functionally an
# institutional player. We use the lender-count threshold downstream.
INSTITUTIONAL_LENDER_LOAN_COUNT_THRESHOLD = 5_000


# ---------------------------------------------------------------------------
# Aggregations endpoint
# ---------------------------------------------------------------------------
def fetch_hmda_aggregations(
    year: int,
    state: str | None = None,
    action_taken: int = 1,
) -> pd.DataFrame:
    """Fetch HMDA aggregations from the CFPB Data Browser API.

    IMPORTANT API LIMITATION (verified 2026-05-21):
    The aggregations endpoint does NOT support a `tracts` parameter; the
    only geographic dimensions are `states`, `counties`, and `msamds`.
    To get true tract-level summaries you must download the LAR CSV and
    aggregate locally.

    What this function DOES return: one row per (state, action_taken,
    loan_purpose) combination, with `count` (number of records) and
    `sum` (sum of loan amounts in dollars). loan_purpose is included as
    a grouping dimension so you can isolate purchase loans (=1) from
    refis. We don't get medians from the aggregations endpoint — only
    counts and sums — so "median loan amount" is approximated as
    mean = sum / count.

    Args:
        year: HMDA reporting year (e.g. 2022).
        state: Two-letter state abbreviation, or None for nationwide.
        action_taken: 1 = originated (default). See module docstring.

    Returns:
        DataFrame with columns: year, state, action_taken, loan_purpose,
        count, sum_loan_amount, mean_loan_amount.
    """
    if state is None:
        url = f"{DATA_BROWSER_BASE}/nationwide/aggregations"
        params: dict = {
            "years": year,
            "actions_taken": action_taken,
            # Group by loan_purpose by passing every value:
            "loan_purposes": "1,2,31,32,4,5",
        }
    else:
        url = f"{DATA_BROWSER_BASE}/aggregations"
        params = {
            "years": year,
            "states": state,
            "actions_taken": action_taken,
            "loan_purposes": "1,2,31,32,4,5",
        }

    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    body = r.json()

    rows = []
    for agg in body.get("aggregations", []):
        count = agg.get("count", 0)
        sum_loan = float(agg.get("sum", 0.0))
        rows.append(
            {
                "year": year,
                "state": state or "US",
                "action_taken": int(agg.get("actions_taken", action_taken)),
                "loan_purpose": int(agg.get("loan_purposes", 0))
                if agg.get("loan_purposes") is not None
                else None,
                "count": int(count),
                "sum_loan_amount": sum_loan,
                "mean_loan_amount": (sum_loan / count) if count else None,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full LAR (large download — see warnings)
# ---------------------------------------------------------------------------
def fetch_hmda_lar(year: int, state: str, **filters) -> pd.DataFrame:
    """Download a full HMDA LAR CSV for ONE state in ONE year.

    *** This is a large download. ***
    California 2022 originations alone are ~250MB compressed. Always
    cache. Do NOT call this for many states in a loop without a queue
    and disk-space check.

    Cached at: data/raw/hmda/lar_{year}_{state}.csv.gz
    Re-reads from disk if the cache file exists.

    `filters` are passed through to the CSV endpoint and may include
    `actions_taken`, `loan_purposes`, `loan_types`, `dwelling_categories`,
    etc. Each value is comma-joined per CFPB API conventions.
    """
    cache = HMDA_RAW / f"lar_{year}_{state}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False, compression="gzip")

    url = f"{DATA_BROWSER_BASE}/csv"
    params: dict = {"years": year, "states": state}
    for k, v in filters.items():
        params[k] = ",".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v

    # Stream to disk to avoid loading multi-hundred-MB body into RAM.
    with requests.get(url, params=params, stream=True, timeout=1800) as r:
        r.raise_for_status()
        with cache.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                if chunk:
                    f.write(chunk)
    return pd.read_csv(cache, low_memory=False, compression="gzip")


# ---------------------------------------------------------------------------
# Derived metric: institutional-buyer share
# ---------------------------------------------------------------------------
def derive_institutional_buyer_share(lar: pd.DataFrame) -> pd.DataFrame:
    """Share of loans that are *institutional non-owner-occupied purchases*.

    Heuristic (documented because no single field flags 'iBuyer'):
      - loan_purpose == 1            (home purchase, not refi)
      - occupancy_type != 1          (not principal residence — i.e.
                                      second home or investment property)
      - lender is "institutional", where institutional means:
          (a) lei is in KNOWN_INSTITUTIONAL_LEIS, OR
          (b) the lender filed >= INSTITUTIONAL_LENDER_LOAN_COUNT_THRESHOLD
              originations nationally in the same LAR — i.e. high-volume.
        (a) is preferred; (b) is a coarse proxy when LEI lookup is
        incomplete. The CFPB lender-panel asset-size field is a better
        signal when joined in, but requires a separate file we don't
        load here.

    Returns a per-tract DataFrame:
        census_tract, total_purchases, institutional_purchases,
        institutional_buyer_share
    """
    needed = {"loan_purpose", "occupancy_type", "lei", "census_tract"}
    missing = needed - set(lar.columns)
    if missing:
        raise ValueError(f"LAR missing required columns: {missing}")

    # Build the high-volume lender set from this LAR
    lender_counts = lar.groupby("lei").size()
    high_volume_leis = set(
        lender_counts[lender_counts >= INSTITUTIONAL_LENDER_LOAN_COUNT_THRESHOLD].index
    )
    institutional_leis = high_volume_leis | KNOWN_INSTITUTIONAL_LEIS

    purchases = lar[lar["loan_purpose"] == 1].copy()
    purchases["is_institutional_nonprimary"] = (
        (purchases["occupancy_type"] != 1)
        & (purchases["lei"].isin(institutional_leis))
    )

    grouped = purchases.groupby("census_tract").agg(
        total_purchases=("loan_purpose", "size"),
        institutional_purchases=("is_institutional_nonprimary", "sum"),
    )
    grouped["institutional_buyer_share"] = (
        grouped["institutional_purchases"] / grouped["total_purchases"]
    )
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# Multi-year aggregations build
# ---------------------------------------------------------------------------
# All 50 states + DC + PR. Drop PR if you don't want territories.
_US_STATES: list[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


def build_aggregations_and_save(
    years: list[int], states: list[str] | None = None
) -> pd.DataFrame:
    """Run the aggregations endpoint per state per year, concat, save parquet.

    Sequential and slow-ish (~50 states * N years HTTP calls) but each
    call returns ~6 rows so total payload is trivial. Adds a tiny sleep
    between calls to avoid hammering CFPB.
    """
    states = states or _US_STATES
    frames: list[pd.DataFrame] = []
    for y in years:
        for st in states:
            try:
                df = fetch_hmda_aggregations(year=y, state=st, action_taken=1)
                frames.append(df)
            except requests.HTTPError as e:
                # Log and continue — one bad state-year shouldn't kill the run
                print(f"[hmda] {y} {st}: HTTP error {e}")
            time.sleep(0.25)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    target = PROCESSED / "hmda_tract_year.parquet"
    out.to_parquet(target, index=False)
    return out
