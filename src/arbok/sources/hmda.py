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
    target = PROCESSED / "hmda_state_year.parquet"
    out.to_parquet(target, index=False)
    return out


# ---------------------------------------------------------------------------
# Tract-level build (the one the assembler actually wants)
# ---------------------------------------------------------------------------
# Minimal LAR columns needed to compute tract-year aggregates. Keeping the
# column set small (a) cuts CSV parse time roughly 5x and (b) keeps the
# per-state response small enough to hold in RAM without streaming-parse.
_LAR_USECOLS = [
    "activity_year",
    "lei",
    "state_code",
    "census_tract",
    "action_taken",
    "loan_purpose",
    "occupancy_type",
    "loan_amount",
]


def _fetch_lar_purchases(year: int, state: str) -> pd.DataFrame:
    """Download HMDA LAR for one state-year filtered to originated home-purchase
    loans, parsing only the columns we aggregate. Cached on disk as gzip CSV
    so re-runs are free.

    Filters at the API: actions_taken=1 (originated), loan_purposes=1 (home
    purchase). This shrinks the CA-2022 payload from ~250 MB to ~30 MB.
    """
    cache = HMDA_RAW / f"lar_{year}_{state}_p1.csv.gz"
    if not cache.exists():
        url = f"{DATA_BROWSER_BASE}/csv"
        params = {
            "years": year,
            "states": state,
            "actions_taken": 1,
            "loan_purposes": 1,
        }
        with requests.get(url, params=params, stream=True, timeout=1800) as r:
            r.raise_for_status()
            # Stream to a temp file then gzip on read; CFPB CSV endpoint isn't
            # gzipped on the wire, but we keep a gzip on disk for cache size.
            import gzip as _gz

            tmp = cache.with_suffix(".csv.gz.tmp")
            with _gz.open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            tmp.rename(cache)
    return pd.read_csv(
        cache,
        compression="gzip",
        usecols=lambda c: c in _LAR_USECOLS,
        dtype={
            "census_tract": "string",
            "lei": "string",
            "state_code": "string",
        },
        low_memory=False,
    )


def _aggregate_tract_year(lar: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one state-year LAR to (tract, year) rows.

    Emits per-tract: count_originated, sum_loan_amount, mean_loan_amount,
    investor_share (occupancy_type != 1, i.e. second-home/investment),
    institutional_share (occupancy_type != 1 AND lender filed >=
    INSTITUTIONAL_LENDER_LOAN_COUNT_THRESHOLD nationally — proxy for
    institutional/SFR-operator buyers since LEI lookup is empty).

    The `tract` column is the 11-digit FIPS GEOID (zero-padded) so the
    feature-store assembler can call aggregate_tract_to_zip on it.
    """
    if lar.empty:
        return pd.DataFrame(
            columns=[
                "tract", "year", "count_originated", "sum_loan_amount",
                "mean_loan_amount", "investor_share", "institutional_share",
            ]
        )
    df = lar.copy()
    df["census_tract"] = df["census_tract"].astype("string").str.strip()
    # CFPB ships census_tract as the full 11-digit GEOID (e.g. "06037206040")
    # but the field is occasionally blank when geocoding failed. Drop those.
    df = df[df["census_tract"].str.fullmatch(r"\d{11}").fillna(False)]
    df["loan_amount"] = pd.to_numeric(df["loan_amount"], errors="coerce")
    df["occupancy_type"] = pd.to_numeric(df["occupancy_type"], errors="coerce")

    # Identify high-volume lenders within this LAR slice. `set(...)` on the
    # pandas Index ensures the resulting union with KNOWN_INSTITUTIONAL_LEIS
    # is a true set; without it, pandas treats the Index as dict-like and the
    # `|` operator raises TypeError.
    lender_counts = df.groupby("lei").size()
    high_volume_leis = set(
        lender_counts[lender_counts >= INSTITUTIONAL_LENDER_LOAN_COUNT_THRESHOLD].index.tolist()
    )
    institutional_leis = high_volume_leis | set(KNOWN_INSTITUTIONAL_LEIS)

    df["_is_nonprimary"] = (df["occupancy_type"] != 1).astype("int8")
    df["_is_institutional"] = (
        (df["occupancy_type"] != 1) & (df["lei"].isin(institutional_leis))
    ).astype("int8")

    grouped = df.groupby("census_tract").agg(
        count_originated=("loan_amount", "size"),
        sum_loan_amount=("loan_amount", "sum"),
        nonprimary_count=("_is_nonprimary", "sum"),
        institutional_count=("_is_institutional", "sum"),
    ).reset_index().rename(columns={"census_tract": "tract"})
    grouped["mean_loan_amount"] = grouped["sum_loan_amount"] / grouped["count_originated"]
    grouped["investor_share"] = grouped["nonprimary_count"] / grouped["count_originated"]
    grouped["institutional_share"] = (
        grouped["institutional_count"] / grouped["count_originated"]
    )
    grouped["year"] = int(lar["activity_year"].iloc[0]) if "activity_year" in lar.columns else None
    return grouped[
        [
            "tract", "year",
            "count_originated", "sum_loan_amount", "mean_loan_amount",
            "investor_share", "institutional_share",
        ]
    ]


def build_tract_year_and_save(
    years: list[int], states: list[str] | None = None, sleep_s: float = 0.25
) -> pd.DataFrame:
    """Build the per-tract-year HMDA panel and save to
    `data/processed/hmda_tract_year.parquet`.

    Schema: tract (11-digit FIPS str), year (int), count_originated,
    sum_loan_amount, mean_loan_amount, investor_share, institutional_share.

    The assembler's `_adapter_hmda` reads this file and the tract-level
    crosswalk (`aggregate_tract_to_zip`) rolls it up to ZIP.

    Filters at the API to originated home-purchase loans only; download per
    state-year is ~1-50 MB after filtering (CA is the largest). Failed
    state-years are logged and skipped, not raised.
    """
    states = states or _US_STATES
    frames: list[pd.DataFrame] = []
    for y in years:
        for st in states:
            try:
                lar = _fetch_lar_purchases(year=y, state=st)
                agg = _aggregate_tract_year(lar)
                if not agg.empty:
                    frames.append(agg)
                    print(f"[hmda] {y} {st}: {len(lar):>7} loans -> {len(agg):>5} tracts")
            except requests.HTTPError as e:
                print(f"[hmda] {y} {st}: HTTP error {e}")
            except Exception as e:  # pragma: no cover - keep build resilient
                print(f"[hmda] {y} {st}: {type(e).__name__}: {e}")
            time.sleep(sleep_s)
    out = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "tract", "year", "count_originated", "sum_loan_amount",
                "mean_loan_amount", "investor_share", "institutional_share",
            ]
        )
    )
    # Same (tract, year) can appear once per state shard at most, but be safe.
    if not out.empty:
        out = (
            out.groupby(["tract", "year"], as_index=False)
            .agg(
                count_originated=("count_originated", "sum"),
                sum_loan_amount=("sum_loan_amount", "sum"),
                investor_share=("investor_share", "mean"),
                institutional_share=("institutional_share", "mean"),
            )
        )
        out["mean_loan_amount"] = out["sum_loan_amount"] / out["count_originated"]
    target = PROCESSED / "hmda_tract_year.parquet"
    out.to_parquet(target, index=False)
    return out
