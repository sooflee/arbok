"""MIT Election Data & Science Lab — county presidential returns 2000-2024.

Upstream publishes one row per (year, county_fips, candidate, party, mode) with
popular vote count `candidatevotes` and the county-year `totalvotes`. We
aggregate to per-county-year: vote share of each major party + winner +
signed winning margin (positive = Dem, negative = Rep).

Pitfalls baked in here:
- Alaska doesn't report by county; rows are by state-house "DISTRICT N" with
  no real county FIPS — area_fips ends up `02xxx` and won't join Census
  county geos. Downstream consumers should drop or broadcast statewide.
- Kansas City, MO has its own 7-digit place FIPS (2938000) carved out of
  Jackson/Clay/Platte/Cass counties. Kept as-is in `area_fips`; not a county.
- Third-party + write-in candidates collapse into `OTHER`/`GREEN`/`LIBERTARIAN`.
  We don't compute their share; they're implicit in `1 - dem - rep`.
- A few county-years (notably 2020 GA) only report by `mode` (ABSENTEE,
  ELECTION DAY, ...) with no `TOTAL` row. We prefer `TOTAL` rows when present,
  else sum across modes — never both, or we'd double-count.
- `county_fips` arrives as an int that drops the leading state-zero (FIPS
  01001 stored as "1001"). We zfill to 5; 7-digit place codes are preserved.
- Connecticut/Maine/RI "STATEWIDE WRITEIN" buckets have NaN FIPS; dropped.

Ref: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from arbok.config import PROCESSED, RAW

DATAVERSE_DOI = "doi:10.7910/DVN/VOQCHQ"
DATAFILE_ID = 13573089  # countypres_2000-2024.csv (original format)
MIT_ELECTIONS_URL = (
    f"https://dataverse.harvard.edu/api/access/datafile/{DATAFILE_ID}?format=original"
)

MIT_DIR = RAW / "mit_elections"
MIT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = MIT_DIR / "countypres_2000-2024.csv"
OUTPUT_PATH = PROCESSED / "mit_elections_county_year.parquet"

# Required by Dataverse guestbook 458 attached to the dataset. POST returns a
# signed one-shot URL we then GET. Values are recorded in MEDSL's download log.
_GUESTBOOK_PAYLOAD = {
    "guestbookResponse": {
        "name": "arbok loader",
        "email": "bensonw.dev@gmail.com",
        "institution": "Personal research",
        "position": "Researcher",
        "downloadtype": "Original",
    }
}


def _download_csv(dest: Path) -> None:
    """Two-step Dataverse download: POST guestbook -> follow signed URL."""
    sess = requests.Session()
    gb = sess.post(MIT_ELECTIONS_URL, json=_GUESTBOOK_PAYLOAD, timeout=60)
    gb.raise_for_status()
    signed_url = gb.json()["data"]["signedUrl"]
    with sess.get(signed_url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def fetch_elections() -> pd.DataFrame:
    """Download (cached) the MEDSL county presidential returns CSV.

    Returns long-form with columns:
        year, state_fips, county_fips, area_fips, candidate, party,
        candidatevotes, totalvotes

    `area_fips` is the 5-digit county FIPS (zero-padded) for normal counties,
    or the raw 7-digit place FIPS for MO Kansas City. `state_fips` is the
    leading 2 digits of `area_fips`. Drops rows with no FIPS at all.
    """
    if not CACHE_PATH.exists():
        print(f"[mit_elections] downloading {MIT_ELECTIONS_URL} -> {CACHE_PATH}")
        _download_csv(CACHE_PATH)
        print(f"[mit_elections] cached {CACHE_PATH.stat().st_size / 1e6:.1f} MB")

    df = pd.read_csv(CACHE_PATH, dtype={"county_fips": "string"})
    df = df.dropna(subset=["county_fips", "candidatevotes"]).copy()
    # 4-digit FIPS = state code lost its leading zero (state codes 1-9).
    raw_fips = df["county_fips"]
    df["area_fips"] = raw_fips.where(raw_fips.str.len() != 4, "0" + raw_fips)
    df["state_fips"] = df["area_fips"].str[:2]
    df["candidatevotes"] = df["candidatevotes"].astype("int64")
    df["totalvotes"] = df["totalvotes"].astype("int64")
    return df[[
        "year", "state_fips", "county_fips", "area_fips",
        "candidate", "party", "candidatevotes", "totalvotes", "mode",
    ]]


def aggregate_per_county_year(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (area_fips, year) with major-party shares.

    Output columns: `area_fips, year, dem_vote_share, rep_vote_share,
    winner_party, margin`. `margin` is signed: positive = Dem won by that
    fraction of total votes, negative = Rep won. `winner_party` is "DEMOCRAT",
    "REPUBLICAN", or "OTHER" (third-party plurality, rare).
    """
    work = df.copy()

    # Prefer TOTAL rows where they exist; else use the by-mode breakdown
    # and sum. Computed per (area_fips, year) to avoid double-counting the
    # ~750 county-years that carry both.
    has_total = (
        work.assign(_is_total=work["mode"].eq("TOTAL"))
        .groupby(["area_fips", "year"])["_is_total"].any()
    )
    keep_total_mask = work.set_index(["area_fips", "year"]).index.map(has_total)
    work = work[
        (keep_total_mask & work["mode"].eq("TOTAL"))
        | (~keep_total_mask & work["mode"].ne("TOTAL"))
    ]

    keys = ["area_fips", "year"]
    party_votes = (
        work.groupby(keys + ["party"])["candidatevotes"].sum().unstack("party", fill_value=0)
    )
    # totalvotes is constant per county-year in TOTAL rows but can disagree
    # across mode rows; use the per-county-year max as the canonical denominator.
    totals = work.groupby(keys)["totalvotes"].max().rename("totalvotes")

    out = party_votes.join(totals).reset_index()
    dem = out.get("DEMOCRAT", pd.Series(0, index=out.index)).astype("int64")
    rep = out.get("REPUBLICAN", pd.Series(0, index=out.index)).astype("int64")
    denom = out["totalvotes"].replace(0, pd.NA)
    out["dem_vote_share"] = (dem / denom).astype("Float64")
    out["rep_vote_share"] = (rep / denom).astype("Float64")
    out["margin"] = ((dem - rep) / denom).astype("Float64")

    # Winner = party with the most votes in that county-year (handles rare
    # third-party pluralities). Restrict to party columns we actually have.
    party_cols = [c for c in party_votes.columns if c in out.columns]
    out["winner_party"] = out[party_cols].idxmax(axis=1)

    return out[[
        "area_fips", "year",
        "dem_vote_share", "rep_vote_share", "winner_party", "margin",
    ]].sort_values(["area_fips", "year"]).reset_index(drop=True)


def build_and_save() -> Path:
    """End-to-end: fetch (cached) -> aggregate -> parquet."""
    raw = fetch_elections()
    panel = aggregate_per_county_year(raw)
    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"[mit_elections] wrote {len(panel):,} county-year rows to {OUTPUT_PATH}")
    return OUTPUT_PATH


def load_elections() -> pd.DataFrame:
    """Convenience loader: build the parquet if missing, then read it."""
    if not OUTPUT_PATH.exists():
        build_and_save()
    return pd.read_parquet(OUTPUT_PATH)
