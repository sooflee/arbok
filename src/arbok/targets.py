"""Forward-return targets for the zip-month panel.

For each horizon in :data:`arbok.config.HORIZONS_MONTHS` we compute the
forward total appreciation in ZHVI:

    fwd_h = (ZHVI(t + h) / ZHVI(t)) ** (12 / h) - 1   if h >= 12
            ZHVI(t + h) / ZHVI(t) - 1                 otherwise

Horizons >= 12 months are annualized geometrically; the 6-month horizon
is returned as a cumulative return (not annualized) to match
``docs/DESIGN.md``.

TODO(Phase 2): add ZORI-based imputed rent so we can report total
return (price appreciation + rental yield) rather than price-only.
"""
from __future__ import annotations

import logging

import pandas as pd

from arbok.config import HORIZONS_MONTHS, PROCESSED, SPLITS

logger = logging.getLogger(__name__)

TARGETS_PARQUET = PROCESSED / "targets_zip_month.parquet"


def compute_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ``fwd_*`` columns to a zip-month panel and persist to parquet.

    Expects ``panel`` to contain at minimum ``zip``, ``year_month``, ``zhvi``.
    Other columns (``cbsa``, ``zori``) are passed through unchanged.

    Forward values are computed by joining each row with its ``h``-month-ahead
    counterpart inside the same zip; rows without a matching future ZHVI value
    yield ``NaN`` and are kept (downstream code filters them when training).
    """
    required = {"zip", "year_month", "zhvi"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    out = panel.sort_values(["zip", "year_month"]).reset_index(drop=True).copy()

    for col, months in HORIZONS_MONTHS.items():
        future = out[["zip", "year_month", "zhvi"]].copy()
        # Shift the future month *back* by `months` so the join key (year_month)
        # represents the *base* month at which we'd be standing.
        future["year_month"] = future["year_month"] - months
        future = future.rename(columns={"zhvi": "_zhvi_future"})
        merged = out.merge(future, on=["zip", "year_month"], how="left")

        ratio = merged["_zhvi_future"] / merged["zhvi"]
        if months >= 12:
            out[col] = (ratio.pow(12.0 / months) - 1.0).values
        else:
            out[col] = (ratio - 1.0).values

    TARGETS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    persisted = out.copy()
    persisted["year_month"] = persisted["year_month"].dt.to_timestamp()
    persisted.to_parquet(TARGETS_PARQUET, index=False)
    logger.info(
        "Targets: %d rows, %d zips, horizons=%s",
        len(out),
        out["zip"].nunique(),
        list(HORIZONS_MONTHS),
    )
    return out


def add_split_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-split membership booleans for each scheme in :data:`SPLITS`.

    For each split key (``pre_2008``, ``post_2012``) we add two columns:
    ``is_train_<key>`` and ``is_test_<key>``. A row can be in multiple
    splits (one per scheme) because train/test boundaries differ.

    Also adds a coarse summary ``split`` column for quick eyeballing:
    ``train_pre_2008`` / ``test_pre_2008`` / ``train_post_2012`` /
    ``test_post_2012`` (first match wins) or ``unused``.
    """
    if "year_month" not in df.columns:
        raise ValueError("df must have a `year_month` column")
    out = df.copy()
    ym = out["year_month"]
    if not isinstance(ym.dtype, pd.PeriodDtype):
        ym = pd.to_datetime(ym).dt.to_period("M")

    summary = pd.Series("unused", index=out.index, dtype=object)
    for key, bounds in SPLITS.items():
        train_end = pd.Period(bounds["train_end"], freq="M")
        test_start = pd.Period(bounds["test_start"], freq="M")
        test_end = pd.Period(bounds["test_end"], freq="M")
        is_train = ym <= train_end
        is_test = (ym >= test_start) & (ym <= test_end)
        out[f"is_train_{key}"] = is_train
        out[f"is_test_{key}"] = is_test
        # Only fill summary for rows still "unused" so we don't clobber pre_2008.
        summary = summary.where(~(is_train & (summary == "unused")), f"train_{key}")
        summary = summary.where(~(is_test & (summary == "unused")), f"test_{key}")
    out["split"] = summary
    return out
