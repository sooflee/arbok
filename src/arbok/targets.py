"""Forward-return targets for the zip-month panel.

For each horizon in :data:`arbok.config.HORIZONS_MONTHS` we compute the
forward total appreciation in ZHVI:

    fwd_h = (ZHVI(t + h) / ZHVI(t)) ** (12 / h) - 1   if h >= 12
            ZHVI(t + h) / ZHVI(t) - 1                 otherwise

Horizons >= 12 months are annualized geometrically; the 6-month horizon
is returned as a cumulative return (not annualized) to match
``docs/DESIGN.md``.

We also emit *total-return* targets that add a constant rent-yield carry to
the price return:

    rent_yield_t        = (ZORI_t * 12) / ZHVI_t       # gross annualized yield
    fwd_<h>y_total      = fwd_<h>y + rent_yield_t       # for h in {1, 3, 5}

The rent-yield carry is held constant across the horizon — i.e. we assume
the gross yield observed at base month ``t`` persists for the entire
forward window. This is an approximation: ZORI evolves over time, owners
incur vacancy/maintenance/tax costs, and the price-return component already
implicitly captures rent dynamics through ZHVI. Treat the total-return
targets as a first-order ranking signal rather than a true IRR.

Zip-months without ZORI coverage (many small zips lack ZORI) receive
``NaN`` for the total-return columns rather than falling back to the
price-only return. Downstream code should drop NaN-target rows for the
total-return models.
"""
from __future__ import annotations

import logging

import pandas as pd

from arbok.config import HORIZONS_MONTHS, PROCESSED, SPLITS

logger = logging.getLogger(__name__)

TARGETS_PARQUET = PROCESSED / "targets_zip_month.parquet"

# Horizons (in years) for which we emit a `fwd_<n>y_total` total-return column.
# We only emit total-return targets for annualized horizons (1y, 3y, 5y); the
# 6-month price return and the speculative 10y horizon are left as price-only.
TOTAL_RETURN_HORIZONS_YEARS: tuple[int, ...] = (1, 3, 5)


def compute_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ``fwd_*`` columns to a zip-month panel and persist to parquet.

    Expects ``panel`` to contain at minimum ``zip``, ``year_month``, ``zhvi``.
    Other columns (``cbsa``, ``zori``) are passed through unchanged.

    Forward values are computed by joining each row with its ``h``-month-ahead
    counterpart inside the same zip; rows without a matching future ZHVI value
    yield ``NaN`` and are kept (downstream code filters them when training).

    If a ``zori`` column is present we additionally emit:

    * ``rent_yield_t``  — gross annualized rental yield ``12 * zori / zhvi``
      at base month ``t`` (NaN where ZORI is missing or ZHVI is zero/missing).
    * ``fwd_1y_total``, ``fwd_3y_total``, ``fwd_5y_total`` — the annualized
      price-return target plus the constant ``rent_yield_t`` carry. NaN
      whenever *either* the price-return or the rent yield is NaN.
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

    # Total-return targets: price return + constant rent-yield carry. Requires
    # ZORI; otherwise we skip and only emit the price-only targets.
    if "zori" in out.columns:
        # Guard against zero/negative ZHVI before division (none expected in
        # real data, but a divide-by-zero would silently yield inf).
        zhvi_safe = out["zhvi"].where(out["zhvi"] > 0)
        rent_yield = (out["zori"] * 12.0) / zhvi_safe
        out["rent_yield_t"] = rent_yield
        for years in TOTAL_RETURN_HORIZONS_YEARS:
            price_col = f"fwd_{years}y"
            total_col = f"fwd_{years}y_total"
            if price_col not in out.columns:
                logger.warning(
                    "Skipping %s: base price-return column %s not present",
                    total_col, price_col,
                )
                continue
            # NaN propagates naturally: if either rent_yield or the price
            # return is NaN, total is NaN (no fallback to price-only).
            out[total_col] = out[price_col] + rent_yield
    else:
        logger.warning(
            "Panel has no `zori` column; skipping total-return targets."
        )

    TARGETS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    persisted = out.copy()
    persisted["year_month"] = persisted["year_month"].dt.to_timestamp()
    persisted.to_parquet(TARGETS_PARQUET, index=False)
    total_cols = [f"fwd_{y}y_total" for y in TOTAL_RETURN_HORIZONS_YEARS
                  if f"fwd_{y}y_total" in out.columns]
    logger.info(
        "Targets: %d rows, %d zips, horizons=%s, total-return cols=%s",
        len(out),
        out["zip"].nunique(),
        list(HORIZONS_MONTHS),
        total_cols,
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
