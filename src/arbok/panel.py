"""Zip-month panel assembly: join ZHVI + ZORI + HUD crosswalk, filter to top-200 metros."""
from __future__ import annotations

import logging

import pandas as pd

from arbok.config import PROCESSED
from arbok.sources.hud_crosswalk import dominant_cbsa_per_zip

logger = logging.getLogger(__name__)

PANEL_PARQUET = PROCESSED / "panel_zip_month.parquet"


def build_panel(
    zhvi: pd.DataFrame,
    zori: pd.DataFrame,
    crosswalk: pd.DataFrame,
    top200: pd.DataFrame,
) -> pd.DataFrame:
    """Build the zip-month panel filtered to zips inside top-200 CBSAs.

    Parameters
    ----------
    zhvi, zori
        Tidy long-format frames from :mod:`arbok.sources.zillow` with columns
        ``zip``, ``year_month`` (Period[M]), ``value``.
    crosswalk
        HUD ZIP-CBSA crosswalk (one or many rows per zip).
    top200
        Output of :func:`arbok.sources.census_metros.load_top200`; must have
        a ``cbsa`` column.

    Returns
    -------
    DataFrame with columns ``zip``, ``cbsa``, ``year_month``, ``zhvi``, ``zori``.
    Also saved to :data:`PANEL_PARQUET`.
    """
    zip_to_cbsa = dominant_cbsa_per_zip(crosswalk)
    top_cbsa = set(top200["cbsa"].astype(str).str.zfill(5))
    zip_to_cbsa = zip_to_cbsa[zip_to_cbsa["cbsa"].isin(top_cbsa)]
    logger.info(
        "Crosswalk -> %d zips inside top-%d metros", len(zip_to_cbsa), len(top_cbsa)
    )

    zhvi_p = zhvi.rename(columns={"value": "zhvi"})
    zori_p = zori.rename(columns={"value": "zori"})

    # Outer join on (zip, year_month) so we keep zips with ZHVI but no ZORI
    # (most zips), and the rare ZORI-only month if it ever appears.
    panel = zhvi_p.merge(zori_p, on=["zip", "year_month"], how="outer")
    panel = panel.merge(zip_to_cbsa[["zip", "cbsa"]], on="zip", how="inner")
    panel = (
        panel[["zip", "cbsa", "year_month", "zhvi", "zori"]]
        .sort_values(["zip", "year_month"])
        .reset_index(drop=True)
    )

    PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    # Parquet doesn't natively support pandas Period; cast to month-start Timestamp.
    out = panel.copy()
    out["year_month"] = out["year_month"].dt.to_timestamp()
    out.to_parquet(PANEL_PARQUET, index=False)
    logger.info(
        "Panel: %d rows, %d zips, %d CBSAs, %s -> %s",
        len(panel),
        panel["zip"].nunique(),
        panel["cbsa"].nunique(),
        panel["year_month"].min(),
        panel["year_month"].max(),
    )
    return panel


def load_panel() -> pd.DataFrame:
    """Read the saved panel parquet and restore the year_month Period[M] dtype."""
    if not PANEL_PARQUET.exists():
        raise FileNotFoundError(f"{PANEL_PARQUET} not found. Run build_panel() first.")
    df = pd.read_parquet(PANEL_PARQUET)
    df["year_month"] = pd.to_datetime(df["year_month"]).dt.to_period("M")
    return df
