"""HUD-USPS Vacancy Data loader (quarterly, zip-code level).

Each quarterly HUD-USPS release contains, for both residential and business
addresses at the ZIP level: total addresses, vacant addresses (delivery
suspended >= 90 days), "no-stat" addresses (under construction, demolished,
vacant urban units, etc.), and "active" address counts across 3/6/12/36-month
lookback windows. We use the 12-month active series as a near-real-time proxy
for net household formation / migration; YoY delta (same quarter, prior year)
strips seasonality.

MANUAL DOWNLOAD REQUIRED — DO NOT AUTOMATE. The data live behind a free
HUD User account + per-session token; there is no public direct URL.

  1. Register at https://www.huduser.gov/portal/dataset/uspszip-api.html
  2. Log in at https://www.huduser.gov/portal/datasets/usps.html and pick the
     ZIP-code-level Excel for the desired year + quarter.
  3. Save the file to: data/raw/hud_usps/{year}q{quarter}.xlsx
     (e.g. data/raw/hud_usps/2023q2.xlsx — exact convention required).

HUD occasionally renames columns between releases; `_COLUMN_ALIASES` covers
the variants observed across 2010-2024 vintages. Extend it on KeyError.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from arbok.config import RAW

HUD_USPS_DIR = RAW / "hud_usps"
HUD_USPS_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_DOWNLOAD_URL = "https://www.huduser.gov/portal/datasets/usps.html"

# Canonical -> list of source-column aliases (lowercased, stripped).
# HUD has used both "AMS_RES" / "RES_TOTAL" and similar variants over the years.
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "zip": ("zip", "zipcode", "zip_code"),
    "residential_total": ("ams_res", "res_total", "total_res", "tot_res"),
    "residential_vacant": ("res_vac", "vac_res", "vacant_res"),
    "residential_active": (  # derived if absent: total - vacant - nostat
        "res_active",
        "active_res",
        "ams_res_active",
    ),
    "residential_nostat": ("res_nostat", "nostat_res", "no_stat_res"),
    "business_total": ("ams_bus", "bus_total", "total_bus", "tot_bus"),
    "business_vacant": ("bus_vac", "vac_bus", "vacant_bus"),
    "business_active": ("bus_active", "active_bus", "ams_bus_active"),
    "business_nostat": ("bus_nostat", "nostat_bus", "no_stat_bus"),
}


def _quarter_path(year: int, quarter: int) -> Path:
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"quarter must be 1-4, got {quarter}")
    return HUD_USPS_DIR / f"{year}q{quarter}.xlsx"


def _resolve_column(df_lower_cols: dict[str, str], canonical: str) -> str | None:
    """Return the original column name in df matching any alias for `canonical`."""
    for alias in _COLUMN_ALIASES[canonical]:
        if alias in df_lower_cols:
            return df_lower_cols[alias]
    return None


def load_usps_quarterly(year: int, quarter: int) -> pd.DataFrame:
    """Load a single quarterly HUD-USPS Excel file.

    Reads `data/raw/hud_usps/{year}q{quarter}.xlsx` (manual download — see
    module docstring) and normalises columns to:

        zip, year_quarter, residential_active, residential_vacant,
        residential_total, business_active, business_vacant, business_total

    `residential_active` is derived as `total - vacant - nostat` when the
    release does not expose it directly. `zip` is left-padded to 5 chars.
    """
    path = _quarter_path(year, quarter)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing HUD-USPS file: {path}\n"
            f"Manually download the ZIP-level Excel for {year}Q{quarter} from "
            f"{MANUAL_DOWNLOAD_URL} (free account required) and save it there."
        )

    raw = pd.read_excel(path, dtype=str)
    lower_map = {c.strip().lower(): c for c in raw.columns}

    zip_col = _resolve_column(lower_map, "zip")
    if zip_col is None:
        raise KeyError(f"No ZIP column found in {path.name}. Columns: {list(raw.columns)}")

    def _num(canonical: str) -> pd.Series:
        col = _resolve_column(lower_map, canonical)
        if col is None:
            return pd.Series([pd.NA] * len(raw), dtype="Float64")
        return pd.to_numeric(raw[col], errors="coerce").astype("Float64")

    out = pd.DataFrame(
        {
            "zip": raw[zip_col].astype(str).str.strip().str.zfill(5),
            "year_quarter": f"{year}Q{quarter}",
            "residential_total": _num("residential_total"),
            "residential_vacant": _num("residential_vacant"),
            "residential_active": _num("residential_active"),
            "residential_nostat": _num("residential_nostat"),
            "business_total": _num("business_total"),
            "business_vacant": _num("business_vacant"),
            "business_active": _num("business_active"),
            "business_nostat": _num("business_nostat"),
        }
    )

    # Derive active when missing: total - vacant - nostat.
    for prefix in ("residential", "business"):
        active = f"{prefix}_active"
        total = f"{prefix}_total"
        vacant = f"{prefix}_vacant"
        nostat = f"{prefix}_nostat"
        mask = out[active].isna() & out[total].notna() & out[vacant].notna()
        out.loc[mask, active] = (
            out.loc[mask, total]
            - out.loc[mask, vacant]
            - out.loc[mask, nostat].fillna(0)
        )

    cols = [
        "zip",
        "year_quarter",
        "residential_active",
        "residential_vacant",
        "residential_total",
        "business_active",
        "business_vacant",
        "business_total",
    ]
    return out[cols]


def compute_net_inflow(df_t: pd.DataFrame, df_tminus4: pd.DataFrame) -> pd.DataFrame:
    """YoY change in residential active addresses per ZIP (inflow proxy).

    Inputs are two outputs of `load_usps_quarterly` for the SAME calendar
    quarter in two different years (4 quarters apart — hence the parameter
    name). Returns one row per ZIP with:

        zip, year_quarter, residential_active_t, residential_active_tminus4,
        net_inflow_addresses, net_inflow_pct
    """
    left = df_t[["zip", "year_quarter", "residential_active"]].rename(
        columns={"residential_active": "residential_active_t"}
    )
    right = df_tminus4[["zip", "residential_active"]].rename(
        columns={"residential_active": "residential_active_tminus4"}
    )
    merged = left.merge(right, on="zip", how="inner")
    merged["net_inflow_addresses"] = (
        merged["residential_active_t"] - merged["residential_active_tminus4"]
    )
    merged["net_inflow_pct"] = merged["net_inflow_addresses"] / merged[
        "residential_active_tminus4"
    ].replace(0, pd.NA)
    return merged


def load_all_quarters(years: list[int]) -> pd.DataFrame:
    """Concatenate every available quarterly file for the given years.

    Quietly skips year/quarter combinations whose Excel is not present, so a
    partial download set still yields a usable long panel. Logs (via print)
    which files were skipped so the user can track them down.
    """
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for year in years:
        for quarter in (1, 2, 3, 4):
            path = _quarter_path(year, quarter)
            if not path.exists():
                missing.append(f"{year}Q{quarter}")
                continue
            frames.append(load_usps_quarterly(year, quarter))
    if missing:
        print(f"[hud_usps] skipped {len(missing)} missing files: {', '.join(missing)}")
    if not frames:
        return pd.DataFrame(
            columns=[
                "zip",
                "year_quarter",
                "residential_active",
                "residential_vacant",
                "residential_total",
                "business_active",
                "business_vacant",
                "business_total",
            ]
        )
    return pd.concat(frames, ignore_index=True)
