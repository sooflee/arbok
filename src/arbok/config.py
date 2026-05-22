"""Project paths, horizons, and backtest splits."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

for _d in (RAW, INTERIM, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

TOP_N_METROS = 200

HORIZONS_MONTHS: dict[str, int] = {
    "fwd_6m": 6,
    "fwd_1y": 12,
    "fwd_3y": 36,
    "fwd_5y": 60,
    "fwd_10y": 120,
}

HORIZON_CLASS: dict[str, str] = {
    "fwd_6m": "short",
    "fwd_1y": "short",
    "fwd_3y": "medium",
    "fwd_5y": "long",
    "fwd_10y": "long",
}

# Backtest splits — both run in parallel; results reported per split.
SPLITS: dict[str, dict[str, str]] = {
    "pre_2008": {
        "train_end": "2007-12",
        "test_start": "2008-01",
        "test_end": "2013-12",
    },
    "post_2012": {
        "train_end": "2017-12",
        "test_start": "2018-01",
        "test_end": "2024-12",
    },
}
