"""One-shot script: build feature store from existing panel + targets parquets."""
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from arbok.config import PROCESSED
from arbok.features.assembler import build_and_save
from arbok.features.crosswalks import load_zip_county, load_zip_tract

panel = pd.read_parquet(PROCESSED / "panel_zip_month.parquet")
targets = pd.read_parquet(PROCESSED / "targets_zip_month.parquet")
zip_tract = load_zip_tract(None, None)
zip_county = load_zip_county(None, None)

print(f"panel: {panel.shape}, targets: {targets.shape}")
print(f"zip_tract: {zip_tract.shape}, zip_county: {zip_county.shape}")
print()

fs = build_and_save(panel, targets, zip_tract, zip_county)
print()
print(f"=== feature store: {fs.shape} ===")
print()
print("Per-source coverage (non-null share, first feature col per namespace):")
for ns in ["fred", "rdc", "bps", "qcew", "irs", "acs", "hmda", "fcc", "fema", "nri", "viirs", "fsq"]:
    cols = [c for c in fs.columns if c.startswith(f"{ns}__")]
    if cols:
        first = cols[0]
        nn = fs[first].notna().mean()
        print(f"  {ns:6s} ({len(cols):2d} cols)  {first:42s}  {nn*100:5.1f}% non-null")
    else:
        print(f"  {ns:6s} (skipped)")
