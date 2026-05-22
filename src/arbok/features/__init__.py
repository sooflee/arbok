"""Feature-store assembler.

Joins every source onto the (zip, year_month) panel spine with crosswalks
and vintage-lag adjustment, producing a single wide parquet ready for modeling.
"""
