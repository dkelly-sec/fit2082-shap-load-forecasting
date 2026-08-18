"""
Central configuration for the FIT2082 data acquisition/cleaning pipeline.

Edit the values in this file rather than hardcoding paths/dates elsewhere,
so every script and notebook stays in sync.
"""

from pathlib import Path
import datetime as dt

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NEMOSIS_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "nemosis_cache"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, NEMOSIS_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# AEMO settings (via the NEMOSIS package)
# --------------------------------------------------------------------------
# NEMOSIS (https://github.com/UNSW-CEEM/NEMOSIS) handles the Current vs.
# Archive split, caching, and AEMO's row-dispatch CSV format internally, so
# we don't need to hand-roll a NEMWEB scraper. It downloads
# DISPATCHREGIONSUM at 5-minute resolution; fetch_aemo.py resamples this
# to half-hourly (mean) to match the project's target resolution.
NEMOSIS_TABLE = "DISPATCHREGIONSUM"

# NEM region of interest
REGION = "VIC1"

# --------------------------------------------------------------------------
# BOM weather settings
# --------------------------------------------------------------------------
# BOM's Climate Data Online does not expose a simple bulk-download API for
# free half-hourly station data -- you request it interactively (or via the
# Weather Station Directory) and download a CSV. Put that manually
# downloaded file here and point BOM_RAW_CSV at it.
BOM_STATION_NAME = "Melbourne Olympic Park"  # change to your chosen station
BOM_STATION_ID = "086338"  # BOM station number, update to match your station
BOM_RAW_CSV = RAW_DIR / "bom_weather_raw.csv"

# --------------------------------------------------------------------------
# Date range (must give >= 2 full years; keep chronological, no shuffling)
# --------------------------------------------------------------------------
START_DATE = dt.date(2024, 1, 1)
END_DATE = dt.date(2025, 12, 31)

# Known anomalous periods to exclude (documented, not silently dropped).
# Add (start, end, reason) tuples as you identify them during EDA.
EXCLUDED_PERIODS = [
    # Example: (dt.date(2020, 3, 1), dt.date(2020, 9, 30), "COVID-19 demand disruption"),
]

# --------------------------------------------------------------------------
# Train / validation / test split (chronological, no leakage)
# --------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
FREQ = "30min"  # half-hourly, matching the project's target resolution
TIMEZONE = "Australia/Melbourne"  # handles AEST/AEDT transitions
