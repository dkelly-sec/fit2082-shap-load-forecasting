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

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# AEMO NEMWEB settings
# --------------------------------------------------------------------------
# NOTE: AEMO decommissioned the old NEMweb HTTP endpoint on 7 Apr 2026 and
# migrated the base URL on 30 Apr 2026. Verify this base URL is still live
# before relying on it -- check https://www.aemo.com.au/energy-systems/
# electricity/national-electricity-market-nem/data-nem/market-data-nemweb
# for the current address if fetch_aemo.py starts failing.
NEMWEB_BASE_URL = "https://nemweb.com.au"

# Half-hourly actual operational demand by region, current reports directory.
# (Only ~13 months are kept here; older data lives in the MMS Data Model
# Archive under a different path -- see fetch_aemo.py docstring.)
NEMWEB_CURRENT_REPORT_PATH = "/Reports/Current/Operational_Demand/ACTUAL_HH/"
NEMWEB_ARCHIVE_REPORT_PATH = "/Reports/Archive/Operational_Demand/ACTUAL_HH/"

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
START_DATE = dt.date(2023, 1, 1)
END_DATE = dt.date(2024, 12, 31)

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
FREQ = "30min"  # half-hourly, matching AEMO demand data resolution
TIMEZONE = "Australia/Melbourne"  # handles AEST/AEDT transitions
