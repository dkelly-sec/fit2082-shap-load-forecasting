"""
Central configuration for the FIT2082 data acquisition/cleaning pipeline.

Edit the values in this file rather than hardcoding paths/dates elsewhere,
so every script and notebook stays in sync.
"""

from pathlib import Path
import datetime as dt
import os

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
# DISPATCHREGIONSUM at its native 5-minute dispatch resolution -- per
# Zeehan's meeting direction, this project keeps that resolution as-is,
# with no resampling to half-hourly.
NEMOSIS_TABLE = "DISPATCHREGIONSUM"

# NEM region of interest
REGION = "VIC1"

# --------------------------------------------------------------------------
# BOM weather settings (temperature)
# --------------------------------------------------------------------------
# BOM's Climate Data Online does not expose a simple bulk-download API for
# free half-hourly station data -- you request it interactively and
# download a CSV. See load_bom.py's docstring for the exact steps.
BOM_STATION_NAME = "Melbourne Olympic Park"
BOM_STATION_ID = "086338"

# --------------------------------------------------------------------------
# Renewables.ninja settings (solar irradiance)
# --------------------------------------------------------------------------
# Free API, but requires a personal account/token -- sign up at
# https://www.renewables.ninja/register, then set the token as an
# environment variable (never commit it to git):
#   Windows (PowerShell): $env:RENEWABLES_NINJA_TOKEN = "your_token_here"
#   Mac/Linux:             export RENEWABLES_NINJA_TOKEN="your_token_here"
RENEWABLES_NINJA_TOKEN = os.environ.get("RENEWABLES_NINJA_TOKEN")

# Coordinates for the same location as the BOM station (Melbourne, Olympic
# Park), so temperature and irradiance describe the same place.
LATITUDE = -37.83
LONGITUDE = 144.98

# Renewables.ninja's high-resolution SARAH satellite dataset only covers
# Europe/Africa/Middle East -- it does NOT cover Australia. Use the
# worldwide (coarser) MERRA-2 reanalysis dataset instead.
RENEWABLES_NINJA_DATASET = "merra2"

# PV simulation parameters (used to derive irradiance-driven output; raw
# irradiance itself is also returned via the API's raw=true option).
# tilt ~ abs(latitude) is a common rule-of-thumb for a fixed panel.
# azim = 0 (north-facing) is correct for the Southern Hemisphere -- the
# opposite of the usual azim=180 (south-facing) convention used for
# Northern Hemisphere examples in renewables.ninja's own docs.
PV_TILT = abs(LATITUDE)
PV_AZIMUTH = 0
PV_SYSTEM_LOSS = 0.1
PV_TRACKING = 0  # 0 = fixed panel, no tracking

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
FREQ = "5min"  # native AEMO dispatch resolution, per Zeehan's meeting direction
TIMEZONE = "Australia/Melbourne"  # handles AEST/AEDT transitions
