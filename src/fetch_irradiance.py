"""
Fetch hourly solar irradiance and PV output data from Renewables.ninja
(https://www.renewables.ninja) for the project's location (Melbourne,
matching the BOM station coordinates in config.py).

Renewables.ninja requires a free account and personal API token -- sign up
at https://www.renewables.ninja/register, find your token on your account
page, then set it as an environment variable (never commit it to git):

    Windows (PowerShell): $env:RENEWABLES_NINJA_TOKEN = "your_token_here"
    Mac/Linux:             export RENEWABLES_NINJA_TOKEN="your_token_here"

Usage
-----
    python src/fetch_irradiance.py

What this does
---------------
1. Calls the renewables.ninja PV API (api/data/pv) with raw=true, once per
   calendar year in the configured date range, at config.LATITUDE /
   config.LONGITUDE, using the MERRA-2 dataset (config.RENEWABLES_NINJA_DATASET
   -- the SARAH dataset does NOT cover Australia, only Europe/Africa/
   Middle East, so this must stay merra2 for a Melbourne location).
2. raw=true returns extra raw-weather columns (irradiance components)
   alongside the simulated PV electricity output.
3. Writes data/raw/irradiance_raw.csv with hourly resolution.

Notes
-----
* Free-tier accounts are rate-limited (a small number of requests per
  hour) -- this script fetches one year at a time with a short pause
  between calls, and will raise a clear error if you hit the API's rate
  limit or if your token/network isn't set up correctly.
* This script has been written against the documented API but not run
  against the live service in this environment (renewables.ninja isn't
  reachable from the sandbox this was built in). Run it yourself and
  sanity-check the first few rows before trusting a full pull.
* If you don't have a token yet, sign up first -- see the docstring above.
"""

from __future__ import annotations
import sys
import time

import pandas as pd
import requests

import config

API_BASE = "https://www.renewables.ninja/api/data/pv"


def _parse_renewables_ninja_index(idx_series: pd.Series) -> pd.Series:
    """
    Renewables.ninja's JSON response uses the row index as the timestamp,
    but the exact representation isn't guaranteed to be a plain date
    string -- in practice it can come through as Unix epoch milliseconds
    (e.g. "1704067200000" for 2024-01-01 00:00:00 UTC). Try that first,
    falling back to normal date-string parsing if the values aren't
    purely numeric.
    """
    try:
        as_int = idx_series.astype("int64")
        return pd.to_datetime(as_int, unit="ms", utc=True).dt.tz_localize(None)
    except (ValueError, TypeError):
        return pd.to_datetime(idx_series)


def _fetch_year(year: int) -> pd.DataFrame:
    date_from = f"{year}-01-01"
    date_to = f"{year}-12-31"

    params = {
        "lat": config.LATITUDE,
        "lon": config.LONGITUDE,
        "date_from": date_from,
        "date_to": date_to,
        "dataset": config.RENEWABLES_NINJA_DATASET,
        "capacity": 1,
        "system_loss": config.PV_SYSTEM_LOSS,
        "tracking": config.PV_TRACKING,
        "tilt": config.PV_TILT,
        "azim": config.PV_AZIMUTH,
        "raw": "true",
        "local_time": "true",
        "format": "json",
    }
    headers = {"Authorization": f"Token {config.RENEWABLES_NINJA_TOKEN}"}

    print(f"Requesting {date_from} to {date_to} at "
          f"({config.LATITUDE}, {config.LONGITUDE}), dataset={config.RENEWABLES_NINJA_DATASET}...")
    resp = requests.get(API_BASE, params=params, headers=headers, timeout=120)

    if resp.status_code == 401:
        raise RuntimeError(
            "Renewables.ninja returned 401 Unauthorized -- check that "
            "RENEWABLES_NINJA_TOKEN is set correctly in your environment."
        )
    if resp.status_code == 429:
        raise RuntimeError(
            "Renewables.ninja returned 429 Too Many Requests -- you've hit "
            "the free-tier rate limit. Wait a while (their limits reset "
            "hourly) and try again."
        )
    resp.raise_for_status()

    payload = resp.json()
    data = payload.get("data", payload)  # header=true (default) nests under "data"
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "timestamp_utc"
    df = df.reset_index()
    df["timestamp_utc"] = _parse_renewables_ninja_index(df["timestamp_utc"])

    return df


def fetch_irradiance() -> pd.DataFrame:
    if not config.RENEWABLES_NINJA_TOKEN:
        raise RuntimeError(
            "RENEWABLES_NINJA_TOKEN is not set. Sign up at "
            "https://www.renewables.ninja/register, find your token on your "
            "account page, and set it as an environment variable -- see the "
            "module docstring for the exact commands."
        )

    years = range(config.START_DATE.year, config.END_DATE.year + 1)
    frames = []
    for i, year in enumerate(years):
        df = _fetch_year(year)
        frames.append(df)
        print(f"  got {len(df)} hourly rows for {year}")
        if i < len(years) - 1:
            time.sleep(6)  # be polite to the free-tier rate limit between calls

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="timestamp_utc")
    combined = combined.sort_values("timestamp_utc")

    # local_time=true adds a local_time column -- use that as our working
    # timestamp so it lines up with AEMO's local-time timestamps.
    if "local_time" in combined.columns:
        # local_time strings carry a UTC offset that changes across the
        # AEST/AEDT daylight-saving transition (+10:00 vs +11:00), so they
        # can't be parsed directly into one naive column -- parse with
        # utc=True first (handles the mixed offsets), then convert to a
        # single consistent local timezone, then drop the tz info.
        combined["timestamp"] = (
            pd.to_datetime(combined["local_time"], utc=True)
            .dt.tz_convert(config.TIMEZONE)
            .dt.tz_localize(None)
        )
    else:
        combined["timestamp"] = combined["timestamp_utc"]

    n_before = len(combined)
    combined = combined[(combined["timestamp"].dt.date >= config.START_DATE) &
                         (combined["timestamp"].dt.date <= config.END_DATE)]
    print(f"Filtered to configured date range: {len(combined)} of {n_before} rows kept.")

    out_path = config.RAW_DIR / "irradiance_raw.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} hourly rows to {out_path}")
    print(f"Columns available: {list(combined.columns)}")
    return combined


if __name__ == "__main__":
    try:
        fetch_irradiance()
    except RuntimeError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        sys.exit(1)
