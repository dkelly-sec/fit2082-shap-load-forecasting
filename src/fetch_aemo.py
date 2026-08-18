"""
Fetch AEMO half-hourly regional demand data using NEMOSIS
(https://github.com/UNSW-CEEM/NEMOSIS), a maintained Python package built
for exactly this purpose. It transparently pulls from AEMO's "Current"
NEMWEB directory or the older MMS Data Model Archive depending on the date
range requested, handles AEMO's row-dispatch CSV format internally, and
caches downloaded files so repeated runs don't re-download from AEMO.

Usage
-----
    python src/fetch_aemo.py

What this does
---------------
1. Calls nemosis.dynamic_data_compiler for the DISPATCHREGIONSUM table
   over the configured date range (config.START_DATE / config.END_DATE).
2. Filters to the configured REGION (config.REGION, default VIC1).
3. Resamples TOTALDEMAND from 5-minute dispatch resolution to half-hourly
   (mean), matching the project's target resolution.
4. Writes data/raw/aemo_demand_raw.csv with columns: timestamp, demand.

Notes
-----
* NEMOSIS downloads real AEMO files into config.NEMOSIS_CACHE_DIR the
  first time you run this, which can be slow and use a fair amount of
  disk space for a 2-year range (5-minute resolution across every NEM
  region, before filtering). Subsequent runs reuse the cache.
* This script has been written against NEMOSIS's documented API but not
  run against the live AEMO site in this environment (AEMO's domain isn't
  reachable from the sandbox this was built in). Run it yourself and
  sanity-check the first few rows before trusting a full pull.
* If NEMOSIS itself starts failing (e.g. AEMO changes its site structure
  again), check https://github.com/UNSW-CEEM/NEMOSIS for updates -- it's
  actively maintained, so a fix is more likely to already exist there
  than in a hand-rolled scraper.
"""

from __future__ import annotations
import pandas as pd
from nemosis import dynamic_data_compiler

import config


def fetch_demand() -> pd.DataFrame:
    start_time = config.START_DATE.strftime("%Y/%m/%d %H:%M:%S")
    # NEMOSIS's end_time is exclusive-ish at the boundary; push to the very
    # end of END_DATE so the last day is fully included.
    end_time = (
        pd.Timestamp(config.END_DATE) + pd.Timedelta(hours=23, minutes=55)
    ).strftime("%Y/%m/%d %H:%M:%S")

    print(f"Fetching {config.NEMOSIS_TABLE} from {start_time} to {end_time} "
          f"(this can take a while on first run while NEMOSIS builds its cache)...")

    raw = dynamic_data_compiler(
        start_time,
        end_time,
        config.NEMOSIS_TABLE,
        str(config.NEMOSIS_CACHE_DIR),
        select_columns=["SETTLEMENTDATE", "REGIONID", "TOTALDEMAND"],
        fformat="parquet",
    )

    print(f"Retrieved {len(raw)} rows across all regions.")
    return raw


def filter_and_resample(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["REGIONID"] == config.REGION].copy()
    print(f"Filtered to {config.REGION}: {len(df)} rows (5-minute resolution).")

    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df.sort_values("SETTLEMENTDATE").set_index("SETTLEMENTDATE")

    half_hourly = df["TOTALDEMAND"].resample(config.FREQ).mean()
    half_hourly = half_hourly.rename("demand").reset_index()
    half_hourly = half_hourly.rename(columns={"SETTLEMENTDATE": "timestamp"})

    n_na = half_hourly["demand"].isna().sum()
    print(f"Resampled to half-hourly: {len(half_hourly)} rows "
          f"({n_na} empty half-hour windows -- these are genuine gaps in the "
          f"source data, handled later by clean_merge.py's gap-filling step).")

    return half_hourly


def main():
    raw = fetch_demand()
    if raw.empty:
        raise SystemExit(
            "NEMOSIS returned no data. Check your network access to AEMO, "
            "the date range in config.py, and that NEMOSIS is up to date "
            "(pip install --upgrade nemosis)."
        )

    half_hourly = filter_and_resample(raw)

    out_path = config.RAW_DIR / "aemo_demand_raw.csv"
    half_hourly.to_csv(out_path, index=False)
    print(f"Wrote {len(half_hourly)} rows to {out_path}")


if __name__ == "__main__":
    main()
