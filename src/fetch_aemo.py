"""
Fetch AEMO native 5-minute regional demand data using NEMOSIS
(https://github.com/UNSW-CEEM/NEMOSIS), a maintained Python package built
for exactly this purpose. It transparently pulls from AEMO's "Current"
NEMWEB directory or the older MMS Data Model Archive depending on the date
range requested, handles AEMO's row-dispatch CSV format internally, and
caches downloaded files so repeated runs don't re-download from AEMO.

Per Zeehan's meeting direction, this project keeps AEMO's native 5-minute
dispatch resolution as-is -- no resampling to half-hourly. BOM temperature
(daily) and renewables.ninja irradiance (hourly) are both broadcast onto
this 5-minute grid in clean_merge.py.

Usage
-----
    python src/fetch_aemo.py

What this does
---------------
1. Calls nemosis.dynamic_data_compiler for the DISPATCHREGIONSUM table
   over the configured date range (config.START_DATE / config.END_DATE).
2. Filters to the configured REGION (config.REGION, default VIC1).
3. Drops AEMO "intervention" pricing-run duplicate rows (INTERVENTION==1),
   keeping only the physical dispatch run (INTERVENTION==0) for each
   SETTLEMENTDATE, since both can otherwise appear for the same timestamp.
4. Writes data/raw/aemo_demand_raw.csv with columns: timestamp, demand, at
   native 5-minute resolution.

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
"""

from __future__ import annotations
import pandas as pd
from nemosis import dynamic_data_compiler

import config


def fetch_demand() -> pd.DataFrame:
    start_time = config.START_DATE.strftime("%Y/%m/%d %H:%M:%S")
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
        select_columns=["SETTLEMENTDATE", "REGIONID", "TOTALDEMAND", "INTERVENTION"],
        fformat="parquet",
    )

    print(f"Retrieved {len(raw)} rows across all regions.")
    return raw


def filter_native_resolution(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to the configured region and drop AEMO's intervention-pricing-run
    duplicate rows, keeping demand at its native 5-minute dispatch
    resolution (no resampling).
    """
    df = raw[raw["REGIONID"] == config.REGION].copy()
    print(f"Filtered to {config.REGION}: {len(df)} rows.")

    if "INTERVENTION" in df.columns:
        n_before = len(df)
        df = df[df["INTERVENTION"].astype(str).isin(["0", "0.0"])]
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"Dropped {n_dropped} intervention-pricing-run duplicate rows "
                  f"(kept INTERVENTION==0, the physical dispatch run).")

    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df.sort_values("SETTLEMENTDATE").drop_duplicates(subset="SETTLEMENTDATE")

    result = df[["SETTLEMENTDATE", "TOTALDEMAND"]].rename(
        columns={"SETTLEMENTDATE": "timestamp", "TOTALDEMAND": "demand"}
    )
    result["demand"] = pd.to_numeric(result["demand"], errors="coerce")

    n_na = result["demand"].isna().sum()
    print(f"Native 5-minute resolution: {len(result)} rows "
          f"({n_na} rows with missing demand -- handled later by "
          f"clean_merge.py's gap-filling step).")

    return result


def main():
    raw = fetch_demand()
    if raw.empty:
        raise SystemExit(
            "NEMOSIS returned no data. Check your network access to AEMO, "
            "the date range in config.py, and that NEMOSIS is up to date "
            "(pip install --upgrade nemosis)."
        )

    result = filter_native_resolution(raw)

    out_path = config.RAW_DIR / "aemo_demand_raw.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
