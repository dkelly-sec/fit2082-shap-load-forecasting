"""
Load and combine Bureau of Meteorology daily temperature data.

BOM's free Climate Data Online download only offers DAILY resolution for
temperature (no free half-hourly/hourly station archive over a multi-year
span -- that requires a paid Data Services request). This project uses
daily minimum and maximum temperature instead, forward-filled across each
day's half-hourly slots in clean_merge.py. Humidity was not freely
available for this station either, so it's out of scope for this project
-- worth a line in your report/EDA noting the substitution.

Expected input
--------------
Each BOM download comes as a folder (BOM zips it; your browser/OS may
auto-extract it) named like:

    IDCJAC0011_086338_2024/
        IDCJAC0011_086338_2024_Data.csv   <- the actual data
        IDCJAC0011_086338_2024_Note.txt   <- ignored

    IDCJAC0011_*  = daily minimum temperature
    IDCJAC0010_*  = daily maximum temperature

Place all four folders (min/max x 2024/2025, or however many years you
downloaded) directly inside data/raw/ -- this script finds them
automatically by matching the IDCJAC0010_*/IDCJAC0011_* folder name
pattern, so you don't need to list exact filenames anywhere.

Usage
-----
    python src/load_bom.py

Output
------
    data/interim/bom_weather_clean.csv
    columns: date, temp_min, temp_max, temperature (mean of min/max)
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

import config

MIN_TEMP_PATTERN = "IDCJAC0011_*"
MAX_TEMP_PATTERN = "IDCJAC0010_*"


def _find_data_csvs(pattern: str) -> list[Path]:
    folders = sorted(config.RAW_DIR.glob(pattern))
    csvs = []
    for folder in folders:
        if not folder.is_dir():
            continue
        matches = list(folder.glob("*_Data.csv"))
        if not matches:
            print(f"  warning: no *_Data.csv found inside {folder}")
            continue
        csvs.append(matches[0])
    return csvs


def _load_bom_temp_csvs(csv_paths: list[Path], value_keyword: str, out_col: str) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        value_col = next((c for c in df.columns if value_keyword.lower() in c.lower()), None)
        if value_col is None:
            raise ValueError(
                f"Could not find a column containing '{value_keyword}' in {path}. "
                f"Columns present: {list(df.columns)}."
            )
        df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]))
        df = df[["date", value_col]].rename(columns={value_col: out_col})
        frames.append(df)
        print(f"  loaded {len(df)} rows from {path.name}")

    if not frames:
        return pd.DataFrame(columns=["date", out_col])

    combined = pd.concat(frames, ignore_index=True)
    combined[out_col] = pd.to_numeric(combined[out_col], errors="coerce")
    combined = combined.sort_values("date").drop_duplicates(subset="date")
    return combined


def load_bom_daily() -> pd.DataFrame:
    print("Looking for minimum temperature files...")
    min_files = _find_data_csvs(MIN_TEMP_PATTERN)
    if not min_files:
        raise FileNotFoundError(
            f"No folders matching '{MIN_TEMP_PATTERN}' found in {config.RAW_DIR}. "
            "Download minimum temperature from BOM Climate Data Online and place "
            "the extracted folder(s) directly in data/raw/."
        )
    min_temp = _load_bom_temp_csvs(min_files, "Minimum temperature", "temp_min")

    print("Looking for maximum temperature files...")
    max_files = _find_data_csvs(MAX_TEMP_PATTERN)
    if not max_files:
        raise FileNotFoundError(
            f"No folders matching '{MAX_TEMP_PATTERN}' found in {config.RAW_DIR}. "
            "Download maximum temperature from BOM Climate Data Online and place "
            "the extracted folder(s) directly in data/raw/."
        )
    max_temp = _load_bom_temp_csvs(max_files, "Maximum temperature", "temp_max")

    merged = pd.merge(min_temp, max_temp, on="date", how="outer").sort_values("date")

    n_before = len(merged)
    merged = merged[(merged["date"].dt.date >= config.START_DATE) &
                     (merged["date"].dt.date <= config.END_DATE)]
    print(f"Filtered to configured date range: {len(merged)} of {n_before} rows kept.")

    n_missing_either = merged[["temp_min", "temp_max"]].isna().any(axis=1).sum()
    if n_missing_either:
        print(f"  warning: {n_missing_either} days are missing min and/or max temperature")

    merged["temperature"] = merged[["temp_min", "temp_max"]].mean(axis=1)

    out_path = config.INTERIM_DIR / "bom_weather_clean.csv"
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} daily rows to {out_path}")
    return merged


if __name__ == "__main__":
    try:
        load_bom_daily()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        sys.exit(1)
