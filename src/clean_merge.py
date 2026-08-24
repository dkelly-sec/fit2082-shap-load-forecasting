"""
Merge AEMO native 5-minute demand data with BOM daily temperature and
renewables.ninja hourly irradiance data into a single feature-complete
dataset, then produce a chronological train/validation/test split.

Weather is daily (BOM) and irradiance is hourly (renewables.ninja), so
each is broadcast across the relevant 5-minute demand rows: temperature
across all of that day's rows, irradiance across all of that hour's rows.

Run after fetch_aemo.py, load_bom.py, and fetch_irradiance.py.

Usage
-----
    python src/clean_merge.py

Outputs
-------
    data/interim/merged_full.csv        -- merged, feature-complete, full range
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
    data/processed/split_manifest.json  -- exact cutoff dates, for reproducibility
"""

from __future__ import annotations
import json

import holidays
import pandas as pd

import config

# Columns from the renewables.ninja raw=true response worth keeping as
# features (beyond the simulated PV "electricity" output itself). Exact
# column names depend on the dataset/API version -- this script keeps any
# numeric column it finds rather than hardcoding names, and prints what it
# kept so you can check it matches what you expected.
IRRADIANCE_ID_COLS = {"timestamp_utc", "local_time", "timestamp"}


def load_aemo_demand() -> pd.DataFrame:
    path = config.RAW_DIR / "aemo_demand_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fetch_aemo.py first.")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    required = {"timestamp", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"aemo_demand_raw.csv is missing expected columns {missing}. "
            f"Columns present: {list(df.columns)}."
        )

    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    return df


def load_weather_daily() -> pd.DataFrame:
    path = config.INTERIM_DIR / "bom_weather_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run load_bom.py first.")
    return pd.read_csv(path, parse_dates=["date"])


def load_irradiance_hourly() -> pd.DataFrame:
    path = config.RAW_DIR / "irradiance_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fetch_irradiance.py first.")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                     if c not in IRRADIANCE_ID_COLS]
    if not numeric_cols:
        raise ValueError(
            f"No numeric feature columns found in {path} besides id columns. "
            f"Columns present: {list(df.columns)}."
        )
    print(f"Using irradiance feature columns: {numeric_cols}")

    df["hour"] = df["timestamp"].dt.floor("h")
    df = df[["hour"] + numeric_cols].drop_duplicates(subset="hour").sort_values("hour")
    df = df.rename(columns={c: f"irr_{c}" for c in numeric_cols})
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = Monday
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    au_vic_holidays = holidays.country_holidays("AU", subdiv="VIC",
                                                  years=range(config.START_DATE.year,
                                                              config.END_DATE.year + 2))
    df["is_public_holiday"] = df["timestamp"].dt.date.isin(au_vic_holidays)

    month = df["timestamp"].dt.month
    season_map = {12: "summer", 1: "summer", 2: "summer",
                  3: "autumn", 4: "autumn", 5: "autumn",
                  6: "winter", 7: "winter", 8: "winter",
                  9: "spring", 10: "spring", 11: "spring"}
    df["season"] = month.map(season_map)

    return df


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    if not config.EXCLUDED_PERIODS:
        return df
    mask = pd.Series(False, index=df.index)
    for start, end, reason in config.EXCLUDED_PERIODS:
        window = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
        n = window.sum()
        print(f"Excluding {n} rows for '{reason}' ({start} .. {end})")
        mask |= window
    return df[~mask]


def fill_demand_gaps(df: pd.DataFrame, max_gap_intervals: int = 24) -> pd.DataFrame:
    """
    Reindex demand to a complete 5-minute grid and linearly interpolate
    gaps up to `max_gap_intervals` long (default: 24 * 5min = 2 hours).
    Longer gaps are left as NaN and reported, not silently interpolated.
    """
    df = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=config.FREQ)
    n_missing_before = full_index.difference(df.index).size
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    df["demand"] = df["demand"].interpolate(
        method="linear", limit=max_gap_intervals, limit_area="inside"
    )

    remaining_na = df["demand"].isna().sum()
    print(f"Reindexed demand to {len(full_index)} five-minute slots "
          f"({n_missing_before} were missing before interpolation).")
    print(f"{remaining_na} demand rows still have NaNs after interpolation "
          f"(gaps longer than {max_gap_intervals} intervals) -- inspect these before modelling.")

    return df.reset_index()


def merge_weather_and_irradiance(demand: pd.DataFrame, weather_daily: pd.DataFrame,
                                   irradiance_hourly: pd.DataFrame) -> pd.DataFrame:
    demand = demand.copy()
    demand["date"] = demand["timestamp"].dt.normalize()
    demand["hour"] = demand["timestamp"].dt.floor("h")

    merged = pd.merge(demand, weather_daily, on="date", how="left")
    merged = pd.merge(merged, irradiance_hourly, on="hour", how="left")
    merged = merged.drop(columns=["date", "hour"])

    n_missing_temp = merged["temperature"].isna().sum() if "temperature" in merged.columns else 0
    if n_missing_temp:
        print(f"  {n_missing_temp} rows have no matching daily temperature "
              "(date not in BOM files) -- extend the BOM date range or drop these rows.")

    irr_cols = [c for c in merged.columns if c.startswith("irr_")]
    if irr_cols:
        n_missing_irr = merged[irr_cols[0]].isna().sum()
        if n_missing_irr:
            print(f"  {n_missing_irr} rows have no matching hourly irradiance "
                  "(hour not in renewables.ninja data) -- extend the date range or drop these rows.")

    return merged


def chronological_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * config.TRAIN_FRAC)
    val_end = train_end + int(n * config.VAL_FRAC)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    manifest = {
        "train_range": [str(train["timestamp"].min()), str(train["timestamp"].max())],
        "val_range": [str(val["timestamp"].min()), str(val["timestamp"].max())],
        "test_range": [str(test["timestamp"].min()), str(test["timestamp"].max())],
        "train_frac": config.TRAIN_FRAC,
        "val_frac": config.VAL_FRAC,
        "test_frac": config.TEST_FRAC,
    }
    return train, val, test, manifest


def main():
    demand = load_aemo_demand()
    demand = fill_demand_gaps(demand)
    demand = apply_exclusions(demand)
    demand = add_calendar_features(demand)

    weather_daily = load_weather_daily()
    irradiance_hourly = load_irradiance_hourly()

    merged = merge_weather_and_irradiance(demand, weather_daily, irradiance_hourly)
    print(f"Merged demand ({len(demand)} rows) with weather + irradiance -> {len(merged)} rows.")

    full_path = config.INTERIM_DIR / "merged_full.csv"
    merged.to_csv(full_path, index=False)
    print(f"Wrote full merged dataset ({len(merged)} rows) to {full_path}")

    train, val, test, manifest = chronological_split(merged)
    train.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(config.PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    with open(config.PROCESSED_DIR / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Manifest written to {config.PROCESSED_DIR / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
