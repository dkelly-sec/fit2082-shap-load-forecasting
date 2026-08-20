"""
Merge cleaned AEMO half-hourly demand data with BOM daily temperature data
into a single half-hourly, feature-complete dataset, then produce a
chronological train/validation/test split.

Weather is daily (see load_bom.py for why), so each day's temp_min/
temp_max/temperature values are broadcast across that day's 48 half-hourly
demand rows -- equivalent to a forward-fill within each day.

Run after fetch_aemo.py and load_bom.py.

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
    df = pd.read_csv(path, parse_dates=["date"])
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
    # Southern hemisphere meteorological seasons
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


def fill_demand_gaps(df: pd.DataFrame, max_gap_intervals: int = 4) -> pd.DataFrame:
    """
    Reindex demand to a complete half-hourly grid and linearly interpolate
    gaps up to `max_gap_intervals` long (default: 4 * 30min = 2 hours).
    Longer gaps are left as NaN and reported, rather than silently
    interpolated -- decide explicitly whether to drop those rows or extend
    this rule. This only touches `demand`; weather columns are handled
    separately since they're daily, not half-hourly (see merge_all).
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
    print(f"Reindexed demand to {len(full_index)} half-hourly slots "
          f"({n_missing_before} were missing before interpolation).")
    print(f"{remaining_na} demand rows still have NaNs after interpolation "
          f"(gaps longer than {max_gap_intervals} intervals) -- inspect these before modelling.")

    return df.reset_index()


def merge_all(demand: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
    demand = demand.copy()
    demand["date"] = demand["timestamp"].dt.normalize()

    merged = pd.merge(demand, weather_daily, on="date", how="left")
    merged = merged.drop(columns=["date"])

    n_missing_weather = merged["temperature"].isna().sum()
    if n_missing_weather:
        print(f"  {n_missing_weather} half-hourly rows have no matching daily weather "
              "(their date wasn't in the BOM files) -- these will have NaN weather "
              "features and are not auto-filled; extend the BOM date range or drop "
              "these rows before modelling.")

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

    merged = merge_all(demand, weather_daily)
    print(f"Merged demand ({len(demand)} rows) with daily weather -> {len(merged)} rows.")

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
