"""
Merge cleaned AEMO demand data and BOM weather data into a single
half-hourly, feature-complete dataset, then produce a chronological
train/validation/test split.

Run after fetch_aemo.py (or a manual equivalent that produces
data/raw/aemo_demand_raw.csv) and load_bom.py.

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
        raise FileNotFoundError(
            f"{path} not found. Run fetch_aemo.py first (or place an equivalent "
            "CSV with SETTLEMENTDATE / REGIONID / OPERATIONAL_DEMAND columns there)."
        )
    df = pd.read_csv(path)

    # Column names come straight from AEMO's I-row headers; be tolerant of
    # case/exact naming since AEMO table schemas do drift between versions.
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if "settlementdate" in low or (low == "date" or "interval" in low and "date" in low):
            rename_map[col] = "timestamp"
        elif low == "regionid":
            rename_map[col] = "region"
        elif "operational_demand" in low or low == "demand":
            rename_map[col] = "demand"
    df = df.rename(columns=rename_map)

    required = {"timestamp", "region", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"aemo_demand_raw.csv is missing expected columns {missing} after "
            f"renaming. Columns present: {list(df.columns)}. Check fetch_aemo.py's "
            "column mapping matches the table you actually pulled."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["region"] == config.REGION]
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df[["timestamp", "demand"]].sort_values("timestamp").drop_duplicates(subset="timestamp")
    return df


def load_weather() -> pd.DataFrame:
    path = config.INTERIM_DIR / "bom_weather_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run load_bom.py first.")
    df = pd.read_csv(path, parse_dates=["timestamp"])
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


def fill_small_gaps(df: pd.DataFrame, max_gap_intervals: int = 4) -> pd.DataFrame:
    """
    Reindex to a complete half-hourly grid and linearly interpolate gaps up
    to `max_gap_intervals` long (default: 4 * 30min = 2 hours). Longer gaps
    are left as NaN and reported, rather than silently interpolated --
    decide explicitly whether to drop those rows or extend this rule.
    """
    df = df.set_index("timestamp").sort_index()
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=config.FREQ)
    n_missing_before = full_index.difference(df.index).size
    df = df.reindex(full_index)
    df.index.name = "timestamp"

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="linear", limit=max_gap_intervals, limit_area="inside"
    )

    remaining_na = df[numeric_cols].isna().any(axis=1).sum()
    print(f"Reindexed to {len(full_index)} half-hourly slots "
          f"({n_missing_before} were missing before interpolation).")
    print(f"{remaining_na} rows still have NaNs after interpolation "
          f"(gaps longer than {max_gap_intervals} intervals) -- inspect these before modelling.")

    df = df.reset_index()
    # re-derive calendar features for any rows that were pure gaps (they won't have them yet)
    df = add_calendar_features(df)
    return df


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
    weather = load_weather()

    merged = pd.merge(demand, weather, on="timestamp", how="left")
    print(f"Merged demand ({len(demand)} rows) with weather -> {len(merged)} rows.")
    n_missing_weather = merged["temperature"].isna().sum()
    if n_missing_weather:
        print(f"  {n_missing_weather} rows have no matching weather observation "
              "(will be addressed by the gap-filling step below).")

    merged = apply_exclusions(merged)
    merged = fill_small_gaps(merged)

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
