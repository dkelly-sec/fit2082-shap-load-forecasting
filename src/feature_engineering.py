"""
Feature engineering for 1-day-ahead LightGBM demand forecasting.

Forecast horizon: 24 hours (288 five-minute intervals), the standard
convention in the short-term load forecasting literature this project
cites (Lee et al. 2023, Van Zyl et al. 2024).

Two categories of feature, handled differently to avoid leakage:

  1. DEMAND-DERIVED features (lags, rolling stats) are computed from the
     ORIGIN time (t, "now") -- only using demand values you'd genuinely
     have in hand at prediction time. None of these look past t.

  2. WEATHER/CALENDAR features are aligned to the TARGET time (t + 24h),
     not the origin. This matches real day-ahead forecasting practice: a
     forecaster knows tomorrow's calendar (is it a public holiday? what
     day of week?) and would use a weather *forecast* for tomorrow, not
     today's actual weather. This project simplifies by using tomorrow's
     *actual observed* weather rather than a genuine weather forecast --
     a standard, documented simplification in this literature (an
     implicit "perfect weather forecast" assumption) since the focus here
     is on demand-side dynamics and SHAP behaviour, not weather
     forecasting itself. Worth one sentence in your report's methodology
     section.

Usage
-----
    python src/feature_engineering.py

Input
-----
    data/interim/merged_full.csv (from clean_merge.py)

Output
------
    data/processed/train_features.csv
    data/processed/val_features.csv
    data/processed/test_features.csv
    data/processed/feature_manifest.json
"""

from __future__ import annotations
import json

import pandas as pd

import config

# 5-minute intervals per day: 24h * 60min / 5min = 288
INTERVALS_PER_DAY = 24 * 60 // 5
FORECAST_HORIZON = INTERVALS_PER_DAY  # 1 day ahead

# Lag intervals for demand-derived features (all look BACKWARD from origin)
LAG_INTERVALS = {
    "lag_5min": 1,
    "lag_10min": 2,
    "lag_15min": 3,
    "lag_1day": INTERVALS_PER_DAY,        # same time yesterday
    "lag_1week": INTERVALS_PER_DAY * 7,   # same time last week
}

# Rolling window sizes (in intervals), trailing/backward-looking ending at origin
ROLLING_WINDOWS = {
    "1h": 12,
    "24h": INTERVALS_PER_DAY,
}

# Columns that describe weather/calendar at a point in time -- these get
# re-aligned to the TARGET time rather than the origin time.
WEATHER_CALENDAR_COLS = [
    "day_of_week", "is_weekend", "is_public_holiday", "season",
    "temp_min", "temp_max", "temperature",
    "irr_electricity", "irr_irradiance_direct", "irr_irradiance_diffuse", "irr_temperature",
]


def load_merged() -> pd.DataFrame:
    path = config.INTERIM_DIR / "merged_full.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run clean_merge.py first.")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_timestamp"] = df["timestamp"].shift(-FORECAST_HORIZON)
    df["target_demand"] = df["demand"].shift(-FORECAST_HORIZON)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, lag in LAG_INTERVALS.items():
        df[name] = df["demand"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, window in ROLLING_WINDOWS.items():
        df[f"roll_mean_{name}"] = df["demand"].rolling(window).mean()
        df[f"roll_std_{name}"] = df["demand"].rolling(window).std()
    return df


def add_time_of_day(df: pd.DataFrame, timestamp_col: str, out_col: str) -> pd.DataFrame:
    df = df.copy()
    ts = df[timestamp_col]
    df[out_col] = ts.dt.hour + ts.dt.minute / 60
    return df


def align_weather_calendar_to_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift each weather/calendar column by -FORECAST_HORIZON so that the
    value stored on row t is that column's value at the TARGET time
    (t + horizon), not at the origin. This is what lets the model use
    "tomorrow's" weather/calendar as a feature for predicting tomorrow's
    demand -- consistent with real day-ahead forecasting practice.
    """
    df = df.copy()
    for col in WEATHER_CALENDAR_COLS:
        if col not in df.columns:
            continue
        df[f"target_{col}"] = df[col].shift(-FORECAST_HORIZON)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_target(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_time_of_day(df, "timestamp", "origin_hour_of_day")
    df = align_weather_calendar_to_target(df)
    df = add_time_of_day(df, "target_timestamp", "target_hour_of_day")
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where the target is missing (end of series, can't train
    without a known target) or where the longest lag feature is missing
    (start of series, no history yet available). This trims a fixed,
    small, and expected number of rows at both ends -- not a data quality
    problem, just the mechanical cost of needing both history and a
    future target for every training row.
    """
    n_before = len(df)
    longest_lag_col = max(LAG_INTERVALS, key=LAG_INTERVALS.get)
    df = df.dropna(subset=["target_demand", longest_lag_col])
    n_after = len(df)
    print(f"Dropped {n_before - n_after} rows lacking a full lag history or a "
          f"target value ({n_before - n_after} = {LAG_INTERVALS[longest_lag_col]} "
          f"(longest lag) + {FORECAST_HORIZON} (horizon), at the start/end of the series).")
    return df.reset_index(drop=True)


def chronological_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * config.TRAIN_FRAC)
    val_end = train_end + int(n * config.VAL_FRAC)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def main():
    df = load_merged()
    print(f"Loaded {len(df)} rows.")

    df = build_features(df)
    df = drop_incomplete_rows(df)
    print(f"{len(df)} rows remain after feature engineering.")

    train, val, test = chronological_split(df)

    train.to_csv(config.PROCESSED_DIR / "train_features.csv", index=False)
    val.to_csv(config.PROCESSED_DIR / "val_features.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test_features.csv", index=False)

    manifest = {
        "forecast_horizon_intervals": FORECAST_HORIZON,
        "forecast_horizon_description": "1 day ahead (288 x 5-minute intervals)",
        "lag_intervals": LAG_INTERVALS,
        "rolling_windows": ROLLING_WINDOWS,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_range": [str(train["timestamp"].min()), str(train["timestamp"].max())],
        "val_range": [str(val["timestamp"].min()), str(val["timestamp"].max())],
        "test_range": [str(test["timestamp"].min()), str(test["timestamp"].max())],
    }
    with open(config.PROCESSED_DIR / "feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Wrote train/val/test features + feature_manifest.json to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
