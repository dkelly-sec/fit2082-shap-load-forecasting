"""
Unit tests for clean_merge.py's multi-resolution broadcast merge: daily
BOM temperature and hourly renewables.ninja irradiance both merged onto
native 5-minute AEMO demand. No network access needed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean_merge import merge_weather_and_irradiance, add_calendar_features  # noqa: E402


def _synthetic_demand():
    ts = pd.date_range("2024-01-01 00:00", "2024-01-02 23:55", freq="5min")
    df = pd.DataFrame({"timestamp": ts, "demand": range(len(ts))})
    return add_calendar_features(df)


def _synthetic_weather_daily():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "temp_min": [10.0, 12.0],
        "temp_max": [20.0, 24.0],
        "temperature": [15.0, 18.0],
    })


def _synthetic_irradiance_hourly():
    hours = pd.date_range("2024-01-01 00:00", "2024-01-02 23:00", freq="h")
    return pd.DataFrame({
        "hour": hours,
        "irr_electricity": [float(h.hour) for h in hours],
    })


def test_temperature_broadcasts_across_full_day():
    demand = _synthetic_demand()
    merged = merge_weather_and_irradiance(demand, _synthetic_weather_daily(), _synthetic_irradiance_hourly())
    day1 = merged[merged["timestamp"].dt.date == pd.Timestamp("2024-01-01").date()]
    assert day1["temperature"].nunique() == 1
    assert day1["temperature"].iloc[0] == 15.0
    assert len(day1) == 288  # 24h * 12 five-min intervals


def test_irradiance_broadcasts_across_hour_not_day():
    demand = _synthetic_demand()
    merged = merge_weather_and_irradiance(demand, _synthetic_weather_daily(), _synthetic_irradiance_hourly())
    hour3 = merged[(merged["timestamp"].dt.date == pd.Timestamp("2024-01-01").date()) &
                    (merged["timestamp"].dt.hour == 3)]
    assert hour3["irr_electricity"].nunique() == 1
    assert hour3["irr_electricity"].iloc[0] == 3.0
    assert len(hour3) == 12  # 12 five-min intervals per hour
    # different hour should have a different value -- not day-level broadcast
    hour4 = merged[(merged["timestamp"].dt.date == pd.Timestamp("2024-01-01").date()) &
                    (merged["timestamp"].dt.hour == 4)]
    assert hour4["irr_electricity"].iloc[0] == 4.0


def test_no_data_loss_from_merge():
    demand = _synthetic_demand()
    merged = merge_weather_and_irradiance(demand, _synthetic_weather_daily(), _synthetic_irradiance_hourly())
    assert len(merged) == len(demand)


if __name__ == "__main__":
    test_temperature_broadcasts_across_full_day()
    test_irradiance_broadcasts_across_hour_not_day()
    test_no_data_loss_from_merge()
    print("All merge tests passed.")
