"""
Unit tests for load_bom.py's BOM-format parsing and clean_merge.py's
daily-to-half-hourly broadcast merge. Uses small synthetic files matching
BOM's real IDCJAC00xx format. No network access needed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from load_bom import _load_bom_temp_csvs  # noqa: E402
from clean_merge import merge_all, add_calendar_features  # noqa: E402


def _write_bom_csv(path: Path, code: str, colname: str, dates, values):
    rows = []
    for d, v in zip(dates, values):
        rows.append({
            "Product code": code,
            "Bureau of Meteorology station number": 86338,
            "Year": d.year, "Month": d.month, "Day": d.day,
            colname: v,
            "Days of accumulation": 1,
            "Quality": "Y",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_bom_temp_csvs_parses_real_column_names(tmp_path=None):
    tmp_path = tmp_path or Path("/tmp/bom_test")
    tmp_path.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=3).to_pydatetime()
    path = tmp_path / "IDCJAC0011_086338_2024_Data.csv"
    _write_bom_csv(path, "IDCJAC0011", "Minimum temperature (Degree C)", dates, [10.0, 11.5, 9.2])

    df = _load_bom_temp_csvs([path], "Minimum temperature", "temp_min")
    assert len(df) == 3
    assert list(df.columns) == ["date", "temp_min"]
    assert df["temp_min"].tolist() == [10.0, 11.5, 9.2]


def test_merge_all_broadcasts_daily_weather_across_half_hours():
    ts = pd.date_range("2024-01-01 00:00", "2024-01-02 23:30", freq="30min")
    demand = pd.DataFrame({"timestamp": ts, "demand": range(len(ts))})
    demand = add_calendar_features(demand)

    weather = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "temp_min": [10.0, 12.0],
        "temp_max": [20.0, 24.0],
        "temperature": [15.0, 18.0],
    })

    merged = merge_all(demand, weather)
    day1 = merged[merged["timestamp"].dt.date == pd.Timestamp("2024-01-01").date()]
    day2 = merged[merged["timestamp"].dt.date == pd.Timestamp("2024-01-02").date()]

    assert day1["temperature"].nunique() == 1
    assert day1["temperature"].iloc[0] == 15.0
    assert day2["temperature"].iloc[0] == 18.0
    assert len(day1) == 48  # full half-hourly day


def test_merge_all_flags_missing_weather_dates():
    ts = pd.date_range("2024-01-01 00:00", "2024-01-01 23:30", freq="30min")
    demand = pd.DataFrame({"timestamp": ts, "demand": range(len(ts))})
    demand = add_calendar_features(demand)

    weather = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-01"]),  # no overlap with demand dates
        "temp_min": [5.0], "temp_max": [10.0], "temperature": [7.5],
    })

    merged = merge_all(demand, weather)
    assert merged["temperature"].isna().all()


if __name__ == "__main__":
    test_load_bom_temp_csvs_parses_real_column_names()
    test_merge_all_broadcasts_daily_weather_across_half_hours()
    test_merge_all_flags_missing_weather_dates()
    print("All BOM load/merge tests passed.")
