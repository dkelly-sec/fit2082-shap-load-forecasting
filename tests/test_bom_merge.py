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
from load_bom import _load_bom_temp_csvs, load_bom_daily  # noqa: E402
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


def test_load_bom_daily_does_not_average_a_single_present_reading(tmp_path=None, monkeypatch=None):
    """
    A day with min present but max missing (or vice versa) must NOT end up
    with temperature == the single present reading (pandas' default
    skipna mean would otherwise silently do this). It should instead be
    filled from a neighbouring complete day.
    """
    tmp_path = tmp_path or Path("/tmp/bom_test_gap")
    tmp_path.mkdir(parents=True, exist_ok=True)

    min_dir = tmp_path / "IDCJAC0011_086338_2025"
    max_dir = tmp_path / "IDCJAC0010_086338_2025"
    min_dir.mkdir(exist_ok=True)
    max_dir.mkdir(exist_ok=True)

    dates = pd.date_range("2025-01-28", periods=4).to_pydatetime()
    _write_bom_csv(min_dir / "IDCJAC0011_086338_2025_Data.csv", "IDCJAC0011",
                    "Minimum temperature (Degree C)", dates, [16.0, 15.5, 15.5, 17.0])
    # max missing on the middle two days
    max_rows = []
    for d, v in zip(dates, [28.0, None, None, 30.0]):
        max_rows.append({
            "Product code": "IDCJAC0010", "Bureau of Meteorology station number": 86338,
            "Year": d.year, "Month": d.month, "Day": d.day,
            "Maximum temperature (Degree C)": v, "Days of accumulation": 1, "Quality": "Y",
        })
    pd.DataFrame(max_rows).to_csv(max_dir / "IDCJAC0010_086338_2025_Data.csv", index=False)

    orig_raw_dir = config.RAW_DIR
    orig_start, orig_end = config.START_DATE, config.END_DATE
    config.RAW_DIR = tmp_path
    config.START_DATE = dates[0].date()
    config.END_DATE = dates[-1].date()
    try:
        df = load_bom_daily()
    finally:
        config.RAW_DIR = orig_raw_dir
        config.START_DATE, config.END_DATE = orig_start, orig_end

    day29 = df[df["date"] == pd.Timestamp("2025-01-29")].iloc[0]
    # must NOT equal temp_min alone (15.5) -- that would mean the buggy
    # skipna-mean behaviour crept back in
    assert day29["temperature"] != 15.5
    # should instead match the nearest complete day's mean (28th: (16+28)/2=22.0)
    assert day29["temperature"] == 22.0


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
    test_load_bom_daily_does_not_average_a_single_present_reading()
    test_merge_all_broadcasts_daily_weather_across_half_hours()
    test_merge_all_flags_missing_weather_dates()
    print("All BOM load/merge tests passed.")
