"""
Unit test for fetch_aemo.filter_and_resample using synthetic 5-minute
DISPATCHREGIONSUM-shaped data. No network access needed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from fetch_aemo import filter_and_resample  # noqa: E402


def _synthetic_raw():
    # 2 hours of 5-minute VIC1 data (24 rows) plus some NSW1 data to be
    # dropped by the region filter.
    ts = pd.date_range("2024-01-01 00:00:00", periods=24, freq="5min")
    vic = pd.DataFrame({
        "SETTLEMENTDATE": ts,
        "REGIONID": "VIC1",
        "TOTALDEMAND": [5000.0 + i * 10 for i in range(24)],
    })
    nsw = pd.DataFrame({
        "SETTLEMENTDATE": ts,
        "REGIONID": "NSW1",
        "TOTALDEMAND": [7000.0] * 24,
    })
    return pd.concat([vic, nsw], ignore_index=True)


def test_filters_to_configured_region():
    raw = _synthetic_raw()
    out = filter_and_resample(raw)
    # every value should have come from the VIC1 series (5000-5230 range),
    # never the NSW1 constant 7000
    assert (out["demand"] < 6000).all()


def test_resamples_to_half_hourly_mean():
    raw = _synthetic_raw()
    out = filter_and_resample(raw)
    # 2 hours of 5-min data -> 4 half-hour buckets
    assert len(out) == 4
    # first half-hour bucket = mean of the first 6 five-minute VIC1 values
    expected_first = sum(5000.0 + i * 10 for i in range(6)) / 6
    assert abs(out.loc[0, "demand"] - expected_first) < 1e-6


def test_output_columns():
    raw = _synthetic_raw()
    out = filter_and_resample(raw)
    assert list(out.columns) == ["timestamp", "demand"]


if __name__ == "__main__":
    test_filters_to_configured_region()
    test_resamples_to_half_hourly_mean()
    test_output_columns()
    print("All fetch_aemo resample tests passed.")
