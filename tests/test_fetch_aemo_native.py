"""
Unit test for fetch_aemo.filter_native_resolution using synthetic
DISPATCHREGIONSUM-shaped data, including AEMO's intervention-pricing-run
duplicate row pattern. No network access needed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from fetch_aemo import filter_native_resolution  # noqa: E402


def _synthetic_raw():
    ts = pd.date_range("2024-01-01 00:00:00", periods=6, freq="5min")
    vic = pd.DataFrame({
        "SETTLEMENTDATE": ts,
        "REGIONID": "VIC1",
        "TOTALDEMAND": [5000.0 + i * 10 for i in range(6)],
        "INTERVENTION": "0",
    })
    vic_intervention = vic.copy()
    vic_intervention["INTERVENTION"] = "1"
    vic_intervention["TOTALDEMAND"] = vic_intervention["TOTALDEMAND"] + 500

    nsw = pd.DataFrame({
        "SETTLEMENTDATE": ts,
        "REGIONID": "NSW1",
        "TOTALDEMAND": [7000.0] * 6,
        "INTERVENTION": "0",
    })

    return pd.concat([vic, vic_intervention, nsw], ignore_index=True)


def test_filters_to_configured_region():
    raw = _synthetic_raw()
    out = filter_native_resolution(raw)
    assert (out["demand"] < 6000).all()


def test_drops_intervention_duplicates():
    raw = _synthetic_raw()
    out = filter_native_resolution(raw)
    assert len(out) == 6
    assert out["demand"].tolist() == [5000.0 + i * 10 for i in range(6)]


def test_preserves_native_5min_resolution():
    raw = _synthetic_raw()
    out = filter_native_resolution(raw)
    diffs = out["timestamp"].diff().dropna().unique()
    assert len(diffs) == 1
    assert pd.Timedelta(diffs[0]) == pd.Timedelta(minutes=5)


def test_output_columns():
    raw = _synthetic_raw()
    out = filter_native_resolution(raw)
    assert list(out.columns) == ["timestamp", "demand"]


if __name__ == "__main__":
    test_filters_to_configured_region()
    test_drops_intervention_duplicates()
    test_preserves_native_5min_resolution()
    test_output_columns()
    print("All fetch_aemo native-resolution tests passed.")
