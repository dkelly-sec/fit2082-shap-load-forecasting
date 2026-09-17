import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sampling import (
    construct_sample,
    define_outcome_demand_events,
    define_rare_events,
    outcome_demand_mask,
    rare_event_mask,
)


@pytest.fixture
def frame():
    n = 2400
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "demand": np.linspace(3000, 8000, n),
        "temperature": 18 + 12 * np.sin(np.arange(n) * 2 * np.pi / (24 * 30)),
        "is_public_holiday": np.arange(n) % 503 == 0,
        "x": np.sin(np.arange(n) / 20),
        "y": np.cos(np.arange(n) / 30),
    })


@pytest.mark.parametrize("method", ["uniform", "time_stratified", "kmeans"])
def test_methods_return_exact_ordered_features(frame, method):
    sample = construct_sample(frame, ["y", "x"], 40, 2082, method)
    assert sample.shape == (40, 2)
    assert list(sample.columns) == ["y", "x"]
    assert not sample.duplicated().any()


def test_uniform_is_reproducible(frame):
    first = construct_sample(frame, ["x", "y"], 50, 7, "uniform")
    second = construct_sample(frame, ["x", "y"], 50, 7, "uniform")
    pd.testing.assert_frame_equal(first, second)


def test_rare_thresholds_are_derived_from_training_only(frame):
    training = frame.iloc[:1200]
    definition = define_rare_events(training)
    assert definition.temperature_low == pytest.approx(training["temperature"].quantile(0.05))
    assert definition.temperature_high == pytest.approx(training["temperature"].quantile(0.95))
    assert "demand_high" not in definition.to_dict()


def test_rare_event_stratification_uses_fixed_definition(frame):
    definition = define_rare_events(frame.iloc[:1200])
    sample = construct_sample(
        frame, ["is_public_holiday", "temperature", "x"], 100, 12,
        "rare_event_stratified", definition, rare_fraction=0.5,
    )
    assert rare_event_mask(sample, definition).sum() == 50


def test_public_holiday_is_a_primary_rare_event(frame):
    definition = define_rare_events(frame.iloc[:1200])
    holiday = frame.iloc[[503]].copy()
    holiday["temperature"] = 18.0
    assert rare_event_mask(holiday, definition).iloc[0]


def test_future_demand_is_separate_outcome_conditioned_cohort(frame):
    training = frame.iloc[:1200]
    definition = define_outcome_demand_events(training)
    assert definition.demand_high == pytest.approx(training["demand"].quantile(0.95))
    sample = construct_sample(
        frame, ["demand", "x"], 100, 12, "outcome_demand_stratified",
        outcome_definition=definition, rare_fraction=0.5,
    )
    assert outcome_demand_mask(sample, definition).sum() == 50


def test_unknown_method_fails(frame):
    with pytest.raises(ValueError, match="Unknown sampling method"):
        construct_sample(frame, ["x"], 10, 1, "not-a-method")


def test_time_stratified_handles_nonzero_source_index(frame):
    test_like_split = frame.iloc[1700:]
    sample = construct_sample(test_like_split, ["x", "y"], 60, 2082, "time_stratified")
    assert sample.shape == (60, 2)
