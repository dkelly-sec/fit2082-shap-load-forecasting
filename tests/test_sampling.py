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


def _frame_with_seasons():
    n = 2880  # exactly 2 full years at daily resolution
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    month = timestamps.month
    season_map = {12: "summer", 1: "summer", 2: "summer",
                  3: "autumn", 4: "autumn", 5: "autumn",
                  6: "winter", 7: "winter", 8: "winter",
                  9: "spring", 10: "spring", 11: "spring"}
    return pd.DataFrame({
        "timestamp": timestamps,
        "season": [season_map[m] for m in month],
        "x": np.sin(np.arange(n) / 20),
        "y": np.cos(np.arange(n) / 30),
    })


def test_season_stratified_returns_exact_ordered_features():
    frame = _frame_with_seasons()
    sample = construct_sample(frame, ["y", "x"], 40, 2082, "season_stratified")
    assert sample.shape == (40, 2)
    assert list(sample.columns) == ["y", "x"]
    assert not sample.duplicated().any()


def test_season_stratified_is_reproducible():
    frame = _frame_with_seasons()
    first = construct_sample(frame, ["x", "y"], 100, 7, "season_stratified")
    second = construct_sample(frame, ["x", "y"], 100, 7, "season_stratified")
    pd.testing.assert_frame_equal(first, second)


def test_season_stratified_matches_population_proportions_better_than_random():
    """
    The actual point of this method: tighter seasonal representation than
    a chance draw, verified over several seeds so this isn't a fluke of
    one lucky/unlucky random sample.
    """
    frame = _frame_with_seasons()
    population = frame["season"].value_counts(normalize=True).sort_index()

    ctx = ["x", "y", "season"]
    strat_errors, random_errors = [], []
    for seed in range(8):
        strat = construct_sample(frame, ctx, 400, seed, "season_stratified")
        rand = construct_sample(frame, ctx, 400, seed, "uniform")
        strat_prop = strat["season"].value_counts(normalize=True).reindex(population.index, fill_value=0)
        rand_prop = rand["season"].value_counts(normalize=True).reindex(population.index, fill_value=0)
        strat_errors.append(np.abs(strat_prop - population).sum())
        random_errors.append(np.abs(rand_prop - population).sum())

    assert np.mean(strat_errors) < np.mean(random_errors), (
        f"season_stratified mean deviation {np.mean(strat_errors):.4f} was not tighter "
        f"than uniform's {np.mean(random_errors):.4f}"
    )


def test_season_stratified_covers_all_four_seasons_even_at_small_size():
    """
    With only 4 strata, even a small evaluation size (e.g. 100, the
    project's smallest planned evaluation size) should meaningfully
    allocate to every season, unlike time_stratified's 48 strata where
    small sizes leave many buckets empty.
    """
    frame = _frame_with_seasons()
    ctx = ["x", "y", "season"]
    sample = construct_sample(frame, ctx, 100, 2082, "season_stratified")
    counts = sample["season"].value_counts()
    assert len(counts) == 4, f"expected all 4 seasons represented, got {list(counts.index)}"
    assert counts.min() >= 15, f"smallest season allocation was {counts.min()}, expected roughly ~25 each"


def test_season_stratified_requires_season_column():
    frame = _frame_with_seasons().drop(columns=["season"])
    with pytest.raises(ValueError, match="season column"):
        construct_sample(frame, ["x", "y"], 50, 1, "season_stratified")


def test_season_stratified_handles_nonzero_source_index():
    frame = _frame_with_seasons()
    test_like_split = frame.iloc[2000:]
    sample = construct_sample(test_like_split, ["x", "y"], 60, 2082, "season_stratified")
    assert sample.shape == (60, 2)
