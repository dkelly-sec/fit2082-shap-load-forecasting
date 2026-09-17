"""
Tests for evaluation_sampling.py.

These verify each construction method genuinely does what it claims --
not merely that it returns the right number of rows without crashing.
Synthetic data, no network or trained model needed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation_sampling import (  # noqa: E402
    EVALUATION_METHODS,
    draw_evaluation_sample,
    identify_rare_event_rows,
)

FEATURES = ["demand_proxy", "temp_max", "temp_min"]


def _synthetic_frame(days: int = 400, per_day: int = 24) -> pd.DataFrame:
    """
    Build a frame with a controlled, known composition:
      - four seasons, unevenly sized (so proportional allocation is testable)
      - a small number of public holidays
      - a clear temperature gradient so percentile extremes are unambiguous
    """
    timestamps = pd.date_range("2024-01-01", periods=days * per_day, freq="h")
    day_index = np.arange(len(timestamps)) // per_day

    month = timestamps.month
    season_map = {12: "summer", 1: "summer", 2: "summer",
                  3: "autumn", 4: "autumn", 5: "autumn",
                  6: "winter", 7: "winter", 8: "winter",
                  9: "spring", 10: "spring", 11: "spring"}

    # temp varies smoothly by day so top/bottom percentiles are well-defined
    daily_temp = 20 + 15 * np.sin(day_index * 2 * np.pi / 365)

    return pd.DataFrame({
        "timestamp": timestamps,
        "season": pd.Series(month).map(season_map).to_numpy(),
        "is_public_holiday": (day_index % 97 == 0),  # sparse, ~1% of days
        "temp_max": daily_temp + 5,
        "temp_min": daily_temp - 5,
        "demand_proxy": 4000 + daily_temp * 10,
    })


def test_all_methods_return_exact_size_and_columns():
    frame = _synthetic_frame()
    for method in EVALUATION_METHODS:
        sample = draw_evaluation_sample(frame, FEATURES, size=500, method=method, seed=1)
        assert len(sample) == 500, f"{method} returned {len(sample)} rows"
        assert list(sample.columns) == FEATURES, f"{method} returned wrong columns"


def test_same_seed_reproduces_identical_sample():
    frame = _synthetic_frame()
    for method in EVALUATION_METHODS:
        a = draw_evaluation_sample(frame, FEATURES, size=300, method=method, seed=99)
        b = draw_evaluation_sample(frame, FEATURES, size=300, method=method, seed=99)
        pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_samples():
    """
    Critical for the experiment design: the ~30 repetitions per configuration
    must actually differ, or the 'distribution of rankings' collapses to a
    single point and the whole stability measurement is meaningless.
    """
    frame = _synthetic_frame()
    for method in EVALUATION_METHODS:
        a = draw_evaluation_sample(frame, FEATURES, size=300, method=method, seed=1)
        b = draw_evaluation_sample(frame, FEATURES, size=300, method=method, seed=2)
        assert not a.equals(b), f"{method} produced identical samples for different seeds"


def test_season_stratified_matches_population_proportions_better_than_random():
    """
    The whole point of season_stratified is tighter seasonal representation
    than a chance draw. Verified over several seeds so this isn't a fluke
    of one lucky/unlucky random sample.
    """
    frame = _synthetic_frame()
    population = frame["season"].value_counts(normalize=True).sort_index()

    ctx = FEATURES + ["season"]
    strat_errors, random_errors = [], []
    for seed in range(8):
        strat = draw_evaluation_sample(frame, ctx, size=400, method="season_stratified", seed=seed)
        rand = draw_evaluation_sample(frame, ctx, size=400, method="random", seed=seed)
        strat_prop = strat["season"].value_counts(normalize=True).reindex(population.index, fill_value=0)
        rand_prop = rand["season"].value_counts(normalize=True).reindex(population.index, fill_value=0)
        strat_errors.append(np.abs(strat_prop - population).sum())
        random_errors.append(np.abs(rand_prop - population).sum())

    assert np.mean(strat_errors) < np.mean(random_errors), (
        f"season_stratified mean deviation {np.mean(strat_errors):.4f} was not tighter "
        f"than random's {np.mean(random_errors):.4f}"
    )


def test_rare_event_stratified_hits_requested_fraction():
    frame = _synthetic_frame()
    ctx = FEATURES + ["is_public_holiday", "season"]
    rare_mask = identify_rare_event_rows(frame)
    frame_marked = frame.copy()
    frame_marked["_rare"] = rare_mask

    for fraction in (0.3, 0.5, 0.8):
        sample = draw_evaluation_sample(
            frame_marked, ctx + ["_rare"], size=500,
            method="rare_event_stratified", seed=3, rare_event_fraction=fraction,
        )
        observed = sample["_rare"].mean()
        assert abs(observed - fraction) < 0.02, (
            f"requested rare_event_fraction={fraction}, observed {observed:.3f}"
        )


def test_rare_event_stratified_genuinely_oversamples_versus_natural_rate():
    """
    Confirms the method does something a random draw wouldn't: rare events
    must be meaningfully MORE concentrated than their natural frequency.
    """
    frame = _synthetic_frame()
    natural_rate = identify_rare_event_rows(frame).mean()

    frame_marked = frame.copy()
    frame_marked["_rare"] = identify_rare_event_rows(frame)
    ctx = FEATURES + ["is_public_holiday", "_rare"]

    sample = draw_evaluation_sample(frame_marked, ctx, size=500,
                                     method="rare_event_stratified", seed=5)
    assert sample["_rare"].mean() > natural_rate * 2, (
        f"rare-event rate {sample['_rare'].mean():.3f} is not meaningfully above "
        f"the natural rate {natural_rate:.3f}"
    )


def test_identify_rare_events_flags_holidays_and_temperature_extremes():
    frame = _synthetic_frame()
    mask = identify_rare_event_rows(frame, extreme_percentile=0.05)

    # every public holiday row must be flagged rare
    assert mask[frame["is_public_holiday"]].all(), "some public holiday rows were not flagged rare"

    # the very hottest and very coldest rows must be flagged rare
    hottest = frame["temp_max"].idxmax()
    coldest = frame["temp_min"].idxmin()
    assert mask[hottest], "hottest row was not flagged as a rare event"
    assert mask[coldest], "coldest row was not flagged as a rare event"

    # but not everything is rare -- otherwise the stratification is vacuous
    assert 0 < mask.mean() < 0.5, f"rare-event rate {mask.mean():.3f} is implausible"


def test_unknown_method_raises_clearly():
    frame = _synthetic_frame()
    with pytest.raises(ValueError, match="Unknown evaluation method"):
        draw_evaluation_sample(frame, FEATURES, size=10, method="not_a_method", seed=1)


def test_oversized_request_raises_clearly():
    frame = _synthetic_frame(days=10, per_day=2)  # only 20 rows
    with pytest.raises(ValueError, match="exceeds available rows"):
        draw_evaluation_sample(frame, FEATURES, size=999, method="random", seed=1)


def test_stratified_methods_fail_clearly_without_context_columns():
    """
    If someone passes a frame already restricted to model features (no
    season/temp columns), the stratified methods must say so plainly rather
    than silently falling back to something else.
    """
    frame = _synthetic_frame()[FEATURES + []].copy()  # drops season, is_public_holiday
    with pytest.raises(ValueError, match="season"):
        draw_evaluation_sample(frame, FEATURES, size=50, method="season_stratified", seed=1)


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "-v", __file__])
