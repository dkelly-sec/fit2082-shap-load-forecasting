"""Focused RQ2 invariants, including temporal pairing and duplicate groups."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reliability import (  # noqa: E402
    cohort_masks, duplicate_features, evaluation_rows, fixed_ablation_params,
    paired_day_bootstrap, permute_group, predict_ordered, rank_agreement, strict_metrics, validate_groups,
)
from sampling import define_outcome_demand_events, define_rare_events  # noqa: E402


def test_group_permutation_uses_one_donor_and_does_not_mutate():
    values = np.column_stack([np.arange(100), np.arange(100), np.arange(100) + 7.0])
    original = values.copy()
    changed = permute_group(values, [0, 1], np.random.default_rng(9))
    np.testing.assert_array_equal(changed[:, 0], changed[:, 1])
    np.testing.assert_array_equal(changed[:, 2], values[:, 2])
    np.testing.assert_array_equal(values, original)
    assert not np.array_equal(changed, values)


def test_restricted_permutation_does_not_cross_cohort():
    values = np.arange(100).reshape(-1, 1)
    strata = values[:, 0] < 30
    changed = permute_group(values, [0], np.random.default_rng(7), strata)
    assert (changed[strata] < 30).all()
    assert (changed[~strata] >= 30).all()


def test_groups_partition_features_and_do_not_omit_duplicate():
    validate_groups(["a", "b"], {"lag": ["a", "b"]})
    with pytest.raises(ValueError, match="partition"):
        validate_groups(["a", "b"], {"lag": ["a"]})
    with pytest.raises(ValueError, match="partition"):
        validate_groups(["a", "b"], {"one": ["a", "b"], "two": ["b"]})
    assert duplicate_features(pd.DataFrame({"a": [1., np.nan], "b": [1., np.nan], "c": [3., 4.]}),
                              ["a", "b", "c"]) == [["a", "b"]]


def test_nonfinite_metrics_fail_instead_of_dropping_rows():
    with pytest.raises(ValueError, match="Non-finite"):
        strict_metrics([1., 2.], np.array([1., np.nan]), 1.)


def test_named_prediction_keeps_order_and_rejects_nonfinite():
    class Model:
        def predict(self, frame, num_threads):
            assert list(frame) == ["b", "a"] and num_threads == 4
            return frame.b.to_numpy()

    assert predict_ordered(Model(), [[3., 1.]], ["b", "a"], 4).tolist() == [3.]
    with pytest.raises(ValueError, match="Non-finite"):
        predict_ordered(Model(), [[np.inf, 1.]], ["b", "a"], 4)


def test_rank_agreement_rejects_infinity():
    with pytest.raises(ValueError, match="finite"):
        rank_agreement(pd.Series([1., 2.]), pd.Series([1., np.inf]))


@pytest.mark.parametrize("method", ["uniform", "time_stratified", "season_stratified",
                                    "rare_event_stratified", "outcome_demand_stratified"])
def test_reconstructed_sample_retains_correct_targets(method):
    n = 500
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
                          "temperature": np.arange(n) % 35, "is_public_holiday": np.arange(n) % 100 == 0,
                          "demand": np.arange(n) + 5000., "x": np.arange(n) * 2.,
                          "season": "summer"})
    features = ["x", "temperature", "is_public_holiday"]
    rare, outcome = define_rare_events(frame), define_outcome_demand_events(frame)
    sample, positions = evaluation_rows(frame, features, method, 100, 2082, rare, outcome, 100000)
    assert len(np.unique(positions)) == 100
    assert list(sample) == features
    np.testing.assert_equal(sample.x.to_numpy(), positions * 2)
    # Selecting through retained row IDs gives the ORIGINAL y, not another sample.
    np.testing.assert_equal(frame.iloc[positions].demand.to_numpy(), positions + 5000)


def test_cohorts_use_origin_temperature_not_target_demand():
    frame = pd.DataFrame({"temperature": [1., 20., 40.], "is_public_holiday": [False, True, False],
                          "demand": [9000., 100., 100.]})
    definition = define_rare_events(pd.DataFrame({"temperature": np.arange(10, 31),
                                                 "is_public_holiday": False}))
    assert cohort_masks(frame, definition)["rare"].tolist() == [True, True, True]


def test_day_bootstrap_preserves_constant_paired_contrast():
    dates = pd.Series(pd.date_range("2024-01-01", periods=48, freq="h"))
    rare = np.arange(48) < 24
    delta = np.where(rare, 4., 1.)
    masks = {"all": np.ones(48, dtype=bool), "rare": rare, "ordinary": ~rare}
    result = paired_day_bootstrap(dates, delta, masks, 500, 9)
    assert result["rare_minus_ordinary"]["delta_mae_mw"] == 3
    assert result["rare_minus_ordinary"]["ci_low_mw"] == 3
    assert result["rare_minus_ordinary"]["ci_high_mw"] == 3
    assert result["rare"]["origin_day_clusters"] == 1
    assert result["rare_minus_ordinary"]["bootstrap_valid"] < 500


def test_agreement_uses_signed_pfi_and_average_ties():
    left = pd.Series([3., 2., 1.], index=["a", "b", "c"])
    right = pd.Series([-3., -2., -1.], index=["a", "b", "c"])
    assert rank_agreement(left, right)["spearman"] == pytest.approx(-1.)
    assert rank_agreement(left, left)["spearman"] == pytest.approx(1.)


def test_fixed_params_do_not_mutate_frozen_estimator():
    class Model:
        def __init__(self):
            self.params = {"n_estimators": 1500, "random_state": 2082, "n_jobs": -1,
                           "objective": "regression_l1", "learning_rate": .03}

        def get_params(self, deep=False):
            return self.params

    model = Model()
    params = fixed_ablation_params(model, 678, 2083, 4)
    assert params["n_estimators"] == 678 and params["random_state"] == 2083
    assert params["learning_rate"] == .03
    assert model.params["n_estimators"] == 1500 and model.params["random_state"] == 2082
