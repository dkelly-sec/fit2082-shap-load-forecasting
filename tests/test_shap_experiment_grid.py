"""
Tests for the Week 9 grid extensions in shap_experiment.py: the new
metrics (Jaccard, Kendall's W), the within- vs between-condition split,
resume behaviour, and the guards that stop a multi-hour run from failing
late or mixing results from different models.
"""
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shap_experiment import (  # noqa: E402
    build_grid,
    default_reference,
    kendalls_w,
    run_experiment,
    top_k_jaccard,
    within_condition_analysis,
)
from training import run_training  # noqa: E402


# --------------------------------------------------------------------------
# Metric correctness
# --------------------------------------------------------------------------

def test_top_k_jaccard_is_intersection_over_union():
    left = pd.Series([4.0, 3.0, 2.0, 1.0], index=["a", "b", "c", "d"])
    right = pd.Series([4.0, 1.0, 3.0, 2.0], index=["a", "b", "c", "d"])
    # top-2: {a, b} vs {a, c} -> 1 shared / 3 total
    assert top_k_jaccard(left, right, 2) == pytest.approx(1 / 3)
    assert top_k_jaccard(left, left, 2) == pytest.approx(1.0)


def test_kendalls_w_is_one_for_identical_rankings():
    row = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert kendalls_w(pd.DataFrame([row, row, row])) == pytest.approx(1.0)


def test_kendalls_w_is_zero_for_two_reversed_rankings():
    assert kendalls_w(pd.DataFrame([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])) == pytest.approx(0.0)


def test_kendalls_w_matches_mean_spearman_identity():
    """
    For untied rankings, mean pairwise Spearman r and W satisfy
    r_mean = (m * W - 1) / (m - 1). Checking this identity against an
    independent Spearman calculation verifies the W implementation.
    """
    rng = np.random.default_rng(0)
    m, n = 6, 12
    matrix = pd.DataFrame([rng.permutation(n).astype(float) for _ in range(m)])
    spearmans = [
        pd.Series(matrix.iloc[i]).corr(pd.Series(matrix.iloc[j]), method="spearman")
        for i, j in itertools.combinations(range(m), 2)
    ]
    w = kendalls_w(matrix)
    assert np.mean(spearmans) == pytest.approx((m * w - 1) / (m - 1), abs=1e-9)


def test_kendalls_w_handles_tied_zero_importance_features():
    """Zero-importance features tie at exactly 0; identical rankings must still give W = 1."""
    row = [5.0, 3.0, 0.0, 0.0, 0.0]
    assert kendalls_w(pd.DataFrame([row, row])) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Within- vs between-condition separation
# --------------------------------------------------------------------------

def test_within_condition_pairs_never_cross_configurations():
    """
    The original pilot compared every run with every other run, mixing
    'repeat the same design' with 'change the design'. Within-condition
    pairs must only ever compare seeds of the same configuration.
    """
    seeds = [1, 2, 3, 4]
    grid = build_grid(["uniform", "kmeans"], [50], ["uniform", "season_stratified"], [100], seeds)
    features = ["a", "b", "c", "d"]
    rng = np.random.default_rng(1)
    rankings = {spec["run_id"]: pd.Series(rng.random(4), index=features) for spec in grid}
    runs = pd.DataFrame([{**spec, "elapsed_seconds": 1.0, "additivity_status": "pass"} for spec in grid])

    pairwise, within = within_condition_analysis(rankings, runs)

    n_configs = 4
    assert len(pairwise) == n_configs * len(list(itertools.combinations(seeds, 2)))
    assert len(within) == n_configs
    assert (within["n_seeds"] == len(seeds)).all()
    # every pair's seeds must belong to the same configuration
    for _, pair in pairwise.iterrows():
        assert pair["left_seed"] != pair["right_seed"]
    assert pairwise.groupby("config_id").size().eq(6).all()


def test_default_reference_is_largest_uniform_configuration():
    reference = default_reference(["kmeans", "uniform"], [50, 200],
                                  ["season_stratified", "uniform"], [100, 500])
    assert reference == "bg-uniform_n-200__eval-uniform_n-500"


# --------------------------------------------------------------------------
# End-to-end against a genuinely trained model
# --------------------------------------------------------------------------

def _synthetic_csv(path: Path, periods: int = 8000) -> None:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="5min")
    phase = np.arange(periods) * 2 * np.pi / 288
    season_map = {12: "summer", 1: "summer", 2: "summer", 3: "autumn", 4: "autumn", 5: "autumn",
                  6: "winter", 7: "winter", 8: "winter", 9: "spring", 10: "spring", 11: "spring"}
    # fake seasons across the short synthetic window so all four strata exist
    fake_month = (np.arange(periods) // (periods // 12)) % 12 + 1
    pd.DataFrame({
        "timestamp": timestamps,
        "demand": 5000 + 700 * np.sin(phase) + 150 * np.sin(phase / 7) + np.arange(periods) * 0.02,
        "temperature": 18 + 10 * np.sin(phase - 1),
        "irr_electricity": np.maximum(0, np.sin(phase - 1.5)),
        "day_of_week": timestamps.dayofweek,
        "is_weekend": timestamps.dayofweek >= 5,
        "is_public_holiday": (np.arange(periods) // 288) % 9 == 0,
        "season": [season_map[m] for m in fake_month],
    }).to_csv(path, index=False)


def _fast_config(target: Path) -> None:
    config = json.loads((ROOT / "configs" / "training.json").read_text(encoding="utf-8"))
    config["parameter_candidates"] = [{
        "learning_rate": 0.1, "n_estimators": 30, "num_leaves": 15, "max_depth": 6,
        "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
        "reg_alpha": 0.0, "reg_lambda": 0.0,
    }]
    config["early_stopping_rounds"] = 10
    target.write_text(json.dumps(config), encoding="utf-8")


@pytest.fixture
def trained(tmp_path):
    data, config, artifacts = tmp_path / "data.csv", tmp_path / "config.json", tmp_path / "artifacts"
    _synthetic_csv(data)
    _fast_config(config)
    run_training(data, config, artifacts)
    return {"data": data, "config": config, "artifacts": artifacts}


GRID = dict(
    background_methods=["uniform", "kmeans"],
    background_sizes=[30],
    evaluation_methods=["uniform", "season_stratified", "rare_event_stratified"],
    evaluation_sizes=[40],
)


def _run(trained, output, seeds):
    return run_experiment(trained["data"], trained["config"], trained["artifacts"], output,
                          seeds=seeds, **GRID)


def test_full_grid_writes_separated_outputs(trained, tmp_path):
    output = tmp_path / "grid"
    summary = _run(trained, output, [1, 2, 3])

    n_configs = 2 * 1 * 3 * 1
    assert summary["run_count"] == n_configs * 3
    assert summary["configuration_count"] == n_configs

    within = pd.read_csv(output / "within_condition_stability.csv")
    pairwise = pd.read_csv(output / "pairwise_within_condition.csv")
    between = pd.read_csv(output / "between_condition_agreement.csv")
    assert len(within) == n_configs
    assert len(pairwise) == n_configs * 3  # C(3,2) = 3 pairs per configuration
    assert within["kendalls_w"].between(0, 1).all()
    assert len(between) == n_configs
    assert between["is_reference"].sum() == 1
    reference_row = between[between["is_reference"]].iloc[0]
    assert reference_row["spearman_vs_reference"] == pytest.approx(1.0)
    assert not (output / "pairwise_stability.csv").exists()  # old mixed metric is gone


def test_resume_skips_completed_runs(trained, tmp_path):
    output = tmp_path / "grid"
    first = _run(trained, output, [1, 2])
    assert first["runs_computed_this_invocation"] == first["run_count"]

    second = _run(trained, output, [1, 2])
    assert second["runs_computed_this_invocation"] == 0

    # simulate a crash part-way through a run: its metadata never got written
    victim = next((output / "runs").glob("*.json"))
    victim.unlink()
    third = _run(trained, output, [1, 2])
    assert third["runs_computed_this_invocation"] == 1


def test_adding_seeds_only_computes_the_new_ones(trained, tmp_path):
    output = tmp_path / "grid"
    _run(trained, output, [1, 2])
    extended = _run(trained, output, [1, 2, 3])
    n_configs = 6
    assert extended["runs_computed_this_invocation"] == n_configs
    assert extended["run_count"] == n_configs * 3


def test_refuses_to_resume_into_a_different_models_directory(trained, tmp_path):
    output = tmp_path / "grid"
    _run(trained, output, [1])
    manifest_path = output / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_names"] = manifest["feature_names"] + ["not_a_real_feature"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot resume"):
        _run(trained, output, [1])


def test_oversized_evaluation_fails_before_any_shap_runs(trained, tmp_path):
    output = tmp_path / "grid"
    with pytest.raises(ValueError, match="exceeds test rows"):
        run_experiment(trained["data"], trained["config"], trained["artifacts"], output,
                       background_methods=["uniform"], background_sizes=[30],
                       evaluation_methods=["uniform"], evaluation_sizes=[10_000_000],
                       seeds=[1])
    assert not (output / "runs").exists() or not any((output / "runs").iterdir())
