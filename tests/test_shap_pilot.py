"""
Tests for shap_pilot.py: the additivity check, feature-mismatch guard, and
a full end-to-end run against a genuinely (fast-)trained model.
"""
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

from training import run_training  # noqa: E402
from load_model import load_frozen_model  # noqa: E402
from shap_pilot import (  # noqa: E402
    draw_sample, compute_global_shap_ranking, check_additivity, run_pilot,
)


def _synthetic_csv(path: Path, periods: int = 6000) -> None:
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="5min")
    phase = np.arange(periods) * 2 * np.pi / 288
    demand = 5000 + 700 * np.sin(phase) + 150 * np.sin(phase / 7)
    pd.DataFrame({
        "timestamp": timestamps,
        "demand": demand,
        "temperature": 18 + 8 * np.sin(phase - 1),
        "day_of_week": timestamps.dayofweek,
        "is_weekend": timestamps.dayofweek >= 5,
        "is_public_holiday": False,
        "season": "summer",
        "irr_electricity": np.maximum(0, np.sin(phase - 1.5)),
    }).to_csv(path, index=False)


def _fast_config(source: Path, target: Path) -> None:
    config = json.loads(source.read_text(encoding="utf-8"))
    config["parameter_candidates"] = [{
        "learning_rate": 0.1, "n_estimators": 80, "num_leaves": 15,
        "max_depth": 6, "min_child_samples": 20, "subsample": 1.0,
        "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0,
    }]
    config["early_stopping_rounds"] = 10
    target.write_text(json.dumps(config), encoding="utf-8")


@pytest.fixture
def trained_setup(tmp_path):
    data = tmp_path / "data.csv"
    config_path = tmp_path / "config.json"
    artifacts = tmp_path / "artifacts"
    _synthetic_csv(data)
    _fast_config(ROOT / "configs" / "training.json", config_path)
    run_training(data, config_path, artifacts)
    return {"data": data, "config": config_path, "artifacts": artifacts}


def test_additivity_holds_within_documented_tolerance(trained_setup):
    """
    The core correctness property: prediction == base_value + sum(shap
    values), within the tolerance documented for LightGBM's regression_l1
    leaf-approximation gap. This is the check that would catch a genuine
    TreeExplainer wiring bug (wrong background, wrong perturbation mode,
    mismatched feature order).
    """
    bundle = load_frozen_model(trained_setup["artifacts"])
    from training import load_config, prepare_frame
    config = load_config(trained_setup["config"])
    frame, _ = prepare_frame(trained_setup["data"], config)

    background = draw_sample(frame, bundle.feature_names, 100, seed=1)
    evaluation = draw_sample(frame, bundle.feature_names, 50, seed=2)

    ranking, explainer, shap_values = compute_global_shap_ranking(bundle, background, evaluation)
    check_additivity(bundle, explainer, shap_values, evaluation)  # raises on failure


def test_additivity_check_actually_detects_a_broken_explainer(trained_setup):
    """
    Confirm the additivity check isn't a rubber stamp: deliberately corrupt
    the shap_values (as a wrong-wiring scenario would) and verify it raises.
    """
    bundle = load_frozen_model(trained_setup["artifacts"])
    from training import load_config, prepare_frame
    config = load_config(trained_setup["config"])
    frame, _ = prepare_frame(trained_setup["data"], config)

    background = draw_sample(frame, bundle.feature_names, 100, seed=1)
    evaluation = draw_sample(frame, bundle.feature_names, 50, seed=2)
    _, explainer, shap_values = compute_global_shap_ranking(bundle, background, evaluation)

    broken_shap_values = shap_values + 500.0  # inject a large, obviously-wrong offset
    with pytest.raises(AssertionError, match="additivity check failed"):
        check_additivity(bundle, explainer, broken_shap_values, evaluation)


def test_ranking_gives_zero_importance_to_constant_feature(trained_setup):
    """
    is_public_holiday is constant (always False) in the synthetic data --
    a zero-variance feature must get exactly zero SHAP importance. If it
    doesn't, something is leaking or mis-attributing importance.
    """
    bundle = load_frozen_model(trained_setup["artifacts"])
    from training import load_config, prepare_frame
    config = load_config(trained_setup["config"])
    frame, _ = prepare_frame(trained_setup["data"], config)

    background = draw_sample(frame, bundle.feature_names, 100, seed=1)
    evaluation = draw_sample(frame, bundle.feature_names, 50, seed=2)
    ranking, _, _ = compute_global_shap_ranking(bundle, background, evaluation)

    assert "is_public_holiday" in ranking.index
    assert ranking["is_public_holiday"] == 0.0


def test_mismatched_features_raise_before_computing_anything(trained_setup, tmp_path):
    """
    If prepare_frame() ever produces a different feature set than the
    frozen model expects (e.g. someone changes config.py without retraining),
    run_pilot must fail loudly before wasting time computing SHAP values on
    a mismatched feature set.
    """
    bundle_dir = trained_setup["artifacts"]
    feature_names_path = bundle_dir / "feature_names.json"
    original = json.loads(feature_names_path.read_text(encoding="utf-8"))
    corrupted = original + ["a_feature_that_does_not_exist"]
    feature_names_path.write_text(json.dumps(corrupted), encoding="utf-8")

    with pytest.raises(ValueError, match="do not match the frozen model"):
        run_pilot(
            data_path=trained_setup["data"],
            config_path=trained_setup["config"],
            artifacts_dir=bundle_dir,
            background_size=50,
            evaluation_size=20,
            seed=1,
            output_dir=tmp_path / "shap_pilot_out",
        )


def test_full_pilot_runs_end_to_end_and_writes_outputs(trained_setup, tmp_path):
    output_dir = tmp_path / "shap_pilot_out"
    ranking = run_pilot(
        data_path=trained_setup["data"],
        config_path=trained_setup["config"],
        artifacts_dir=trained_setup["artifacts"],
        background_size=100,
        evaluation_size=50,
        seed=2082,
        output_dir=output_dir,
    )
    assert (output_dir / "pilot_global_ranking.csv").exists()
    assert (output_dir / "pilot_global_ranking.png").exists()
    assert (output_dir / "rare_event_definition.json").exists()
    assert (output_dir / "outcome_demand_definition.json").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert len(ranking) > 0
    assert (ranking >= 0).all()  # mean absolute SHAP values are never negative


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "-v", __file__])
