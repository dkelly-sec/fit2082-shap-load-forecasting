"""
Tests for load_model.py, using the real training pipeline (not a mock) to
generate a genuine artifacts directory, then confirming the loader reads
it back correctly.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend -- training.py's plotting code
                        # otherwise tries to open a GUI window via Tkinter,
                        # which fails on some Windows Python installs (e.g.
                        # the Microsoft Store build) with a broken Tcl/Tk
                        # component. Agg never needs a display, so this
                        # sidesteps that entirely and is also just correct
                        # practice for a test that shouldn't pop up windows.

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import run_training  # noqa: E402
from load_model import load_frozen_model, sanity_check  # noqa: E402


def _synthetic_csv(path: Path, periods: int = 5000) -> None:
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
def trained_artifacts(tmp_path):
    data = tmp_path / "data.csv"
    config_path = tmp_path / "config.json"
    output = tmp_path / "artifacts"
    _synthetic_csv(data)
    _fast_config(ROOT / "configs" / "training.json", config_path)
    run_training(data, config_path, output)
    return output


def test_load_frozen_model_reads_real_artifacts(trained_artifacts):
    bundle = load_frozen_model(trained_artifacts)
    assert hasattr(bundle.model, "predict")
    assert len(bundle.feature_names) > 0
    assert "demand_lag_288" in bundle.feature_names
    assert "validation" in bundle.metrics
    assert "test" in bundle.metrics


def test_loaded_model_predicts_matching_feature_order(trained_artifacts):
    bundle = load_frozen_model(trained_artifacts)
    dummy = pd.DataFrame([[1.0] * len(bundle.feature_names)], columns=bundle.feature_names)
    prediction = bundle.model.predict(dummy)
    assert len(prediction) == 1
    assert np.isfinite(prediction[0])


def test_sanity_check_passes_on_real_model(trained_artifacts, capsys):
    bundle = load_frozen_model(trained_artifacts)
    sanity_check(bundle)
    captured = capsys.readouterr()
    assert "Sanity check passed" in captured.out


def test_missing_artifacts_directory_raises_clear_error(tmp_path):
    empty_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Missing required artifact"):
        load_frozen_model(empty_dir)


def test_wrong_feature_order_is_caught_by_predict_shape_mismatch(trained_artifacts):
    bundle = load_frozen_model(trained_artifacts)
    # deliberately drop a column -- predicting with the wrong shape should
    # fail loudly, not silently produce a nonsense number
    wrong_shape = pd.DataFrame([[1.0] * (len(bundle.feature_names) - 1)],
                                columns=bundle.feature_names[:-1])
    with pytest.raises(Exception):
        bundle.model.predict(wrong_shape)


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "-v", __file__])
