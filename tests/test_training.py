"""Leakage, metric, feature-order and end-to-end training tests."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import regression_metrics  # noqa: E402
from training import chronological_split, prepare_frame, run_training  # noqa: E402


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


def _smoke_config(source: Path, target: Path) -> None:
    config = json.loads(source.read_text(encoding="utf-8"))
    config["parameter_candidates"] = [{
        "learning_rate": 0.1, "n_estimators": 80, "num_leaves": 15,
        "max_depth": 6, "min_child_samples": 20, "subsample": 1.0,
        "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0,
    }]
    config["early_stopping_rounds"] = 10
    target.write_text(json.dumps(config), encoding="utf-8")


def test_metrics_are_safe_near_zero():
    metrics = regression_metrics([0.0, 2.0], [1.0, 1.0], mape_epsilon=1.0)
    assert metrics["mae"] == 1.0
    assert np.isfinite(metrics["mape_percent"])


def test_split_is_strictly_chronological(tmp_path):
    data = tmp_path / "data.csv"
    _synthetic_csv(data)
    config = json.loads((ROOT / "configs" / "training.json").read_text(encoding="utf-8"))
    frame, _ = prepare_frame(data, config)
    splits = chronological_split(frame, config)
    assert splits["train"].timestamp.max() < splits["validation"].timestamp.min()
    assert splits["validation"].timestamp.max() < splits["test"].timestamp.min()


def test_smoke_pipeline_and_reload(tmp_path):
    data, config_path = tmp_path / "data.csv", tmp_path / "config.json"
    output, output_repeat = tmp_path / "artifacts", tmp_path / "artifacts-repeat"
    _synthetic_csv(data)
    _smoke_config(ROOT / "configs" / "training.json", config_path)
    result = run_training(data, config_path, output)
    run_training(data, config_path, output_repeat)
    model = joblib.load(output / "model.joblib")
    features = json.loads((output / "feature_names.json").read_text(encoding="utf-8"))
    frame, regenerated_features = prepare_frame(data, json.loads(config_path.read_text(encoding="utf-8")))
    assert features == regenerated_features
    assert len(model.predict(frame[features].tail(10))) == 10
    assert set(result["metrics"]) == {"validation", "test"}
    predictions = pd.read_csv(output / "predictions.csv")
    repeated = pd.read_csv(output_repeat / "predictions.csv")
    assert not predictions["prediction"].isna().any()
    np.testing.assert_allclose(predictions["prediction"], repeated["prediction"], rtol=0, atol=1e-10)
