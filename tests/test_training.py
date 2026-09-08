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
from training import audit_dataset, chronological_split, prepare_frame, run_training  # noqa: E402


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
    assert splits["train"].target_timestamp.max() < splits["validation"].timestamp.min()
    assert splits["validation"].target_timestamp.max() < splits["test"].timestamp.min()


def test_24_hour_target_lags_and_baseline_have_explicit_origin_semantics(tmp_path):
    data = tmp_path / "data.csv"
    periods = 5000
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="5min")
    pd.DataFrame({
        "timestamp": timestamps,
        "demand": np.arange(periods, dtype=float),
        "temperature": 20.0,
        "irr_electricity": 0.0,
    }).to_csv(data, index=False)
    config = json.loads((ROOT / "configs" / "training.json").read_text(encoding="utf-8"))
    frame, _ = prepare_frame(data, config)
    row = frame.iloc[0]
    origin_index = 2016
    assert row.timestamp == timestamps[origin_index]
    assert row.target_timestamp == timestamps[origin_index + 288]
    assert row.demand == origin_index + 288
    assert row.demand_at_origin == origin_index
    assert row.naive_prediction == origin_index
    assert row.demand_lag_288 == origin_index - 288
    assert row.demand_lag_2016 == origin_index - 2016
    assert len(frame) == periods - 2304


def test_missing_required_weather_is_explicit(tmp_path):
    data = tmp_path / "data.csv"
    _synthetic_csv(data)
    raw = pd.read_csv(data).drop(columns=["irr_electricity"])
    raw.to_csv(data, index=False)
    config = json.loads((ROOT / "configs" / "training.json").read_text(encoding="utf-8"))
    try:
        prepare_frame(data, config)
    except ValueError as exc:
        assert "required weather columns" in str(exc)
    else:
        raise AssertionError("Missing weather columns must fail explicitly")


def test_dataset_audit_records_boundary_and_missing_counts(tmp_path):
    data = tmp_path / "data.csv"
    _synthetic_csv(data)
    raw = pd.read_csv(data)
    raw.loc[:4, "temperature"] = np.nan
    raw.loc[:2, "irr_electricity"] = np.nan
    raw.to_csv(data, index=False)
    config = json.loads((ROOT / "configs" / "training.json").read_text(encoding="utf-8"))
    audit = audit_dataset(data, config)
    assert audit["timestamps_sorted"]
    assert audit["duplicate_timestamps"] == 0
    assert audit["missing_weather_rows"] == {"temperature": 5, "irr_electricity": 3}
    assert audit["expected_structural_boundary_loss"] == 2304


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
