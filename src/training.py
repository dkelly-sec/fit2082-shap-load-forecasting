"""LightGBM pipeline with explicit forecast-origin and target-time semantics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import regression_metrics, seasonal_naive


NON_FEATURE_COLUMNS = {
    "timestamp", "target_timestamp", "demand", "demand_at_origin",
    "naive_prediction", "season",
}


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_dataset(data_path: Path, config: dict) -> dict:
    """Return reproducible pre-training checks without inventing corrections."""
    raw = pd.read_csv(data_path)
    timestamp_col, target_col = config["timestamp_column"], config["target_column"]
    expected = [timestamp_col, target_col, *config.get("required_weather_columns", [])]
    missing_columns = sorted(set(expected) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")
    timestamps = pd.to_datetime(raw[timestamp_col], errors="raise")
    missing_counts = {column: int(raw[column].isna().sum()) for column in config.get("required_weather_columns", [])}
    known = config.get("known_missing_weather_rows", {})
    return {
        "rows": int(len(raw)),
        "required_columns": expected,
        "timestamps_sorted": bool(timestamps.is_monotonic_increasing),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "missing_weather_rows": missing_counts,
        "known_missing_weather_rows": known,
        "known_missing_counts_match": {key: missing_counts.get(key) == value for key, value in known.items()},
        "forecast_horizon_intervals": int(config["forecast_horizon_intervals"]),
        "maximum_origin_lag_intervals": int(max(config["lag_intervals"])),
        "expected_structural_boundary_loss": int(max(config["lag_intervals"]) + config["forecast_horizon_intervals"]),
        "weather_feature_timing": config["weather_feature_timing"],
    }


def prepare_frame(data_path: Path, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build one row per forecast origin t.

    ``timestamp`` is forecast origin t and ``target_timestamp`` is t+h.
    ``demand`` is target y_(t+h). ``demand_at_origin`` is y_t and is the
    24-hour seasonal-naive prediction when h=288 five-minute intervals.
    ``demand_lag_k`` is origin-relative y_(t-k), never target-relative.

    Weather defaults to observations available at the origin. Actual future
    weather is not accepted as leakage-safe forecast information.
    """
    timestamp_col = config["timestamp_column"]
    target_col = config["target_column"]
    frame = pd.read_csv(data_path)
    missing = {timestamp_col, target_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")
    frame = frame.rename(columns={timestamp_col: "timestamp", target_col: "demand_at_origin"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["demand_at_origin"] = pd.to_numeric(frame["demand_at_origin"], errors="coerce")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("Input contains duplicate timestamps; resolve them before training.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be chronological after sorting.")

    horizon = int(config["forecast_horizon_intervals"])
    interval_minutes = int(config.get("interval_minutes", 5))
    if horizon <= 0:
        raise ValueError("forecast_horizon_intervals must be positive.")
    if config.get("weather_feature_timing", "origin") != "origin":
        raise ValueError(
            "Only origin-available weather is supported. Actual future weather "
            "must be labelled and implemented separately as forecast/oracle weather."
        )

    frame["target_timestamp"] = frame["timestamp"].shift(-horizon)
    frame["demand"] = frame["demand_at_origin"].shift(-horizon)
    expected = frame["timestamp"] + pd.to_timedelta(horizon * interval_minutes, unit="min")
    aligned = frame["target_timestamp"].isna() | (frame["target_timestamp"] == expected)
    if not aligned.all():
        raise ValueError("Input timestamps are not a complete regular grid at the configured horizon.")

    frame["hour"] = frame["timestamp"].dt.hour
    frame["minute"] = frame["timestamp"].dt.minute
    frame["month"] = frame["timestamp"].dt.month
    frame["day_of_year"] = frame["timestamp"].dt.dayofyear
    for lag in config["lag_intervals"]:
        frame[f"demand_lag_{lag}"] = frame["demand_at_origin"].shift(int(lag))
    frame["naive_prediction"] = seasonal_naive(frame)

    required_weather = list(config.get("required_weather_columns", []))
    missing_weather_columns = sorted(set(required_weather) - set(frame.columns))
    if missing_weather_columns:
        raise ValueError(f"Input data is missing required weather columns: {missing_weather_columns}")

    feature_names = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype("int8")
        elif not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        feature_names.append(column)

    required = ["target_timestamp", "demand", "demand_at_origin"] + [
        f"demand_lag_{lag}" for lag in config["lag_intervals"]
    ]
    if config.get("missing_weather_strategy", "drop") == "drop":
        required += required_weather
    elif required_weather and frame[required_weather].isna().any().any():
        raise ValueError("Missing weather values require an explicit supported handling strategy.")
    frame = frame.dropna(subset=required)
    return frame.reset_index(drop=True), feature_names


def chronological_split(frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    """Split by origin time with a horizon-length purge at both boundaries."""
    fractions = [config["train_fraction"], config["validation_fraction"], config["test_fraction"]]
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train/validation/test fractions must sum to 1.")
    purge = int(config.get("purge_intervals", config["forecast_horizon_intervals"]))
    if purge < int(config["forecast_horizon_intervals"]):
        raise ValueError("purge_intervals must be at least the forecast horizon.")
    usable = len(frame) - 2 * purge
    train_end = int(usable * fractions[0])
    validation_start = train_end + purge
    validation_end = validation_start + int(usable * fractions[1])
    test_start = validation_end + purge
    if train_end == 0 or validation_end <= validation_start or test_start >= len(frame):
        raise ValueError("Dataset is too small for the configured chronological split and purge gaps.")
    splits = {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[validation_start:validation_end].copy(),
        "test": frame.iloc[test_start:].copy(),
    }
    assert splits["train"]["timestamp"].max() < splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() < splits["test"]["timestamp"].min()
    assert splits["train"]["target_timestamp"].max() < splits["validation"]["timestamp"].min()
    assert splits["validation"]["target_timestamp"].max() < splits["test"]["timestamp"].min()
    timestamp_sets = [set(part["timestamp"]) for part in splits.values()]
    assert not (timestamp_sets[0] & timestamp_sets[1] or timestamp_sets[0] & timestamp_sets[2] or timestamp_sets[1] & timestamp_sets[2])
    return splits


def fit_lightgbm(splits: dict[str, pd.DataFrame], feature_names: list[str], config: dict):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required. Install requirements.txt first.") from exc
    common = {"objective": "regression_l1", "random_state": config["random_seed"], "n_jobs": -1,
              "verbosity": -1, "subsample_freq": 1}
    x_train, y_train = splits["train"][feature_names], splits["train"]["demand"]
    x_valid, y_valid = splits["validation"][feature_names], splits["validation"]["demand"]
    results, best = [], None
    for index, candidate in enumerate(config["parameter_candidates"]):
        model = lgb.LGBMRegressor(**common, **candidate)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric="mae",
                  callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False)])
        prediction = model.predict(x_valid, num_iteration=model.best_iteration_)
        metrics = regression_metrics(y_valid, prediction, config["mape_epsilon"])
        result = {"candidate": index, "params": candidate, "best_iteration": int(model.best_iteration_), "validation": metrics}
        results.append(result)
        if best is None or metrics["mae"] < best[0]:
            best = (metrics["mae"], model, result)
    return best[1], best[2], results


def _split_metadata(splits: dict[str, pd.DataFrame]) -> dict:
    return {name: {"origin_start": part["timestamp"].min().isoformat(),
                   "origin_end": part["timestamp"].max().isoformat(),
                   "target_start": part["target_timestamp"].min().isoformat(),
                   "target_end": part["target_timestamp"].max().isoformat(),
                   "rows": int(len(part))} for name, part in splits.items()}


def _save_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[(predictions["split"] == "test") & (predictions["model"] == "lightgbm")]
    sample = test.iloc[: min(len(test), 2016)]
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(sample["target_timestamp"], sample["actual"], label="Actual", linewidth=1)
    axis.plot(sample["target_timestamp"], sample["prediction"], label="LightGBM", linewidth=1)
    axis.set_title("Test target demand vs prediction (first week)"); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4)); axis.hist(test["actual"] - test["prediction"], bins=50)
    axis.set_title("Test residual distribution"); axis.set_xlabel("Actual - prediction"); fig.tight_layout()
    fig.savefig(output_dir / "residual_distribution.png", dpi=150); plt.close(fig)


def run_training(data_path: Path, config_path: Path, output_dir: Path) -> dict:
    config = load_config(config_path)
    audit = audit_dataset(data_path, config)
    if not audit["timestamps_sorted"]:
        raise ValueError("Input timestamps must already be sorted chronologically.")
    if audit["duplicate_timestamps"]:
        raise ValueError("Input contains duplicate timestamps; resolve them before training.")
    frame, feature_names = prepare_frame(data_path, config)
    splits = chronological_split(frame, config)
    model, selected, tuning_results = fit_lightgbm(splits, feature_names, config)
    metrics, prediction_rows = {}, []
    for split_name in ("validation", "test"):
        part = splits[split_name]
        model_prediction = model.predict(part[feature_names], num_iteration=model.best_iteration_)
        baseline_prediction = part["naive_prediction"].to_numpy()
        metrics[split_name] = {
            "lightgbm": regression_metrics(part["demand"], model_prediction, config["mape_epsilon"]),
            "seasonal_naive": regression_metrics(part["demand"], baseline_prediction, config["mape_epsilon"]),
        }
        for model_name, values in (("lightgbm", model_prediction), ("seasonal_naive", baseline_prediction)):
            prediction_rows.append(pd.DataFrame({"timestamp": part["timestamp"].to_numpy(),
                                                 "target_timestamp": part["target_timestamp"].to_numpy(),
                                                 "actual": part["demand"].to_numpy(), "prediction": values,
                                                 "split": split_name, "model": model_name}))
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if not np.isfinite(predictions[["actual", "prediction"]].to_numpy()).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    output_dir.mkdir(parents=True, exist_ok=True)
    audit["rows_after_target_lag_and_weather_filtering"] = int(len(frame))
    (output_dir / "data_quality.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    joblib.dump(model, output_dir / "model.joblib")
    model.booster_.save_model(str(output_dir / "model.txt"), num_iteration=model.best_iteration_)
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    best_params = {**selected["params"], "best_iteration": selected["best_iteration"],
                   "random_seed": config["random_seed"], "forecast_horizon_intervals": config["forecast_horizon_intervals"],
                   "weather_feature_timing": config["weather_feature_timing"],
                   "strategy": "train-fitted model selected on validation; purged test used once"}
    (output_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "split_metadata.json").write_text(json.dumps(_split_metadata(splits), indent=2), encoding="utf-8")
    (output_dir / "tuning_results.json").write_text(json.dumps(tuning_results, indent=2), encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    _save_plots(predictions, output_dir)
    return {"metrics": metrics, "splits": _split_metadata(splits), "features": feature_names}
