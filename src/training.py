"""Leakage-safe LightGBM training pipeline for the Week 5–6 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import regression_metrics, seasonal_naive


NON_FEATURE_COLUMNS = {"timestamp", "demand", "season"}


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_frame(data_path: Path, config: dict) -> tuple[pd.DataFrame, list[str]]:
    timestamp_col = config["timestamp_column"]
    target_col = config["target_column"]
    frame = pd.read_csv(data_path)
    missing = {timestamp_col, target_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")
    frame = frame.rename(columns={timestamp_col: "timestamp", target_col: "demand"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("Input contains duplicate timestamps; resolve them before training.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be chronological after sorting.")

    frame["hour"] = frame["timestamp"].dt.hour
    frame["minute"] = frame["timestamp"].dt.minute
    frame["month"] = frame["timestamp"].dt.month
    frame["day_of_year"] = frame["timestamp"].dt.dayofyear
    for lag in config["lag_intervals"]:
        frame[f"demand_lag_{lag}"] = frame["demand"].shift(int(lag))
    frame["naive_prediction"] = seasonal_naive(frame, int(config["lag_intervals"][0]))

    feature_names = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS or column == "naive_prediction":
            continue
        if pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = frame[column].astype("int8")
        elif not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        feature_names.append(column)
    frame = frame.dropna(subset=["demand"] + [f"demand_lag_{lag}" for lag in config["lag_intervals"]])
    return frame.reset_index(drop=True), feature_names


def chronological_split(frame: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    fractions = [config["train_fraction"], config["validation_fraction"], config["test_fraction"]]
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train/validation/test fractions must sum to 1.")
    train_end = int(len(frame) * fractions[0])
    validation_end = train_end + int(len(frame) * fractions[1])
    if train_end == 0 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError("Dataset is too small for the configured chronological split.")
    splits = {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:validation_end].copy(),
        "test": frame.iloc[validation_end:].copy(),
    }
    assert splits["train"]["timestamp"].max() < splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() < splits["test"]["timestamp"].min()
    timestamp_sets = [set(part["timestamp"]) for part in splits.values()]
    assert not (timestamp_sets[0] & timestamp_sets[1] or timestamp_sets[0] & timestamp_sets[2] or timestamp_sets[1] & timestamp_sets[2])
    return splits


def fit_lightgbm(splits: dict[str, pd.DataFrame], feature_names: list[str], config: dict):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required. Install requirements.txt first.") from exc
    common = {
        "objective": "regression_l1",
        "random_state": config["random_seed"],
        "n_jobs": -1,
        "verbosity": -1,
        "subsample_freq": 1,
    }
    x_train, y_train = splits["train"][feature_names], splits["train"]["demand"]
    x_valid, y_valid = splits["validation"][feature_names], splits["validation"]["demand"]
    results = []
    best = None
    for index, candidate in enumerate(config["parameter_candidates"]):
        model = lgb.LGBMRegressor(**common, **candidate)
        model.fit(
            x_train, y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False)],
        )
        prediction = model.predict(x_valid, num_iteration=model.best_iteration_)
        metrics = regression_metrics(y_valid, prediction, config["mape_epsilon"])
        result = {"candidate": index, "params": candidate, "best_iteration": int(model.best_iteration_), "validation": metrics}
        results.append(result)
        if best is None or metrics["mae"] < best[0]:
            best = (metrics["mae"], model, result)
    return best[1], best[2], results


def _split_metadata(splits: dict[str, pd.DataFrame]) -> dict:
    return {
        name: {
            "start": part["timestamp"].min().isoformat(),
            "end": part["timestamp"].max().isoformat(),
            "rows": int(len(part)),
        }
        for name, part in splits.items()
    }


def _save_plots(predictions: pd.DataFrame, output_dir: Path) -> None:
    test = predictions[(predictions["split"] == "test") & (predictions["model"] == "lightgbm")]
    sample = test.iloc[: min(len(test), 2016)]
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(sample["timestamp"], sample["actual"], label="Actual", linewidth=1)
    axis.plot(sample["timestamp"], sample["prediction"], label="LightGBM", linewidth=1)
    axis.set_title("Test actual vs predicted (first week)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.hist(test["actual"] - test["prediction"], bins=50)
    axis.set_title("Test residual distribution")
    axis.set_xlabel("Actual − prediction")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_distribution.png", dpi=150)
    plt.close(fig)


def run_training(data_path: Path, config_path: Path, output_dir: Path) -> dict:
    config = load_config(config_path)
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
            prediction_rows.append(pd.DataFrame({
                "timestamp": part["timestamp"].to_numpy(),
                "actual": part["demand"].to_numpy(),
                "prediction": values,
                "split": split_name,
                "model": model_name,
            }))
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if not np.isfinite(predictions[["actual", "prediction"]].to_numpy()).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    model.booster_.save_model(str(output_dir / "model.txt"), num_iteration=model.best_iteration_)
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    best_params = {**selected["params"], "best_iteration": selected["best_iteration"], "random_seed": config["random_seed"], "strategy": "train-fitted model selected and early-stopped on validation; test used once for final evaluation"}
    (output_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "split_metadata.json").write_text(json.dumps(_split_metadata(splits), indent=2), encoding="utf-8")
    (output_dir / "tuning_results.json").write_text(json.dumps(tuning_results, indent=2), encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    _save_plots(predictions, output_dir)
    return {"metrics": metrics, "splits": _split_metadata(splits), "features": feature_names}
