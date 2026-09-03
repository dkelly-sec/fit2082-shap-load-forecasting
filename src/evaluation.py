"""Evaluation metrics and the seasonal-naive electricity-demand baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual, prediction, mape_epsilon: float = 1.0) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if not valid.any():
        raise ValueError("No finite actual/prediction pairs are available for evaluation.")
    actual, prediction = actual[valid], prediction[valid]
    residual = actual - prediction
    denominator = np.maximum(np.abs(actual), mape_epsilon)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mape_percent": float(np.mean(np.abs(residual) / denominator) * 100.0),
        "n": int(len(actual)),
    }


def seasonal_naive(frame: pd.DataFrame) -> pd.Series:
    """Predict y_(t+24h) with y_t for rows whose timestamp is origin t."""
    if "demand_at_origin" not in frame:
        raise ValueError("seasonal_naive requires a demand_at_origin column.")
    return frame["demand_at_origin"].copy()
