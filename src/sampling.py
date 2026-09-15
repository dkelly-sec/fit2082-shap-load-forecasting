"""Controlled sample construction for the SHAP stability experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RareEventDefinition:
    """Thresholds fixed from training data before inspecting SHAP results."""

    demand_quantile: float
    temperature_low_quantile: float
    temperature_high_quantile: float
    demand_high: float
    temperature_low: float
    temperature_high: float

    def to_dict(self) -> dict:
        return asdict(self)


def define_rare_events(
    training_frame: pd.DataFrame,
    demand_quantile: float = 0.95,
    temperature_low_quantile: float = 0.05,
    temperature_high_quantile: float = 0.95,
) -> RareEventDefinition:
    """Pre-register rare-event thresholds using the training split only."""
    required = {"demand", "temperature"}
    missing = required - set(training_frame.columns)
    if missing:
        raise ValueError(f"Cannot define rare events; missing columns: {sorted(missing)}")
    if not (0 < temperature_low_quantile < temperature_high_quantile < 1):
        raise ValueError("Temperature quantiles must satisfy 0 < low < high < 1.")
    if not 0 < demand_quantile < 1:
        raise ValueError("demand_quantile must be between 0 and 1.")
    return RareEventDefinition(
        demand_quantile=demand_quantile,
        temperature_low_quantile=temperature_low_quantile,
        temperature_high_quantile=temperature_high_quantile,
        demand_high=float(training_frame["demand"].quantile(demand_quantile)),
        temperature_low=float(training_frame["temperature"].quantile(temperature_low_quantile)),
        temperature_high=float(training_frame["temperature"].quantile(temperature_high_quantile)),
    )


def rare_event_mask(frame: pd.DataFrame, definition: RareEventDefinition) -> pd.Series:
    """High demand or either temperature tail, using pre-registered cut-offs."""
    return (
        (frame["demand"] >= definition.demand_high)
        | (frame["temperature"] <= definition.temperature_low)
        | (frame["temperature"] >= definition.temperature_high)
    )


def _validate(frame: pd.DataFrame, feature_names: list[str], size: int) -> None:
    if size <= 0:
        raise ValueError("Sample size must be positive.")
    if size > len(frame):
        raise ValueError(f"Requested sample size {size} exceeds available rows ({len(frame)}).")
    missing = set(feature_names) - set(frame.columns)
    if missing:
        raise ValueError(f"Sampling frame is missing model features: {sorted(missing)}")


def _time_stratified_indices(frame: pd.DataFrame, size: int, seed: int) -> np.ndarray:
    if "timestamp" not in frame:
        raise ValueError("Time-stratified sampling requires timestamp.")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    # Month captures season; four six-hour blocks capture intraday demand regimes.
    strata = timestamps.dt.month.astype(str) + "_" + (timestamps.dt.hour // 6).astype(str)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    groups = {key: np.asarray(values, dtype=int) for key, values in strata.groupby(strata).groups.items()}
    exact = {key: size * len(values) / len(frame) for key, values in groups.items()}
    allocations = {key: min(len(groups[key]), int(np.floor(value))) for key, value in exact.items()}
    remainder = size - sum(allocations.values())
    order = sorted(groups, key=lambda key: exact[key] - allocations[key], reverse=True)
    for key in order:
        if remainder == 0:
            break
        if allocations[key] < len(groups[key]):
            allocations[key] += 1
            remainder -= 1
    for key, values in groups.items():
        selected.extend(rng.choice(values, size=allocations[key], replace=False).tolist())
    if len(selected) < size:
        remaining = np.setdiff1d(np.arange(len(frame)), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, size=size - len(selected), replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected[:size])


def _kmeans_representatives(
    frame: pd.DataFrame, feature_names: list[str], size: int, seed: int
) -> pd.DataFrame:
    values = frame[feature_names].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(values)
    model = MiniBatchKMeans(
        n_clusters=size,
        random_state=seed,
        batch_size=min(4096, len(frame)),
        n_init=3,
    ).fit(scaled)
    # Return real, distinct rows nearest the centroids rather than synthetic rows.
    distances = model.transform(scaled)
    selected: list[int] = []
    used: set[int] = set()
    for cluster in range(size):
        for index in np.argsort(distances[:, cluster]):
            candidate = int(index)
            if candidate not in used:
                used.add(candidate)
                selected.append(candidate)
                break
    return frame.iloc[selected][feature_names].reset_index(drop=True)


def construct_sample(
    frame: pd.DataFrame,
    feature_names: list[str],
    size: int,
    seed: int,
    method: str = "uniform",
    rare_definition: RareEventDefinition | None = None,
    rare_fraction: float = 0.5,
) -> pd.DataFrame:
    """Construct an exact-size feature sample using a registered method."""
    _validate(frame, feature_names, size)
    # Chronological splits retain source row labels; all sampling internals use
    # local positional indices so their behaviour cannot depend on those labels.
    frame = frame.reset_index(drop=True)
    if method == "uniform":
        return frame.sample(n=size, random_state=seed)[feature_names].reset_index(drop=True)
    if method == "time_stratified":
        return frame.iloc[_time_stratified_indices(frame, size, seed)][feature_names].reset_index(drop=True)
    if method == "kmeans":
        return _kmeans_representatives(frame, feature_names, size, seed)
    if method == "rare_event_stratified":
        if rare_definition is None:
            raise ValueError("rare_event_stratified sampling requires a fixed rare-event definition.")
        if not 0 < rare_fraction < 1:
            raise ValueError("rare_fraction must be between 0 and 1.")
        mask = rare_event_mask(frame, rare_definition)
        rare, ordinary = frame[mask], frame[~mask]
        rare_n = min(len(rare), round(size * rare_fraction))
        ordinary_n = size - rare_n
        if ordinary_n > len(ordinary):
            ordinary_n = len(ordinary)
            rare_n = size - ordinary_n
        if rare_n > len(rare):
            raise ValueError("Not enough rare-event rows for the requested stratified sample.")
        result = pd.concat([
            rare.sample(n=rare_n, random_state=seed),
            ordinary.sample(n=ordinary_n, random_state=seed + 1),
        ]).sample(frac=1, random_state=seed + 2)
        return result[feature_names].reset_index(drop=True)
    raise ValueError(f"Unknown sampling method: {method}")
