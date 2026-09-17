"""
Evaluation sample construction for the Weeks 8-9 SHAP sampling experiment
(RQ1).

The evaluation sample determines WHICH predictions get explained and
aggregated into a global SHAP feature-importance ranking. Varying how it
is constructed -- alongside the background sample -- is the central
experimental manipulation of this project.

Three construction methods are provided:

  "random"
      Plain uniform random sampling. The neutral baseline every other
      method is compared against.

  "season_stratified"
      Proportional allocation across summer/autumn/winter/spring, so each
      season is represented in proportion to its presence in the data
      rather than left to chance. Tests whether guaranteeing seasonal
      coverage stabilises the ranking relative to plain random draws.

  "rare_event_stratified"
      Deliberately OVER-samples infrequent but high-impact periods --
      public holidays and extreme-temperature days -- relative to their
      natural frequency. This directly targets the limitation Van Zyl et
      al. (2024) identified: that SHAP can underweight features which
      matter enormously in rare conditions but appear unimportant when
      averaged over samples dominated by ordinary periods.

      Note the over-sampling is deliberate and is the point. Proportional
      representation of rare events would put only ~10-15% rare rows in
      the sample, barely distinguishable from a random draw; the
      `rare_event_fraction` parameter (default 0.5) makes the
      concentration explicit and tunable so both can be tested.

Extreme temperature is defined by PERCENTILE, not fixed thresholds:
  - extreme hot  = daily temp_max in the top    `extreme_percentile`
  - extreme cold = daily temp_min in the bottom `extreme_percentile`

Percentiles are used because they guarantee a consistent stratum size
regardless of how hot or mild the study period happened to be (a fixed
">30C" threshold would catch wildly different row counts year to year,
injecting noise into exactly the ranking-stability measurement this
project exists to make), and because they self-calibrate to the study
location rather than hard-coding a Melbourne-specific number.

Interface contract (agreed with the background-sample side):
    draw_evaluation_sample(frame, feature_names, size, method, seed) -> DataFrame
Returns exactly `size` rows, restricted to `feature_names` in that exact
order, so the result can be handed straight to the SHAP explainer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVALUATION_METHODS = ("random", "season_stratified", "rare_event_stratified")

# Default: top/bottom 5% of daily temperature counts as "extreme".
DEFAULT_EXTREME_PERCENTILE = 0.05

# Default: half the rare-event-stratified sample is drawn from rare rows.
DEFAULT_RARE_EVENT_FRACTION = 0.5


def _validate(frame: pd.DataFrame, feature_names: list[str], size: int, method: str) -> None:
    if method not in EVALUATION_METHODS:
        raise ValueError(f"Unknown evaluation method {method!r}. Expected one of {EVALUATION_METHODS}.")
    if size <= 0:
        raise ValueError(f"Sample size must be positive, got {size}.")
    if size > len(frame):
        raise ValueError(f"Requested sample size {size} exceeds available rows ({len(frame)}).")
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing required feature columns: {missing}")


def _finalise(sampled: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Restrict to the model's exact feature columns, in the exact order."""
    return sampled[feature_names].reset_index(drop=True)


def identify_rare_event_rows(
    frame: pd.DataFrame,
    extreme_percentile: float = DEFAULT_EXTREME_PERCENTILE,
) -> pd.Series:
    """
    Return a boolean mask marking rows belonging to a "rare event" period.

    A row is rare if ANY of the following hold:
      - it falls on a public holiday (`is_public_holiday`)
      - its day's maximum temperature is in the top `extreme_percentile`
      - its day's minimum temperature is in the bottom `extreme_percentile`

    Requires `is_public_holiday`, `temp_max`, and `temp_min` columns to be
    present on `frame` (they are non-feature context columns, so this must
    be called on the frame BEFORE restricting to model features).
    """
    required = {"is_public_holiday", "temp_max", "temp_min"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Cannot identify rare events: frame is missing {sorted(missing)}. "
            "Call this on the full prepared frame, not one already restricted "
            "to model feature columns."
        )
    if not 0 < extreme_percentile < 0.5:
        raise ValueError(
            f"extreme_percentile must be strictly between 0 and 0.5, got {extreme_percentile}."
        )

    hot_threshold = frame["temp_max"].quantile(1 - extreme_percentile)
    cold_threshold = frame["temp_min"].quantile(extreme_percentile)

    is_holiday = frame["is_public_holiday"].astype(bool)
    is_extreme_hot = frame["temp_max"] >= hot_threshold
    is_extreme_cold = frame["temp_min"] <= cold_threshold

    return is_holiday | is_extreme_hot | is_extreme_cold


def _proportional_allocation(group_sizes: pd.Series, total: int, rng: np.random.Generator) -> dict:
    """
    Allocate `total` samples across groups in proportion to group size,
    distributing the rounding remainder randomly rather than always to the
    same (e.g. largest or alphabetically-first) group, so allocation is
    not systematically biased across repeated seeds.
    """
    proportions = group_sizes / group_sizes.sum()
    exact = proportions * total
    allocation = np.floor(exact).astype(int)
    remainder = total - allocation.sum()

    if remainder > 0:
        # distribute leftover slots weighted by each group's fractional part
        fractional = (exact - allocation).to_numpy(dtype=float)
        if fractional.sum() > 0:
            weights = fractional / fractional.sum()
        else:
            weights = np.full(len(fractional), 1.0 / len(fractional))
        extra_indices = rng.choice(len(allocation), size=remainder, replace=True, p=weights)
        for idx in extra_indices:
            allocation.iloc[idx] += 1

    # never allocate more than a group actually has
    overflow = 0
    for name in allocation.index:
        if allocation[name] > group_sizes[name]:
            overflow += allocation[name] - group_sizes[name]
            allocation[name] = group_sizes[name]

    # redistribute any overflow to groups with spare capacity
    while overflow > 0:
        spare = {n: group_sizes[n] - allocation[n] for n in allocation.index if group_sizes[n] > allocation[n]}
        if not spare:
            raise ValueError("Cannot allocate the requested sample size across the available groups.")
        names = list(spare.keys())
        chosen = names[int(rng.integers(len(names)))]
        allocation[chosen] += 1
        overflow -= 1

    return allocation.to_dict()


def _sample_random(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    return frame.sample(n=size, random_state=seed)


def _sample_season_stratified(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if "season" not in frame.columns:
        raise ValueError(
            "season_stratified requires a 'season' column. Call this on the full "
            "prepared frame, not one already restricted to model feature columns."
        )
    rng = np.random.default_rng(seed)
    group_sizes = frame.groupby("season").size()
    allocation = _proportional_allocation(group_sizes, size, rng)

    parts = []
    for season_name, n in allocation.items():
        if n == 0:
            continue
        group = frame[frame["season"] == season_name]
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed)  # shuffle so order isn't grouped


def _sample_rare_event_stratified(
    frame: pd.DataFrame,
    size: int,
    seed: int,
    rare_event_fraction: float,
    extreme_percentile: float,
) -> pd.DataFrame:
    if not 0 <= rare_event_fraction <= 1:
        raise ValueError(f"rare_event_fraction must be between 0 and 1, got {rare_event_fraction}.")

    rare_mask = identify_rare_event_rows(frame, extreme_percentile=extreme_percentile)
    rare_rows = frame[rare_mask]
    ordinary_rows = frame[~rare_mask]

    if len(rare_rows) == 0:
        raise ValueError(
            "No rare-event rows found in the supplied frame -- cannot build a "
            "rare-event-stratified sample. Check that is_public_holiday/temp_max/"
            "temp_min contain the expected values."
        )

    n_rare = int(round(size * rare_event_fraction))
    n_ordinary = size - n_rare

    # If either stratum is too small to supply its quota, take everything it
    # has and make up the difference from the other -- and say so, rather
    # than silently returning a sample with a different composition than
    # the caller asked for.
    if n_rare > len(rare_rows):
        print(f"  note: requested {n_rare} rare-event rows but only {len(rare_rows)} exist; "
              f"taking all of them and drawing the remaining {size - len(rare_rows)} from ordinary rows.")
        n_rare = len(rare_rows)
        n_ordinary = size - n_rare
    if n_ordinary > len(ordinary_rows):
        print(f"  note: requested {n_ordinary} ordinary rows but only {len(ordinary_rows)} exist; "
              f"taking all of them and drawing the remaining {size - len(ordinary_rows)} from rare-event rows.")
        n_ordinary = len(ordinary_rows)
        n_rare = size - n_ordinary

    parts = []
    if n_rare > 0:
        parts.append(rare_rows.sample(n=n_rare, random_state=seed))
    if n_ordinary > 0:
        parts.append(ordinary_rows.sample(n=n_ordinary, random_state=seed))

    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def draw_evaluation_sample(
    frame: pd.DataFrame,
    feature_names: list[str],
    size: int,
    method: str,
    seed: int,
    rare_event_fraction: float = DEFAULT_RARE_EVENT_FRACTION,
    extreme_percentile: float = DEFAULT_EXTREME_PERCENTILE,
) -> pd.DataFrame:
    """
    Draw an evaluation sample of exactly `size` rows using `method`.

    Parameters
    ----------
    frame : the FULL prepared frame (must still contain context columns
        `season`, `is_public_holiday`, `temp_max`, `temp_min` -- these are
        used for stratification, then dropped from the returned sample).
    feature_names : the model's exact ordered feature columns.
    size : number of rows to return.
    method : one of EVALUATION_METHODS.
    seed : random seed, so repeated draws differ but stay reproducible.
    rare_event_fraction : for "rare_event_stratified" only -- what share of
        the sample should come from rare-event rows (default 0.5, i.e.
        deliberate over-sampling relative to their ~10-15% natural rate).
    extreme_percentile : for "rare_event_stratified" only -- the tail
        fraction of daily temperatures treated as extreme (default 0.05).

    Returns
    -------
    DataFrame with exactly `size` rows and exactly `feature_names` columns,
    in that order.
    """
    _validate(frame, feature_names, size, method)

    if method == "random":
        sampled = _sample_random(frame, size, seed)
    elif method == "season_stratified":
        sampled = _sample_season_stratified(frame, size, seed)
    elif method == "rare_event_stratified":
        sampled = _sample_rare_event_stratified(
            frame, size, seed, rare_event_fraction, extreme_percentile
        )
    else:  # unreachable -- _validate already checked, but keeps intent explicit
        raise ValueError(f"Unhandled method {method!r}")

    if len(sampled) != size:
        raise AssertionError(
            f"Internal error: {method} produced {len(sampled)} rows, expected {size}."
        )

    return _finalise(sampled, feature_names)
