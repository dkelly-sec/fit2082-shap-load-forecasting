"""
Results schema and storage for the Weeks 8-9 SHAP sampling experiment
(RQ1).

Defines the single canonical structure every experiment run writes into,
so the orchestration loop, the stability metrics (Weeks 9-10) and the
reliability checks (RQ2) all read the same format rather than each
inventing their own.

Format: LONG, one row per (configuration x seed x feature)
---------------------------------------------------------
Long format is used rather than wide (one column per feature) because
every downstream analysis groups by configuration and/or feature:

  - within-configuration stability: group by config, compare rankings
    across seeds
  - between-configuration agreement: group by feature, compare across
    configs
  - bootstrap CIs on a feature's mean |SHAP|: filter to that feature
  - Spearman / Kendall's W / top-k Jaccard: all operate on per-seed
    ranking vectors reconstructed from these rows

Wide format would need reshaping for most of those, and would break
entirely if the feature set ever changed mid-experiment. Long format also
means adding a new stability metric later requires no schema change.

Columns
-------
    background_size          int    rows in the background sample
    background_method        str    uniform | kmeans | time_stratified
    evaluation_size          int    rows in the evaluation sample
    evaluation_fraction      float  evaluation_size as a fraction of the pool
    evaluation_method        str    random | season_stratified | rare_event_stratified
    seed                     int    repetition seed (~30 per configuration)
    feature                  str    model feature name
    mean_abs_shap            float  mean |SHAP value| for this feature
    rank                     int    1 = most important within this run
    n_features               int    total features ranked in this run
    additivity_max_error     float  from the correctness check, per run
    computed_at              str    ISO timestamp, for provenance

`config_id` is derived (not stored) via `add_config_id()` -- a stable
string identifying a configuration independent of seed, used for grouping
in the stability analysis.

Scale (full grid: 5 bg sizes x 3 bg methods x 5 eval sizes x 3 eval
methods x 30 seeds x 26 features) is ~175,500 rows, roughly 21 MB as CSV
or ~4 MB as Parquet -- small enough that no database is warranted.

Usage
-----
    from results_schema import ExperimentResult, ResultsWriter

    writer = ResultsWriter(Path("artifacts/sampling_experiment/results.csv"))
    for config in configurations:
        ranking = ...  # pd.Series, index=feature, value=mean|SHAP|
        writer.append(ExperimentResult(
            background_size=200, background_method="uniform",
            evaluation_size=500, evaluation_fraction=0.25,
            evaluation_method="random", seed=7,
            ranking=ranking, additivity_max_error=6.97,
        ))
    writer.close()
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SCHEMA_COLUMNS = [
    "background_size",
    "background_method",
    "evaluation_size",
    "evaluation_fraction",
    "evaluation_method",
    "seed",
    "feature",
    "mean_abs_shap",
    "rank",
    "n_features",
    "additivity_max_error",
    "computed_at",
]

SCHEMA_DTYPES = {
    "background_size": "int64",
    "background_method": "string",
    "evaluation_size": "int64",
    "evaluation_fraction": "float64",
    "evaluation_method": "string",
    "seed": "int64",
    "feature": "string",
    "mean_abs_shap": "float64",
    "rank": "int64",
    "n_features": "int64",
    "additivity_max_error": "float64",
    "computed_at": "string",
}

# Columns that together identify a CONFIGURATION (i.e. everything except
# the repetition seed and the per-feature payload). Grouping by these gives
# the ~30 repeated draws whose agreement is the stability measurement.
CONFIG_COLUMNS = [
    "background_size",
    "background_method",
    "evaluation_size",
    "evaluation_method",
]


@dataclass
class ExperimentResult:
    """
    One completed SHAP run: a single (configuration, seed) pair and the
    global ranking it produced. Expands to `n_features` schema rows.
    """
    background_size: int
    background_method: str
    evaluation_size: int
    evaluation_fraction: float
    evaluation_method: str
    seed: int
    ranking: pd.Series  # index = feature name, value = mean |SHAP|
    additivity_max_error: float
    computed_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))

    def to_frame(self) -> pd.DataFrame:
        if not isinstance(self.ranking, pd.Series):
            raise TypeError(f"ranking must be a pandas Series, got {type(self.ranking).__name__}")
        if self.ranking.empty:
            raise ValueError("ranking is empty -- nothing to record for this run.")
        if self.ranking.isna().any():
            raise ValueError("ranking contains NaN values; refusing to record a partial result.")
        if (self.ranking < 0).any():
            raise ValueError(
                "ranking contains negative values -- mean |SHAP| must be non-negative. "
                "Check the aggregation used absolute values."
            )

        # rank 1 = most important. method="min" so ties share the better
        # rank rather than being ordered arbitrarily, which would otherwise
        # create fake instability between seeds purely from tie-breaking.
        ordered = self.ranking.sort_values(ascending=False)
        ranks = ordered.rank(ascending=False, method="min").astype(int)

        frame = pd.DataFrame({
            "background_size": self.background_size,
            "background_method": self.background_method,
            "evaluation_size": self.evaluation_size,
            "evaluation_fraction": self.evaluation_fraction,
            "evaluation_method": self.evaluation_method,
            "seed": self.seed,
            "feature": ordered.index.astype(str),
            "mean_abs_shap": ordered.to_numpy(dtype=float),
            "rank": ranks.to_numpy(dtype=int),
            "n_features": len(ordered),
            "additivity_max_error": self.additivity_max_error,
            "computed_at": self.computed_at,
        })
        return frame[SCHEMA_COLUMNS]


class ResultsWriter:
    """
    Append-only CSV writer.

    Appends incrementally rather than accumulating everything in memory and
    writing once at the end, so a long grid run that crashes at hour three
    keeps every result computed up to that point. Results are also
    resumable: `completed_runs()` reports which (config, seed) pairs
    already exist, so a restarted run can skip them.
    """

    def __init__(self, path: Path, overwrite: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and self.path.exists():
            self.path.unlink()
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def append(self, result: ExperimentResult) -> int:
        frame = result.to_frame()
        frame.to_csv(self.path, mode="a", header=not self._header_written, index=False)
        self._header_written = True
        return len(frame)

    def completed_runs(self) -> set[tuple]:
        """
        Return the set of (background_size, background_method,
        evaluation_size, evaluation_method, seed) tuples already recorded,
        so an interrupted grid run can resume without recomputing them.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return set()
        existing = pd.read_csv(self.path, usecols=CONFIG_COLUMNS + ["seed"])
        return set(map(tuple, existing.drop_duplicates().to_numpy()))

    def close(self) -> None:
        # nothing buffered -- present for symmetry and future-proofing
        pass


def add_config_id(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add a stable `config_id` string identifying a configuration
    independent of seed, e.g.:

        bg200-uniform__ev500-random

    Derived rather than stored so the underlying columns stay queryable in
    their own right (you can filter on background_size numerically), while
    still giving the stability analysis a single convenient grouping key.
    """
    missing = [c for c in CONFIG_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Cannot build config_id: frame is missing {missing}")

    frame = frame.copy()
    frame["config_id"] = (
        "bg" + frame["background_size"].astype(str)
        + "-" + frame["background_method"].astype(str)
        + "__ev" + frame["evaluation_size"].astype(str)
        + "-" + frame["evaluation_method"].astype(str)
    )
    return frame


def load_results(path: Path, with_config_id: bool = True) -> pd.DataFrame:
    """Load a results file, validate its shape, and optionally add config_id."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- has the experiment been run yet?")

    frame = pd.read_csv(path)
    missing = [c for c in SCHEMA_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing expected schema columns: {missing}. "
            "It may have been written by an older/incompatible version."
        )
    return add_config_id(frame) if with_config_id else frame


def validate_results(frame: pd.DataFrame, expected_seeds: int | None = None) -> dict:
    """
    Check a results frame for the problems that would silently invalidate
    the stability analysis, and return a summary.

    Specifically catches:
      - duplicate (config, seed, feature) rows (double-counted runs)
      - configurations with fewer seeds than expected (partial/crashed runs)
      - configurations whose runs don't all rank the same feature set
        (which would make rank correlations meaningless)
    """
    if frame.empty:
        raise ValueError("Results frame is empty.")

    working = frame if "config_id" in frame.columns else add_config_id(frame)

    duplicates = working.duplicated(subset=["config_id", "seed", "feature"]).sum()

    seeds_per_config = working.groupby("config_id")["seed"].nunique()
    features_per_run = working.groupby(["config_id", "seed"])["feature"].nunique()
    inconsistent_feature_counts = features_per_run.groupby("config_id").nunique()
    inconsistent = inconsistent_feature_counts[inconsistent_feature_counts > 1].index.tolist()

    summary = {
        "rows": int(len(working)),
        "configurations": int(working["config_id"].nunique()),
        "features": int(working["feature"].nunique()),
        "duplicate_rows": int(duplicates),
        "seeds_per_config_min": int(seeds_per_config.min()),
        "seeds_per_config_max": int(seeds_per_config.max()),
        "configs_with_inconsistent_feature_sets": inconsistent,
    }

    if expected_seeds is not None:
        short = seeds_per_config[seeds_per_config < expected_seeds]
        summary["configs_below_expected_seeds"] = short.to_dict()

    return summary
