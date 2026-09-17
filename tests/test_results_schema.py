"""
Tests for results_schema.py -- the canonical storage format for the
Weeks 8-9 sampling experiment.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from results_schema import (  # noqa: E402
    SCHEMA_COLUMNS,
    ExperimentResult,
    ResultsWriter,
    add_config_id,
    load_results,
    validate_results,
)


def _ranking(values: dict) -> pd.Series:
    return pd.Series(values)


def _result(seed: int = 1, bg_size: int = 200, bg_method: str = "uniform",
            ev_size: int = 500, ev_method: str = "random") -> ExperimentResult:
    return ExperimentResult(
        background_size=bg_size,
        background_method=bg_method,
        evaluation_size=ev_size,
        evaluation_fraction=0.25,
        evaluation_method=ev_method,
        seed=seed,
        ranking=_ranking({"lag_5min": 296.2, "day_of_week": 144.4, "temperature": 60.4}),
        additivity_max_error=6.97,
    )


def test_result_expands_to_one_row_per_feature_with_full_schema():
    frame = _result().to_frame()
    assert len(frame) == 3
    assert list(frame.columns) == SCHEMA_COLUMNS


def test_ranks_are_assigned_by_descending_importance():
    frame = _result().to_frame()
    ordered = frame.sort_values("rank")
    assert ordered.iloc[0]["feature"] == "lag_5min"
    assert ordered.iloc[0]["rank"] == 1
    assert ordered.iloc[-1]["feature"] == "temperature"
    assert ordered.iloc[-1]["rank"] == 3


def test_tied_importances_share_a_rank_rather_than_breaking_arbitrarily():
    """
    Arbitrary tie-breaking would create fake seed-to-seed instability that
    has nothing to do with sampling -- precisely the thing this project
    measures. Ties must resolve deterministically.
    """
    result = ExperimentResult(
        background_size=100, background_method="uniform",
        evaluation_size=100, evaluation_fraction=0.1,
        evaluation_method="random", seed=1,
        ranking=_ranking({"a": 50.0, "b": 50.0, "c": 10.0}),
        additivity_max_error=1.0,
    )
    frame = result.to_frame()
    ranks = dict(zip(frame["feature"], frame["rank"]))
    assert ranks["a"] == ranks["b"] == 1
    assert ranks["c"] == 3


def test_rejects_nan_negative_and_empty_rankings():
    base = dict(
        background_size=100, background_method="uniform", evaluation_size=100,
        evaluation_fraction=0.1, evaluation_method="random", seed=1,
        additivity_max_error=1.0,
    )
    with pytest.raises(ValueError, match="NaN"):
        ExperimentResult(**base, ranking=_ranking({"a": 1.0, "b": np.nan})).to_frame()
    with pytest.raises(ValueError, match="negative"):
        ExperimentResult(**base, ranking=_ranking({"a": 1.0, "b": -2.0})).to_frame()
    with pytest.raises(ValueError, match="empty"):
        ExperimentResult(**base, ranking=pd.Series(dtype=float)).to_frame()


def test_writer_appends_incrementally_and_reloads(tmp_path):
    path = tmp_path / "results.csv"
    writer = ResultsWriter(path)
    for seed in range(3):
        writer.append(_result(seed=seed))
    writer.close()

    loaded = load_results(path)
    assert len(loaded) == 9           # 3 seeds x 3 features
    assert loaded["seed"].nunique() == 3
    assert "config_id" in loaded.columns


def test_writer_survives_interruption_and_can_resume(tmp_path):
    """
    A multi-hour grid run must not lose everything if it crashes partway.
    Simulates a crash by abandoning the writer, then confirms prior results
    persisted and completed_runs() reports exactly what to skip on restart.
    """
    path = tmp_path / "results.csv"

    writer = ResultsWriter(path)
    writer.append(_result(seed=0))
    writer.append(_result(seed=1))
    del writer  # simulate abrupt termination -- no close(), no flush call

    resumed = ResultsWriter(path)
    done = resumed.completed_runs()
    assert (200, "uniform", 500, "random", 0) in done
    assert (200, "uniform", 500, "random", 1) in done
    assert (200, "uniform", 500, "random", 2) not in done

    resumed.append(_result(seed=2))
    assert len(load_results(path)) == 9


def test_config_id_groups_seeds_together_but_separates_configurations():
    frames = [
        _result(seed=1).to_frame(),
        _result(seed=2).to_frame(),                      # same config, different seed
        _result(seed=1, bg_method="kmeans").to_frame(),  # different config
    ]
    combined = add_config_id(pd.concat(frames, ignore_index=True))
    assert combined["config_id"].nunique() == 2
    per_config_seeds = combined.groupby("config_id")["seed"].nunique().to_dict()
    assert sorted(per_config_seeds.values()) == [1, 2]


def test_validate_flags_duplicates_and_short_seed_counts(tmp_path):
    frames = [_result(seed=s).to_frame() for s in range(3)]
    frames.append(_result(seed=0).to_frame())  # deliberate duplicate run
    combined = pd.concat(frames, ignore_index=True)

    summary = validate_results(combined, expected_seeds=30)
    assert summary["duplicate_rows"] == 3          # 3 features duplicated
    assert summary["configurations"] == 1
    assert summary["configs_below_expected_seeds"]  # only 3 of 30 seeds present


def test_validate_flags_inconsistent_feature_sets_within_a_configuration():
    """
    If one seed ranks 26 features and another ranks 24, rank correlations
    between them are meaningless -- this must be caught, not averaged over.
    """
    normal = _result(seed=1).to_frame()
    odd = ExperimentResult(
        background_size=200, background_method="uniform", evaluation_size=500,
        evaluation_fraction=0.25, evaluation_method="random", seed=2,
        ranking=_ranking({"lag_5min": 1.0, "day_of_week": 2.0}),  # only 2 features
        additivity_max_error=1.0,
    ).to_frame()

    summary = validate_results(pd.concat([normal, odd], ignore_index=True))
    assert len(summary["configs_with_inconsistent_feature_sets"]) == 1


def test_load_rejects_a_file_missing_schema_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"feature": ["a"], "mean_abs_shap": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected schema columns"):
        load_results(path)


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", "-v", __file__])
