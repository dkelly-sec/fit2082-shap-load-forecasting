"""
Tests for analyse_stability.py. The statistics helpers are checked against
known answers, and an end-to-end test builds a small synthetic grid with
shap_experiment.py's own analysis functions, so the analysis script is
tested against exactly the file formats the real grid produces.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyse_stability import (  # noqa: E402
    bootstrap_ci,
    dunn_test,
    holm_adjust,
    run_analysis,
    size_paired_tests,
    top5_swaps,
)
from shap_experiment import (  # noqa: E402
    CONFIG_COLUMNS,
    between_condition_analysis,
    build_grid,
    consensus_rankings,
    within_condition_analysis,
)

FEATURES = [f"f{i}" for i in range(8)]


# --------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------

def test_holm_matches_hand_calculation():
    # sorted p: 0.01, 0.03, 0.04 -> 3*0.01, 2*0.03, max(prev, 1*0.04) = 0.03, 0.06, 0.06
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_holm_never_exceeds_one_and_is_monotone_in_sorted_order():
    p = np.array([0.5, 0.2, 0.9, 0.001])
    adjusted = holm_adjust(p)
    assert (adjusted <= 1).all()
    assert np.all(np.diff(adjusted[np.argsort(p)]) >= 0)


def test_dunn_separates_shifted_groups_but_not_identical_ones():
    rng = np.random.default_rng(0)
    values = pd.Series(np.concatenate([rng.normal(0, 1, 40), rng.normal(0, 1, 40), rng.normal(5, 1, 40)]))
    groups = pd.Series(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    result = dunn_test(values, groups).set_index(["group_a", "group_b"])
    assert result.loc[("a", "b"), "p_holm"] > 0.05
    assert result.loc[("a", "c"), "p_holm"] < 0.001
    assert result.loc[("b", "c"), "p_holm"] < 0.001


def test_bootstrap_ci_contains_mean_and_collapses_for_constant_feature():
    rows = []
    rng = np.random.default_rng(1)
    for seed in range(20):
        rows.append({"config_id": "c", "seed": seed, "feature": "varying", "mean_abs_shap": rng.normal(100, 10)})
        rows.append({"config_id": "c", "seed": seed, "feature": "constant", "mean_abs_shap": 50.0})
    ci = bootstrap_ci(pd.DataFrame(rows), n_boot=500).set_index("feature")
    assert ci.loc["varying", "ci_low"] < ci.loc["varying", "mean_abs_shap"] < ci.loc["varying", "ci_high"]
    assert ci.loc["constant", "ci_low"] == ci.loc["constant", "ci_high"] == 50.0


def test_paired_size_tests_pair_matching_configurations():
    rows = []
    for bg_method in ["uniform", "kmeans"]:
        for ev_method in ["uniform", "rare"]:
            for ev_size in [100, 500]:
                for bg_size, w in [(25, 0.90), (200, 0.95)]:
                    rows.append({"background_method": bg_method, "evaluation_method": ev_method,
                                 "evaluation_size": ev_size, "background_size": bg_size,
                                 "kendalls_w": w + 0.001 * ev_size / 100,
                                 "jaccard_top5_mean": 1.0, "jaccard_top10_mean": 1.0})
    table = size_paired_tests(pd.DataFrame(rows))
    row = table[(table["size_factor"] == "background_size") & (table["metric"] == "kendalls_w")].iloc[0]
    assert row["pairs"] == 8
    assert row["median_difference"] == pytest.approx(-0.05)
    assert row["share_smaller_is_lower"] == 1.0
    identical = table[(table["size_factor"] == "background_size") & (table["metric"] == "jaccard_top5_mean")].iloc[0]
    assert identical["p"] == 1.0  # no differences at all


# --------------------------------------------------------------------------
# Synthetic grid built with the real shap_experiment analysis functions
# --------------------------------------------------------------------------

def _synthetic_grid(folder: Path, seeds=range(6)) -> None:
    rng = np.random.default_rng(2)
    grid = build_grid(["uniform", "kmeans"], [25, 50], ["uniform", "rare_event_stratified"], [100, 500], list(seeds))
    base = pd.Series(np.linspace(100, 10, len(FEATURES)), index=FEATURES)
    rankings, run_rows = {}, []
    for spec in grid:
        noise = 2 if spec["background_size"] == 50 else 6
        ranking = base + rng.normal(0, noise, len(FEATURES))
        if spec["evaluation_method"] == "rare_event_stratified":
            ranking["f5"] += 10  # pushes f5 towards the top-5 boundary, like temp_max
        rankings[spec["run_id"]] = ranking.clip(lower=0)
        run_rows.append({**spec, "elapsed_seconds": spec["background_size"] * spec["evaluation_size"] / 5000,
                         "additivity_status": "pass", "additivity_max_error_mw": 0.0,
                         "additivity_warning_threshold_mw": 50.0, "additivity_hard_limit_mw": 250.0})
    runs = pd.DataFrame(run_rows)
    pairwise, within = within_condition_analysis(rankings, runs)
    consensus = consensus_rankings(rankings, runs)
    reference = "bg-uniform_n-50__eval-uniform_n-500"
    between = between_condition_analysis(consensus, runs, reference)
    long = pd.concat([
        rankings[row["run_id"]].rename("mean_abs_shap").rename_axis("feature").reset_index()
        .assign(run_id=row["run_id"], config_id=row["config_id"], seed=row["seed"],
                **{c: row[c] for c in CONFIG_COLUMNS})
        for _, row in runs.iterrows()], ignore_index=True)
    long["rank"] = long.groupby("run_id")["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    folder.mkdir(parents=True, exist_ok=True)
    runs.to_csv(folder / "runs.csv", index=False)
    long.to_csv(folder / "rankings_long.csv", index=False)
    within.to_csv(folder / "within_condition_stability.csv", index=False)
    consensus.rename_axis("feature").to_csv(folder / "consensus_rankings.csv")
    between.to_csv(folder / "between_condition_agreement.csv", index=False)


def test_end_to_end_writes_every_output(tmp_path):
    grid, out = tmp_path / "grid", tmp_path / "analysis"
    _synthetic_grid(grid)
    summary = run_analysis(grid, out, n_boot=200)
    expected = ["factor_effects_within.csv", "factor_effects_between.csv", "kruskal_wallis.csv",
                "dunn_posthoc.csv", "size_tradeoff.csv", "size_paired_tests.csv", "size_regression.csv",
                "top5_boundary.csv", "feature_importance_by_method.csv", "topk_changes_vs_reference.csv",
                "top5_swaps_by_run.csv", "top5_swap_summary.csv", "bootstrap_ci.csv",
                "stability_vs_background_size.png", "top5_by_evaluation_method.png",
                "temperature_importance_by_method.png", "cost_vs_stability.png", "summary.json"]
    for name in expected:
        assert (out / name).exists(), name
    assert summary["configurations"] == 16
    assert summary["runs"] == 96
    assert json.loads((out / "summary.json").read_text())["reference_configuration"] == \
        "bg-uniform_n-50__eval-uniform_n-500"


def test_swap_detection_matches_a_direct_recount(tmp_path):
    grid = tmp_path / "grid"
    _synthetic_grid(grid)
    long = pd.read_csv(grid / "rankings_long.csv")
    consensus = pd.read_csv(grid / "consensus_rankings.csv", index_col="feature")
    reference = "bg-uniform_n-50__eval-uniform_n-500"
    by_run, summary = top5_swaps(long, consensus, reference)
    reference_top5 = set(consensus[reference].nlargest(5).index)
    direct = sum(set(g.loc[g["rank"] <= 5, "feature"]) != reference_top5 for _, g in long.groupby("run_id"))
    assert int(by_run["differs"].sum()) == direct
    assert summary["runs"].sum() == long["run_id"].nunique()


def test_missing_grid_files_raise_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        run_analysis(tmp_path / "nothing_here", tmp_path / "out")
