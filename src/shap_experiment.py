"""Run a controlled repeated-sampling SHAP pilot and quantify rank stability."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_model import load_frozen_model  # noqa: E402
from sampling import construct_sample, define_outcome_demand_events, define_rare_events  # noqa: E402
from shap_pilot import (  # noqa: E402
    BACKGROUND_METHODS,
    EVALUATION_METHODS,
    check_additivity,
    compute_global_shap_ranking,
)
from training import chronological_split, load_config, prepare_frame  # noqa: E402


def ranking_stability(left: pd.Series, right: pd.Series, top_k: int = 10) -> dict:
    """Compare two complete feature rankings using three complementary metrics."""
    if set(left.index) != set(right.index):
        raise ValueError("Rankings must contain exactly the same features.")
    left_rank = left.rank(ascending=False, method="average")
    right_rank = right.reindex(left.index).rank(ascending=False, method="average")
    top_k = min(top_k, len(left))
    left_top = set(left.nlargest(top_k).index)
    right_top = set(right.nlargest(top_k).index)
    return {
        "spearman": float(left_rank.corr(right_rank, method="spearman")),
        "kendall": float(left_rank.corr(right_rank, method="kendall")),
        "top_k": int(top_k),
        "top_k_overlap": len(left_top & right_top) / top_k,
    }


def summarise_stability(pairwise: pd.DataFrame) -> dict:
    if pairwise.empty:
        return {"pair_count": 0}
    return {
        "pair_count": int(len(pairwise)),
        "spearman_mean": float(pairwise["spearman"].mean()),
        "spearman_min": float(pairwise["spearman"].min()),
        "kendall_mean": float(pairwise["kendall"].mean()),
        "kendall_min": float(pairwise["kendall"].min()),
        "top_k_overlap_mean": float(pairwise["top_k_overlap"].mean()),
        "top_k_overlap_min": float(pairwise["top_k_overlap"].min()),
    }


def run_experiment(
    data_path: Path,
    config_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    background_methods: list[str],
    background_sizes: list[int],
    evaluation_method: str,
    evaluation_size: int,
    seeds: list[int],
    top_k: int = 10,
) -> dict:
    """Run the grid against one frozen model and separated train/test pools."""
    bundle = load_frozen_model(artifacts_dir)
    config = load_config(config_path)
    frame, feature_names = prepare_frame(data_path, config)
    if feature_names != bundle.feature_names:
        raise ValueError("Prepared feature order does not match the frozen model.")
    splits = chronological_split(frame, config)
    rare_definition = define_rare_events(splits["train"])
    outcome_definition = define_outcome_demand_events(splits["train"])
    unsupported_background = set(background_methods) - set(BACKGROUND_METHODS)
    if unsupported_background:
        raise ValueError(f"Unsupported background methods: {sorted(unsupported_background)}")
    if evaluation_method not in EVALUATION_METHODS:
        raise ValueError(f"Unsupported evaluation method: {evaluation_method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings: dict[str, pd.Series] = {}
    run_rows: list[dict] = []
    for method, size, seed in itertools.product(background_methods, background_sizes, seeds):
        run_id = f"bg-{method}_n-{size}_eval-{evaluation_method}_n-{evaluation_size}_seed-{seed}"
        print(f"Running {run_id}")
        background = construct_sample(
            splits["train"], bundle.feature_names, size, seed,
            method, rare_definition, outcome_definition,
        )
        evaluation = construct_sample(
            splits["test"], bundle.feature_names, evaluation_size, seed + 100_000,
            evaluation_method, rare_definition, outcome_definition,
        )
        started = time.perf_counter()
        ranking, explainer, shap_values = compute_global_shap_ranking(bundle, background, evaluation)
        additivity = check_additivity(bundle, explainer, shap_values, evaluation)
        elapsed = time.perf_counter() - started
        rankings[run_id] = ranking
        ranking.rename("mean_abs_shap").to_csv(output_dir / f"{run_id}.csv")
        run_rows.append({
            "run_id": run_id,
            "background_method": method,
            "background_size": size,
            "evaluation_method": evaluation_method,
            "evaluation_size": evaluation_size,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "additivity_status": additivity["status"],
            "additivity_max_error_mw": additivity["max_error_mw"],
            "additivity_warning_threshold_mw": additivity["warning_threshold_mw"],
            "additivity_hard_limit_mw": additivity["hard_limit_mw"],
        })

    pair_rows: list[dict] = []
    for left_id, right_id in itertools.combinations(rankings, 2):
        metrics = ranking_stability(rankings[left_id], rankings[right_id], top_k)
        pair_rows.append({"left_run": left_id, "right_run": right_id, **metrics})
    runs = pd.DataFrame(run_rows)
    pairwise = pd.DataFrame(pair_rows)
    summary = {
        "run_count": len(run_rows),
        "background_pool": "purged training split",
        "evaluation_pool": "held-out purged test split",
        "rare_event_definition": rare_definition.to_dict(),
        "outcome_demand_definition": outcome_definition.to_dict(),
        "runtime_seconds_total": float(runs["elapsed_seconds"].sum()),
        "runtime_seconds_mean": float(runs["elapsed_seconds"].mean()),
        "stability": summarise_stability(pairwise),
    }
    runs.to_csv(output_dir / "runs.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_stability.csv", index=False)
    rankings_wide = pd.DataFrame(rankings).rename_axis("feature")
    rankings_wide.to_csv(output_dir / "rankings_wide.csv")
    rankings_long = rankings_wide.reset_index().melt(
        id_vars="feature", var_name="run_id", value_name="mean_abs_shap"
    )
    rankings_long["rank"] = rankings_long.groupby("run_id")["mean_abs_shap"].rank(
        ascending=False, method="min"
    ).astype(int)
    rankings_long.to_csv(output_dir / "rankings_long.csv", index=False)
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "rare_event_definition.json").write_text(
        json.dumps(rare_definition.to_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / "outcome_demand_definition.json").write_text(
        json.dumps(outcome_definition.to_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / "results_schema.json").write_text(json.dumps({
        "schema_version": 1,
        "runs.csv": list(runs.columns),
        "pairwise_stability.csv": list(pairwise.columns),
        "rankings_long.csv": list(rankings_long.columns),
        "rankings_wide.csv": ["feature", "<one column per run_id>"],
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/training_real"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/shap_week8_experiment"))
    parser.add_argument("--background-methods", default="uniform,time_stratified")
    parser.add_argument("--background-sizes", default="50,200")
    parser.add_argument("--evaluation-method", default="time_stratified")
    parser.add_argument("--evaluation-size", type=int, default=100)
    parser.add_argument("--seeds", default="2082,2083,2084")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    run_experiment(
        args.data, args.config, args.artifacts, args.output,
        _csv_strings(args.background_methods), _csv_ints(args.background_sizes),
        args.evaluation_method, args.evaluation_size, _csv_ints(args.seeds), args.top_k,
    )


if __name__ == "__main__":
    main()
