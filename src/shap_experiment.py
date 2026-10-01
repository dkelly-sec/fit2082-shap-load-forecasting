"""Run the repeated-sampling SHAP grid and quantify ranking stability (RQ1).

Grid
----
Every combination of background method x background size x evaluation
method x evaluation size x seed is run against the one frozen model.
Background samples come only from the purged training split; evaluation
samples come only from the held-out purged test split.

Stability is measured at two distinct levels, matching the project spec:

  Within-condition stability
      Seeds are compared ONLY against other seeds of the SAME
      configuration. This answers "if I repeat this exact sampling
      design, do I get the same ranking?" -- the core RQ1 measurement.
      Reported as pairwise Spearman, Kendall tau, top-5/top-10 Jaccard,
      and Kendall's coefficient of concordance (W) across all seeds.

  Between-condition agreement
      Each configuration's CONSENSUS ranking (mean |SHAP| across its
      seeds) is compared against the consensus ranking of a fixed
      REFERENCE configuration (by default the largest uniform/uniform
      configuration, per the spec's treatment of the largest sample as
      the stable reference). This answers "does changing the sampling
      design change the answer?"

The original Week 8 pilot compared every run against every other run,
which blended these two questions into one number. They are now kept
separate.

Resume
------
Each completed run is written to ``<output>/runs/<run_id>.csv`` and
``<run_id>.json`` as soon as it finishes. Re-running the same command
with the same output directory skips completed runs, so a long grid that
is interrupted can be continued rather than restarted. An experiment
manifest guards against resuming into a directory produced from a
different model or dataset.

Usage
-----
    python src/shap_experiment.py \\
        --data data/interim/merged_full_frozen.csv \\
        --artifacts artifacts/training_real \\
        --background-methods uniform,kmeans,time_stratified \\
        --background-sizes 50,200 \\
        --evaluation-methods uniform,time_stratified,season_stratified,rare_event_stratified,outcome_demand_stratified \\
        --evaluation-sizes 100,500 \\
        --n-seeds 30 \\
        --output artifacts/shap_week9_grid
"""

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

CONFIG_COLUMNS = ["background_method", "background_size", "evaluation_method", "evaluation_size"]
EVALUATION_SEED_OFFSET = 100_000


# --------------------------------------------------------------------------
# Ranking comparison metrics
# --------------------------------------------------------------------------

def top_k_jaccard(left: pd.Series, right: pd.Series, k: int) -> float:
    """|A intersect B| / |A union B| for the top-k feature sets."""
    k = min(k, len(left))
    left_top = set(left.nlargest(k).index)
    right_top = set(right.nlargest(k).index)
    union = left_top | right_top
    return len(left_top & right_top) / len(union) if union else 1.0


def ranking_stability(left: pd.Series, right: pd.Series, top_k: int = 10) -> dict:
    """Compare two complete feature rankings using complementary metrics."""
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
        "top_k_jaccard": top_k_jaccard(left, right, top_k),
    }


def kendalls_w(rankings: pd.DataFrame) -> float:
    """
    Kendall's coefficient of concordance across repeated rankings.

    ``rankings`` has one row per repetition (seed) and one column per
    feature, holding mean |SHAP| values. Returns W in [0, 1]: 1 means every
    repetition produced the identical ordering, 0 means no agreement.
    Uses average ranks for ties with the standard tie correction, since
    zero-importance features commonly tie at exactly 0.
    """
    m, n = rankings.shape
    if m < 2 or n < 2:
        return float("nan")
    ranks = rankings.rank(axis=1, ascending=False, method="average")
    rank_sums = ranks.sum(axis=0)
    s = float(((rank_sums - rank_sums.mean()) ** 2).sum())
    tie_correction = 0.0
    for _, row in ranks.iterrows():
        counts = row.value_counts()
        tie_correction += float(((counts ** 3) - counts).sum())
    denominator = m ** 2 * (n ** 3 - n) - m * tie_correction
    if denominator <= 0:
        return float("nan")
    return float(12 * s / denominator)


def summarise_stability(pairwise: pd.DataFrame) -> dict:
    """Mean/minimum summary of a set of pairwise comparisons."""
    if pairwise.empty:
        return {"pair_count": 0}
    summary = {
        "pair_count": int(len(pairwise)),
        "spearman_mean": float(pairwise["spearman"].mean()),
        "spearman_min": float(pairwise["spearman"].min()),
        "kendall_mean": float(pairwise["kendall"].mean()),
        "kendall_min": float(pairwise["kendall"].min()),
        "top_k_overlap_mean": float(pairwise["top_k_overlap"].mean()),
        "top_k_overlap_min": float(pairwise["top_k_overlap"].min()),
    }
    for column in ("jaccard_top5", "jaccard_top10"):
        if column in pairwise:
            summary[f"{column}_mean"] = float(pairwise[column].mean())
            summary[f"{column}_min"] = float(pairwise[column].min())
    return summary


# --------------------------------------------------------------------------
# Grid bookkeeping
# --------------------------------------------------------------------------

def config_id(background_method: str, background_size: int,
              evaluation_method: str, evaluation_size: int) -> str:
    return (f"bg-{background_method}_n-{background_size}"
            f"__eval-{evaluation_method}_n-{evaluation_size}")


def run_id(config: str, seed: int) -> str:
    return f"{config}__seed-{seed}"


def build_grid(background_methods, background_sizes, evaluation_methods,
               evaluation_sizes, seeds) -> list[dict]:
    grid = []
    for bg_method, bg_size, ev_method, ev_size, seed in itertools.product(
        background_methods, background_sizes, evaluation_methods, evaluation_sizes, seeds
    ):
        cid = config_id(bg_method, bg_size, ev_method, ev_size)
        grid.append({
            "run_id": run_id(cid, seed),
            "config_id": cid,
            "background_method": bg_method,
            "background_size": int(bg_size),
            "evaluation_method": ev_method,
            "evaluation_size": int(ev_size),
            "seed": int(seed),
        })
    return grid


def default_reference(background_methods, background_sizes,
                      evaluation_methods, evaluation_sizes) -> str:
    """Largest configuration, preferring uniform sampling on both sides."""
    bg_method = "uniform" if "uniform" in background_methods else background_methods[0]
    ev_method = "uniform" if "uniform" in evaluation_methods else evaluation_methods[0]
    return config_id(bg_method, max(background_sizes), ev_method, max(evaluation_sizes))


def _validate_grid(background_methods, background_sizes, evaluation_methods,
                   evaluation_sizes, seeds, splits) -> None:
    unsupported_bg = set(background_methods) - set(BACKGROUND_METHODS)
    if unsupported_bg:
        raise ValueError(f"Unsupported background methods: {sorted(unsupported_bg)}")
    unsupported_ev = set(evaluation_methods) - set(EVALUATION_METHODS)
    if unsupported_ev:
        raise ValueError(f"Unsupported evaluation methods: {sorted(unsupported_ev)}")
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    if max(background_sizes) > len(splits["train"]):
        raise ValueError(f"Background size {max(background_sizes)} exceeds training rows ({len(splits['train'])}).")
    if max(evaluation_sizes) > len(splits["test"]):
        raise ValueError(f"Evaluation size {max(evaluation_sizes)} exceeds test rows ({len(splits['test'])}).")


def _preflight(splits, feature_names, background_methods, background_sizes,
               evaluation_methods, evaluation_sizes, rare_definition, outcome_definition) -> None:
    """
    Construct one sample for every distinct (method, size) before starting,
    so a problem such as too few rare-event rows for the largest
    evaluation size fails in seconds rather than hours into the run.
    """
    for method, size in itertools.product(background_methods, background_sizes):
        construct_sample(splits["train"], feature_names, size, 0, method,
                         rare_definition, outcome_definition)
    for method, size in itertools.product(evaluation_methods, evaluation_sizes):
        construct_sample(splits["test"], feature_names, size, EVALUATION_SEED_OFFSET, method,
                         rare_definition, outcome_definition)


def _check_or_write_manifest(output_dir: Path, manifest: dict) -> None:
    """Refuse to resume into a directory produced from a different model/dataset."""
    path = output_dir / "experiment_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("feature_names", "data_rows", "train_rows", "test_rows", "model_best_iteration"):
            if existing.get(key) != manifest.get(key):
                raise ValueError(
                    f"Cannot resume into {output_dir}: '{key}' differs from the run that created it "
                    f"({existing.get(key)!r} vs {manifest.get(key)!r}). Use a new --output directory."
                )
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_completed(runs_dir: Path, rid: str):
    meta_path = runs_dir / f"{rid}.json"
    ranking_path = runs_dir / f"{rid}.csv"
    # The metadata JSON is written last, so its presence marks a complete run.
    if not (meta_path.exists() and ranking_path.exists()):
        return None
    ranking = pd.read_csv(ranking_path, index_col="feature")["mean_abs_shap"]
    return ranking, json.loads(meta_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def within_condition_analysis(rankings: dict[str, pd.Series], runs: pd.DataFrame):
    """Pairwise comparisons restricted to seeds of the same configuration."""
    pair_rows, condition_rows = [], []
    for cid, group in runs.groupby("config_id", sort=False):
        group = group.sort_values("seed")
        ids = group["run_id"].tolist()
        for left_id, right_id in itertools.combinations(ids, 2):
            left, right = rankings[left_id], rankings[right_id]
            metrics = ranking_stability(left, right, top_k=10)
            pair_rows.append({
                "config_id": cid,
                **{c: group.iloc[0][c] for c in CONFIG_COLUMNS},
                "left_seed": int(group.loc[group["run_id"] == left_id, "seed"].iloc[0]),
                "right_seed": int(group.loc[group["run_id"] == right_id, "seed"].iloc[0]),
                "spearman": metrics["spearman"],
                "kendall": metrics["kendall"],
                "top_k_overlap": metrics["top_k_overlap"],
                "jaccard_top5": top_k_jaccard(left, right, 5),
                "jaccard_top10": top_k_jaccard(left, right, 10),
            })
        matrix = pd.DataFrame([rankings[i] for i in ids])
        config_pairs = pd.DataFrame([r for r in pair_rows if r["config_id"] == cid])
        summary = summarise_stability(config_pairs) if not config_pairs.empty else {"pair_count": 0}
        condition_rows.append({
            "config_id": cid,
            **{c: group.iloc[0][c] for c in CONFIG_COLUMNS},
            "n_seeds": int(len(ids)),
            **summary,
            "kendalls_w": kendalls_w(matrix),
            "runtime_seconds_mean": float(group["elapsed_seconds"].mean()),
            "additivity_warnings": int((group["additivity_status"] == "warning").sum()),
        })
    return pd.DataFrame(pair_rows), pd.DataFrame(condition_rows)


def consensus_rankings(rankings: dict[str, pd.Series], runs: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature across the seeds of each configuration."""
    rows = {}
    for cid, group in runs.groupby("config_id", sort=False):
        rows[cid] = pd.DataFrame([rankings[i] for i in group["run_id"]]).mean(axis=0)
    return pd.DataFrame(rows)  # features x config_id


def between_condition_analysis(consensus: pd.DataFrame, runs: pd.DataFrame, reference: str) -> pd.DataFrame:
    if reference not in consensus.columns:
        raise ValueError(f"Reference configuration {reference!r} is not in the completed grid.")
    reference_ranking = consensus[reference]
    configs = runs.drop_duplicates("config_id").set_index("config_id")
    rows = []
    for cid in consensus.columns:
        metrics = ranking_stability(consensus[cid], reference_ranking, top_k=10)
        rows.append({
            "config_id": cid,
            **{c: configs.loc[cid, c] for c in CONFIG_COLUMNS},
            "is_reference": cid == reference,
            "spearman_vs_reference": metrics["spearman"],
            "kendall_vs_reference": metrics["kendall"],
            "jaccard_top5_vs_reference": top_k_jaccard(consensus[cid], reference_ranking, 5),
            "jaccard_top10_vs_reference": top_k_jaccard(consensus[cid], reference_ranking, 10),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Experiment runner
# --------------------------------------------------------------------------

def run_experiment(
    data_path: Path,
    config_path: Path,
    artifacts_dir: Path,
    output_dir: Path,
    background_methods: list[str],
    background_sizes: list[int],
    evaluation_methods: list[str],
    evaluation_sizes: list[int],
    seeds: list[int],
    reference: str | None = None,
) -> dict:
    """Run (or resume) the full grid against one frozen model."""
    output_dir = Path(output_dir)
    bundle = load_frozen_model(artifacts_dir)
    config = load_config(config_path)
    frame, feature_names = prepare_frame(data_path, config)
    if feature_names != bundle.feature_names:
        raise ValueError("Prepared feature order does not match the frozen model.")
    splits = chronological_split(frame, config)
    _validate_grid(background_methods, background_sizes, evaluation_methods,
                   evaluation_sizes, seeds, splits)

    rare_definition = define_rare_events(splits["train"])
    outcome_definition = define_outcome_demand_events(splits["train"])
    reference = reference or default_reference(
        background_methods, background_sizes, evaluation_methods, evaluation_sizes)

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    _check_or_write_manifest(output_dir, {
        "data_path": str(Path(data_path)),
        "artifacts_dir": str(Path(artifacts_dir)),
        "feature_names": bundle.feature_names,
        "data_rows": int(len(frame)),
        "train_rows": int(len(splits["train"])),
        "test_rows": int(len(splits["test"])),
        "model_best_iteration": bundle.best_params.get("best_iteration"),
    })

    print("Preflight: constructing one sample per method/size...")
    _preflight(splits, bundle.feature_names, background_methods, background_sizes,
               evaluation_methods, evaluation_sizes, rare_definition, outcome_definition)

    grid = build_grid(background_methods, background_sizes, evaluation_methods,
                      evaluation_sizes, seeds)
    rankings: dict[str, pd.Series] = {}
    run_rows: list[dict] = []
    computed = 0
    started_all = time.perf_counter()

    for index, spec in enumerate(grid, start=1):
        rid = spec["run_id"]
        completed = _load_completed(runs_dir, rid)
        if completed is not None:
            ranking, meta = completed
            rankings[rid] = ranking
            run_rows.append(meta)
            continue

        background = construct_sample(
            splits["train"], bundle.feature_names, spec["background_size"], spec["seed"],
            spec["background_method"], rare_definition, outcome_definition,
        )
        evaluation = construct_sample(
            splits["test"], bundle.feature_names, spec["evaluation_size"],
            spec["seed"] + EVALUATION_SEED_OFFSET,
            spec["evaluation_method"], rare_definition, outcome_definition,
        )
        started = time.perf_counter()
        ranking, explainer, shap_values = compute_global_shap_ranking(bundle, background, evaluation)
        additivity = check_additivity(bundle, explainer, shap_values, evaluation)
        elapsed = time.perf_counter() - started

        meta = {
            **spec,
            "elapsed_seconds": elapsed,
            "additivity_status": additivity["status"],
            "additivity_max_error_mw": additivity["max_error_mw"],
            "additivity_warning_threshold_mw": additivity["warning_threshold_mw"],
            "additivity_hard_limit_mw": additivity["hard_limit_mw"],
        }
        ranking.rename("mean_abs_shap").rename_axis("feature").to_csv(runs_dir / f"{rid}.csv")
        (runs_dir / f"{rid}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rankings[rid] = ranking
        run_rows.append(meta)
        computed += 1

        spent = time.perf_counter() - started_all
        remaining = len(grid) - index
        eta_minutes = (spent / computed) * remaining / 60 if computed else float("nan")
        print(f"[{index}/{len(grid)}] {rid}  {elapsed:.1f}s  "
              f"({additivity['status']})  ETA ~{eta_minutes:.0f} min")

    runs = pd.DataFrame(run_rows)
    pairwise, within = within_condition_analysis(rankings, runs)
    consensus = consensus_rankings(rankings, runs)
    between = between_condition_analysis(consensus, runs, reference)

    rankings_long = pd.concat(
        [
            rankings[row["run_id"]].rename("mean_abs_shap").rename_axis("feature").reset_index()
            .assign(run_id=row["run_id"], config_id=row["config_id"], seed=row["seed"],
                    **{c: row[c] for c in CONFIG_COLUMNS})
            for _, row in runs.iterrows()
        ],
        ignore_index=True,
    )
    rankings_long["rank"] = rankings_long.groupby("run_id")["mean_abs_shap"].rank(
        ascending=False, method="min").astype(int)

    runs.to_csv(output_dir / "runs.csv", index=False)
    rankings_long.to_csv(output_dir / "rankings_long.csv", index=False)
    pd.DataFrame(rankings).rename_axis("feature").to_csv(output_dir / "rankings_wide.csv")
    pairwise.to_csv(output_dir / "pairwise_within_condition.csv", index=False)
    within.to_csv(output_dir / "within_condition_stability.csv", index=False)
    consensus.rename_axis("feature").to_csv(output_dir / "consensus_rankings.csv")
    between.to_csv(output_dir / "between_condition_agreement.csv", index=False)

    summary = {
        "run_count": int(len(runs)),
        "runs_computed_this_invocation": computed,
        "configuration_count": int(runs["config_id"].nunique()),
        "seeds_per_configuration": int(len(seeds)),
        "background_pool": "purged training split",
        "evaluation_pool": "held-out purged test split",
        "reference_configuration": reference,
        "rare_event_definition": rare_definition.to_dict(),
        "outcome_demand_definition": outcome_definition.to_dict(),
        "runtime_seconds_total": float(runs["elapsed_seconds"].sum()),
        "runtime_seconds_mean": float(runs["elapsed_seconds"].mean()),
        "additivity_warnings_total": int((runs["additivity_status"] == "warning").sum()),
        "within_condition_overall": summarise_stability(pairwise),
        "kendalls_w_mean": float(within["kendalls_w"].mean()),
        "kendalls_w_min": float(within["kendalls_w"].min()),
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "results_schema.json").write_text(json.dumps({
        "schema_version": 2,
        "runs.csv": list(runs.columns),
        "rankings_long.csv": list(rankings_long.columns),
        "rankings_wide.csv": ["feature", "<one column per run_id>"],
        "pairwise_within_condition.csv": list(pairwise.columns),
        "within_condition_stability.csv": list(within.columns),
        "consensus_rankings.csv": ["feature", "<one column per config_id>"],
        "between_condition_agreement.csv": list(between.columns),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/training_real"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/shap_week9_grid"))
    parser.add_argument("--background-methods", default="uniform,time_stratified")
    parser.add_argument("--background-sizes", default="50,200")
    parser.add_argument("--evaluation-methods", default=None,
                        help="Comma-separated list. Defaults to --evaluation-method.")
    parser.add_argument("--evaluation-sizes", default=None,
                        help="Comma-separated list. Defaults to --evaluation-size.")
    parser.add_argument("--evaluation-method", default="time_stratified",
                        help="Single evaluation method (kept for backward compatibility).")
    parser.add_argument("--evaluation-size", type=int, default=100,
                        help="Single evaluation size (kept for backward compatibility).")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Use N consecutive seeds starting at --seed-start instead of --seeds.")
    parser.add_argument("--seed-start", type=int, default=2082)
    parser.add_argument("--reference", default=None,
                        help="Reference config_id for between-condition agreement.")
    args = parser.parse_args()

    evaluation_methods = (_csv_strings(args.evaluation_methods)
                          if args.evaluation_methods else [args.evaluation_method])
    evaluation_sizes = (_csv_ints(args.evaluation_sizes)
                        if args.evaluation_sizes else [args.evaluation_size])
    if args.n_seeds is not None:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    elif args.seeds:
        seeds = _csv_ints(args.seeds)
    else:
        seeds = [2082, 2083, 2084]

    run_experiment(
        args.data, args.config, args.artifacts, args.output,
        _csv_strings(args.background_methods), _csv_ints(args.background_sizes),
        evaluation_methods, evaluation_sizes, seeds, args.reference,
    )


if __name__ == "__main__":
    main()
