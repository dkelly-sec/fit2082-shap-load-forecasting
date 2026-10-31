"""Audit saved RQ2 tables and rebuild report summaries with the standard library.

No model import, fitting, prediction, SHAP calculation or external input is used.
Default mode checks committed summaries without modifying any file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                f"Missing/duplicate header: {path}")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows),
            f"Malformed CSV row: {path}")
    return rows


def unique(rows, fields):
    keys = [tuple(row[field] for field in fields) for row in rows]
    require(len(keys) == len(set(keys)), f"Duplicate keys: {fields}")


def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-10, rel_tol=1e-12)


def ranks(values):
    return [1 + sum(y > x for y in values) + (sum(y == x for y in values) - 1) / 2
            for x in values]


def top_features(values, features, count):
    # Match pandas.nlargest's first-occurrence tie convention in frozen order.
    return sorted(features, key=lambda f: -values[f])[:count]


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right)


def csv_text(rows):
    require(bool(rows), "Cannot write an empty summary")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def mean_agreement(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[f] for f in fields)].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(int(x) for x in item[0])):
        result.append({**dict(zip(fields, key)), "configurations": len(group),
                       "mean_spearman": statistics.fmean(float(r["spearman"]) for r in group),
                       "mean_top10_jaccard": statistics.fmean(float(r["jaccard_top10"]) for r in group)})
    return result


def verify_permutation_summaries(results):
    runs = records(results / "permutation_full_test_runs.csv")
    unique(runs, ["scope", "target", "repeat", "cohort"])
    groups = defaultdict(list)
    for row in runs:
        value = float(row["delta_mae_mw"])
        require(math.isfinite(value), "Non-finite full-test PFI")
        groups[row["scope"], row["target"], row["cohort"]].append(value)
    summary = records(results / "permutation_full_test_summary.csv")
    unique(summary, ["scope", "target", "cohort"])
    require(len(summary) == len(groups), "Missing PFI summary groups")
    for row in summary:
        values = groups[row["scope"], row["target"], row["cohort"]]
        require(len(values) == int(row["repeats"]) == 10, "Full-test repeat coverage differs")
        calculated = [statistics.fmean(values), statistics.stdev(values), min(values), max(values)]
        require(all(close(a, row[k]) for a, k in zip(calculated, ["mean", "std", "minimum", "maximum"])),
                "Full-test PFI summary differs from repeat records")
    runs = records(results / "permutation_matched_runs.csv")
    unique(runs, ["evaluation_method", "evaluation_size", "seed", "feature"])
    groups = defaultdict(list)
    for row in runs:
        value = float(row["delta_mae_mw"])
        require(math.isfinite(value), "Non-finite matched PFI")
        groups[row["evaluation_method"], row["evaluation_size"], row["feature"]].append(value)
    summary = records(results / "permutation_matched_summary.csv")
    require(len(summary) == len(groups), "Missing matched PFI summary groups")
    for row in summary:
        values = groups[row["evaluation_method"], row["evaluation_size"], row["feature"]]
        require(len(values) == int(row["n_samples"]) == 30, "Matched sample coverage differs")
        require(close(statistics.fmean(values), row["mean"]) and close(statistics.stdev(values), row["std"]),
                "Matched PFI summary differs from sample records")


def compute(results):
    provenance = json.loads((results.parent / "provenance.json").read_text(encoding="utf-8"))
    for name, entry in provenance["saved_tables"].items():
        file = results / name
        require(hashlib.sha256(file.read_bytes()).hexdigest() == entry["sha256"],
                f"Snapshot SHA-256 mismatch: {name}")
        require(len(records(file)) == entry["rows"], f"Snapshot row count mismatch: {name}")

    verify_permutation_summaries(results)

    agreement = records(results / "shap_permutation_agreement.csv")
    unique(agreement, ["config_id"])
    expected_grid = {(b, str(n), e, str(m)) for b in ["uniform", "kmeans", "time_stratified"]
                     for n in [25, 50, 100, 200]
                     for e in ["uniform", "time_stratified", "season_stratified",
                               "rare_event_stratified", "outcome_demand_stratified"]
                     for m in [100, 500]}
    observed = {tuple(r[f] for f in ["background_method", "background_size", "evaluation_method", "evaluation_size"])
                for r in agreement}
    require(observed == expected_grid and len(agreement) == 120, "Incomplete 120-condition grid")
    for row in agreement:
        require(all(math.isfinite(float(row[k])) for k in ["spearman", "kendall", "jaccard_top5", "jaccard_top10"]),
                "Non-finite ranking agreement")

    features = provenance["feature_names"]
    require(len(features) == len(set(features)) == 26, "Frozen feature order is incomplete")
    consensus = records(results / "shap_consensus_importance.csv")
    unique(consensus, ["feature"])
    require(set(r["feature"] for r in consensus) == set(features), "SHAP feature coverage differs")
    by_feature = {r["feature"]: r for r in consensus}
    permutation = records(results / "permutation_matched_summary.csv")
    unique(permutation, ["evaluation_method", "evaluation_size", "feature"])
    mismatch, patterns = [], defaultdict(int)
    for row in agreement:
        selected = [p for p in permutation if p["evaluation_method"] == row["evaluation_method"]
                    and p["evaluation_size"] == row["evaluation_size"]]
        require(set(p["feature"] for p in selected) == set(features) and len(selected) == 26,
                "PFI feature coverage differs")
        require(all(int(p["n_samples"]) == 30 for p in selected), "PFI consensus lacks 30 samples")
        pv = {p["feature"]: float(p["mean"]) for p in selected}
        sv = {f: float(by_feature[f][row["config_id"]]) for f in features}
        require(all(math.isfinite(x) for x in [*pv.values(), *sv.values()]), "Non-finite importance")
        rho = statistics.correlation(ranks([sv[f] for f in features]), ranks([pv[f] for f in features]))
        require(close(rho, row["spearman"]), f"Spearman differs: {row['config_id']}")
        for size in [5, 10]:
            require(close(jaccard(top_features(sv, features, size), top_features(pv, features, size)),
                          row[f"jaccard_top{size}"]), f"Top-{size} Jaccard differs")
        st, pt = top_features(sv, features, 5), top_features(pv, features, 5)
        only_s, only_p = sorted(set(st) - set(pt)), sorted(set(pt) - set(st))
        mismatch.append({"config_id": row["config_id"], "evaluation_method": row["evaluation_method"],
                         "evaluation_size": row["evaluation_size"], "shap_only": ";".join(only_s),
                         "permutation_only": ";".join(only_p), "shared_features": len(set(st) & set(pt))})
        patterns[(row["evaluation_method"], row["evaluation_size"], ";".join(only_s), ";".join(only_p))] += 1

    metrics = records(results / "ablation_metrics.csv")
    unique(metrics, ["variant", "seed", "cohort"])
    expected_variants = {"full_control", "temperature", "irradiance", "recent_demand", "seasonal_demand",
                         "rolling_demand", "calendar", "temp_max_only"}
    expected_metrics = {(v, str(s), c) for v in expected_variants for s in [2082, 2083, 2084]
                        for c in ["all", "rare", "ordinary", "extreme_temperature", "public_holiday"]}
    require({(r["variant"], r["seed"], r["cohort"]) for r in metrics} == expected_metrics,
            "Incomplete 8 variants x 3 seeds x 5 cohorts")
    controls = {(r["seed"], r["cohort"]): r for r in metrics if r["variant"] == "full_control"}
    groups = defaultdict(list)
    for row in metrics:
        control = controls[row["seed"], row["cohort"]]
        require(int(row["n"]) == provenance["cohort_rows"][row["cohort"]], "Cohort row count differs")
        require(all(math.isfinite(float(row[k])) for k in ["mae", "rmse", "mape_percent"]), "Non-finite metric")
        delta = float(row["mae"]) - float(control["mae"])
        require(close(delta, row["delta_mae_vs_paired_control_mw"]), "Unpaired MAE delta")
        groups[row["variant"], row["cohort"]].append({"delta_mae_mw": delta,
                   "delta_rmse_mw": float(row["rmse"]) - float(control["rmse"])})
    ablation = [{"variant": v, "cohort": c, "training_seeds": len(rows),
                 "mean_delta_mae_mw": statistics.fmean(r["delta_mae_mw"] for r in rows),
                 "mean_delta_rmse_mw": statistics.fmean(r["delta_rmse_mw"] for r in rows)}
                for (v, c), rows in sorted(groups.items())]
    intervals = records(results / "ablation_day_bootstrap.csv")
    unique(intervals, ["variant", "seed", "cohort"])
    require(len(intervals) == 144, "Expected 144 stored bootstrap summaries")
    require(all(float(r["ci_low_mw"]) <= float(r["ci_high_mw"]) and
                0 < int(r["bootstrap_valid"]) <= 2000 for r in intervals), "Invalid bootstrap interval")
    timings = records(results / "training_timings.csv")
    unique(timings, ["variant", "seed"])
    require(len(timings) == 24 and all(int(r["rounds"]) == 678 for r in timings), "Ablation fit budget differs")
    return {
        "agreement_by_background_size.csv": mean_agreement(agreement, ["background_size"]),
        "agreement_by_evaluation_size.csv": mean_agreement(agreement, ["evaluation_size"]),
        "agreement_by_both_sizes.csv": mean_agreement(agreement, ["background_size", "evaluation_size"]),
        "ablation_paired_summary.csv": ablation,
        "top5_membership_by_configuration.csv": mismatch,
        "top5_membership_patterns.csv": [dict(zip(["evaluation_method", "evaluation_size", "shap_only", "permutation_only"], key),
                                               configurations=n) for key, n in sorted(patterns.items())],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, help="Write derived tables to a NEW directory instead of checking committed summaries")
    args = parser.parse_args(argv)
    calculated = compute(args.results)
    if args.output:
        # Refuse overwrites, including the archived input directory.
        args.output.mkdir(parents=True, exist_ok=False)
        for name, rows in calculated.items():
            (args.output / name).write_text(csv_text(rows), encoding="utf-8", newline="")
    else:
        for name, rows in calculated.items():
            require((HERE / "summary" / name).read_text(encoding="utf-8") == csv_text(rows),
                    f"Committed derived table differs: {name}")
    print(json.dumps({"status": "passed", "configurations": 120, "paired_metric_rows": 120,
                      "fits": 24, "bootstrap_rows": 144, "derived_tables": len(calculated),
                      "model_refits": 0, "shap_runs": 0}))


if __name__ == "__main__":
    main()
