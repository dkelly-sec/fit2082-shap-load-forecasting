"""RQ2: frozen-model permutation reliance and paired retraining ablations.

Never refit the frozen estimator. Read the supplied SHAP archive, reconstruct
evaluation samples with the shared sampler, and save every new model separately.
Positive permutation/ablation delta means an increase in prediction error.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import time
import zipfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn

from evaluation import regression_metrics
from load_model import load_frozen_model
from sampling import construct_sample, define_outcome_demand_events, define_rare_events, rare_event_mask
from training import _split_metadata, chronological_split, load_config, prepare_frame


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def archive_tables(path: Path):
    with zipfile.ZipFile(path) as archive:
        roots = [name.rsplit("/", 1)[0] for name in archive.namelist()
                 if name.endswith("/experiment_manifest.json")]
        if len(roots) != 1:
            raise ValueError("Expected exactly one experiment manifest in SHAP zip.")
        root = roots[0]
        manifest = json.loads(archive.read(f"{root}/experiment_manifest.json"))
        summary = json.loads(archive.read(f"{root}/experiment_summary.json"))
        runs = pd.read_csv(archive.open(f"{root}/runs.csv"))
        rankings = pd.read_csv(archive.open(f"{root}/rankings_long.csv"))
    if runs.run_id.duplicated().any() or rankings.duplicated(["run_id", "feature"]).any():
        raise ValueError("Duplicate run/feature identifiers in SHAP archive.")
    return manifest, summary, runs, rankings


def validate_groups(features: list[str], groups: dict[str, list[str]]) -> None:
    flattened = [feature for group in groups.values() for feature in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(features):
        raise ValueError("Primary groups must partition the exact frozen features.")


def strict_metrics(actual, prediction, epsilon: float) -> dict:
    if not np.isfinite(np.asarray(actual, dtype=float)).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite targets/predictions must not be silently omitted.")
    return regression_metrics(actual, prediction, epsilon)


def predict_ordered(model, values, features, threads: int) -> np.ndarray:
    """Keep frozen column names/order and fail on non-finite predictions."""
    frame = pd.DataFrame(values, columns=features)
    prediction = np.asarray(model.predict(frame, num_threads=threads), dtype=float)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError("Non-finite or incorrectly shaped permutation predictions.")
    return prediction


def cohort_masks(test: pd.DataFrame, rare_definition) -> dict[str, np.ndarray]:
    rare = rare_event_mask(test, rare_definition).to_numpy(dtype=bool)
    extreme = ((test.temperature <= rare_definition.temperature_low)
               | (test.temperature >= rare_definition.temperature_high)).to_numpy()
    holiday = test.is_public_holiday.astype(bool).to_numpy()
    return {"all": np.ones(len(test), dtype=bool), "rare": rare, "ordinary": ~rare,
            "extreme_temperature": extreme, "public_holiday": holiday}


def permute_group(values: np.ndarray, columns: list[int], rng,
                  strata: np.ndarray | None = None) -> np.ndarray:
    """Use one donor row for all grouped columns; optionally permute within cohorts."""
    result = values.copy()
    donor = np.arange(len(values))
    if strata is None:
        donor = rng.permutation(donor)
    else:
        for label in np.unique(strata):
            positions = np.flatnonzero(strata == label)
            donor[positions] = rng.permutation(positions)
    result[:, columns] = values[donor][:, columns]
    return result


def duplicate_features(frame: pd.DataFrame, features: list[str]) -> list[list[str]]:
    return [[left, right] for i, left in enumerate(features) for right in features[i + 1:]
            if np.array_equal(frame[left].to_numpy(), frame[right].to_numpy(), equal_nan=True)]


def validate_context(data: Path, config_path: Path, artifacts: Path, grid: Path, protocol: dict):
    config = load_config(config_path)
    bundle = load_frozen_model(artifacts)
    raw = pd.read_csv(data, usecols=[config["timestamp_column"]])
    timestamps = pd.to_datetime(raw.iloc[:, 0])
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Frozen input must be sorted and duplicate-free.")
    frame, features = prepare_frame(data, config)
    if features != bundle.feature_names or features != list(bundle.model.feature_name_):
        raise ValueError("Exact model feature order mismatch.")
    validate_groups(features, protocol["groups"])
    splits = chronological_split(frame, config)
    metadata_path = artifacts / "split_metadata.json"
    if not metadata_path.exists():
        raise ValueError("Exact frozen split_metadata.json is required for RQ2.")
    if _split_metadata(splits) != load_config(metadata_path):
        raise ValueError("Reconstructed split boundaries differ from frozen training.")
    manifest, summary, runs, rankings = archive_tables(grid)
    expected = {"feature_names": features, "data_rows": len(frame),
                "train_rows": len(splits["train"]), "test_rows": len(splits["test"]),
                "model_best_iteration": bundle.model.best_iteration_}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"SHAP archive mismatch: {key}")
    if len(runs) != summary["run_count"] or set(rankings.run_id) != set(runs.run_id):
        raise ValueError("SHAP runs/rankings coverage mismatch.")
    if not (rankings.groupby("run_id").size() == len(features)).all():
        raise ValueError("SHAP feature coverage mismatch.")
    if set(rankings.feature) != set(features) or not np.isfinite(rankings.mean_abs_shap).all():
        raise ValueError("Invalid archived SHAP importances.")
    if set(protocol["supplementary_ablations"].get("temp_max_only", [])) - set(features):
        raise ValueError("Unknown supplementary ablation feature.")
    rare = define_rare_events(splits["train"])
    if rare.to_dict() != summary["rare_event_definition"]:
        raise ValueError("Rare-event thresholds differ from the SHAP grid.")
    outcome = define_outcome_demand_events(splits["train"])
    if outcome.to_dict() != summary["outcome_demand_definition"]:
        raise ValueError("Outcome-demand thresholds differ from the SHAP grid.")
    test = splits["test"]
    prediction = bundle.model.predict(test[features], num_threads=protocol["prediction_threads"])
    baseline = strict_metrics(test.demand, prediction, config["mape_epsilon"])
    saved = bundle.metrics["test"]["lightgbm"]
    for key in ("mae", "rmse", "mape_percent"):
        if not np.isclose(baseline[key], saved[key], atol=1e-7, rtol=0):
            raise ValueError(f"Frozen test metric failed to reproduce: {key}.")
    if baseline["n"] != saved["n"]:
        raise ValueError("Frozen test observation count mismatch.")
    return bundle, config, splits, rare, outcome, runs, rankings, prediction, baseline


def full_test_permutation(bundle, test, masks, config, protocol, output):
    features = bundle.feature_names
    values = test[features].to_numpy(dtype=float)
    actual = test.demand.to_numpy()
    base = predict_ordered(bundle.model, values, features, protocol["prediction_threads"])
    loss = np.abs(actual - base)
    targets = {**{f"feature:{f}": [f] for f in features},
               **{f"group:{key}": group for key, group in protocol["groups"].items()}}
    records = []
    for index, (name, group) in enumerate(targets.items()):
        # Same random donor mapping for every feature/group in a repeat.
        for repeat in range(protocol["full_test_permutation_repeats"]):
            rng = np.random.default_rng(np.random.SeedSequence([protocol["random_seed"], repeat, 0]))
            corrupted = permute_group(values, [features.index(f) for f in group], rng)
            prediction = predict_ordered(bundle.model, corrupted, features, protocol["prediction_threads"])
            corrupted_loss = np.abs(actual - prediction)
            for cohort, mask in masks.items():
                if mask.any():
                    records.append({"scope": "full_test", "target": name, "repeat": repeat,
                                    "cohort": cohort, "n": int(mask.sum()),
                                    "delta_mae_mw": float((corrupted_loss - loss)[mask].mean())})
        print(f"Permutation full-test {index + 1}/{len(targets)}: {name}", flush=True)
    # Cohort-restricted temperature permutation is a sensitivity analysis:
    # donor temperatures do not cross the rare/ordinary boundary.
    for repeat in range(protocol["full_test_permutation_repeats"]):
        rng = np.random.default_rng(np.random.SeedSequence([protocol["random_seed"], repeat, 1]))
        corrupted = permute_group(values, [features.index(f) for f in protocol["groups"]["temperature"]],
                                 rng, masks["rare"])
        prediction = predict_ordered(bundle.model, corrupted, features, protocol["prediction_threads"])
        delta = np.abs(actual - prediction) - loss
        for cohort in ("rare", "ordinary"):
            mask = masks[cohort]
            records.append({"scope": "within_rare_ordinary", "target": "group:temperature",
                            "repeat": repeat, "cohort": cohort, "n": int(mask.sum()),
                            "delta_mae_mw": float(delta[mask].mean())})
    result = pd.DataFrame(records)
    result.to_csv(output / "permutation_full_test_runs.csv", index=False)
    summary = result.groupby(["scope", "target", "cohort"]).delta_mae_mw.agg(
        mean="mean", std="std", minimum="min", maximum="max", repeats="size").reset_index()
    summary.to_csv(output / "permutation_full_test_summary.csv", index=False)
    return summary


def evaluation_rows(test, features, method, size, seed, rare, outcome, offset):
    candidate = test.reset_index(drop=True).copy()
    candidate["__rq2_row_position"] = np.arange(len(candidate))
    sample = construct_sample(candidate, features + ["__rq2_row_position"], size,
                              seed + offset, method, rare, outcome)
    positions = sample.pop("__rq2_row_position").to_numpy(dtype=int)
    np.testing.assert_allclose(sample.to_numpy(dtype=float),
                               candidate.iloc[positions][features].to_numpy(dtype=float), equal_nan=True)
    return sample, positions


def rank_agreement(left: pd.Series, right: pd.Series) -> dict:
    right = right.reindex(left.index)
    if not np.isfinite(right.to_numpy(dtype=float)).all() or not np.isfinite(left.to_numpy(dtype=float)).all():
        raise ValueError("Rank comparisons require complete finite vectors.")
    rho = left.rank(method="average").corr(right.rank(method="average"))
    tau = left.corr(right, method="kendall")
    result = {"spearman": None if pd.isna(rho) else float(rho),
              "kendall": None if pd.isna(tau) else float(tau)}
    for k in (5, 10):
        a, b = set(left.nlargest(min(k, len(left))).index), set(right.nlargest(min(k, len(right))).index)
        result[f"jaccard_top{k}"] = len(a & b) / len(a | b)
    return result


def matched_permutation(bundle, test, rare, outcome, runs, rankings, config, protocol, output):
    features = bundle.feature_names
    records, sample_rows = [], []
    specs = runs[["evaluation_method", "evaluation_size", "seed"]].drop_duplicates()
    for index, spec in enumerate(specs.itertuples(index=False)):
        x, positions = evaluation_rows(test, features, spec.evaluation_method, int(spec.evaluation_size),
                                       int(spec.seed), rare, outcome,
                                       protocol["matched_evaluation_seed_offset"])
        y = test.iloc[positions].demand.to_numpy()
        values = x.to_numpy(dtype=float)
        base = predict_ordered(bundle.model, values, features, protocol["prediction_threads"])
        base_mae = float(np.abs(y - base).mean())
        rng = np.random.default_rng(np.random.SeedSequence([protocol["random_seed"], int(spec.seed),
                                                          int(spec.evaluation_size), 2]))
        # One permutation per seed-matched sample; variability combines sample
        # selection and donor randomness. It is NOT 30 repeats on one dataset.
        donor = rng.permutation(len(x))
        for column, feature in enumerate(features):
            corrupted = values.copy()
            corrupted[:, column] = values[donor, column]
            pred = predict_ordered(bundle.model, corrupted, features, protocol["prediction_threads"])
            records.append({"evaluation_method": spec.evaluation_method,
                            "evaluation_size": int(spec.evaluation_size), "seed": int(spec.seed),
                            "feature": feature, "baseline_mae_mw": base_mae,
                            "delta_mae_mw": float(np.abs(y - pred).mean() - base_mae)})
        sample_rows.append({"evaluation_method": spec.evaluation_method,
                            "evaluation_size": int(spec.evaluation_size), "seed": int(spec.seed),
                            "sample_positions": positions.tolist(),
                            "positions_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest()})
        if (index + 1) % 30 == 0:
            print(f"Matched permutation {index + 1}/{len(specs)} samples", flush=True)
    result = pd.DataFrame(records)
    result.to_csv(output / "permutation_matched_runs.csv", index=False)
    write_json(output / "evaluation_samples.json", sample_rows)
    pi = result.groupby(["evaluation_method", "evaluation_size", "feature"]).delta_mae_mw.agg(
        mean="mean", std="std", n_samples="size").reset_index()
    pi.to_csv(output / "permutation_matched_summary.csv", index=False)
    comparisons = []
    for cid, part in rankings.groupby("config_id"):
        meta = runs[runs.config_id == cid].iloc[0]
        selected = pi[(pi.evaluation_method == meta.evaluation_method)
                      & (pi.evaluation_size == meta.evaluation_size)]
        reference = selected.set_index("feature")["mean"].reindex(features)
        shap = part.groupby("feature").mean_abs_shap.mean().reindex(features)
        comparisons.append({"config_id": cid, "background_method": meta.background_method,
                            "background_size": int(meta.background_size),
                            "evaluation_method": meta.evaluation_method,
                            "evaluation_size": int(meta.evaluation_size),
                            **rank_agreement(shap, reference)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "shap_permutation_agreement.csv", index=False)
    return comparison


def paired_day_bootstrap(timestamps, error_delta, masks, repeats: int, seed: int) -> dict:
    """Resample whole origin-date clusters; intervals conditional on fitted models."""
    days = pd.to_datetime(timestamps).dt.normalize()
    labels, inverse = np.unique(days.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=repeats)
    draws, result = {}, {}
    for name, mask in masks.items():
        counts = np.bincount(inverse, weights=mask.astype(int), minlength=len(labels))
        sums = np.bincount(inverse, weights=np.asarray(error_delta) * mask, minlength=len(labels))
        numerator, denominator = weights @ sums, weights @ counts
        valid = denominator > 0
        values = numerator[valid] / denominator[valid]
        draws[name] = (numerator, denominator)
        if len(values):
            lo, hi = np.quantile(values, [0.025, 0.975])
            result[name] = {"delta_mae_mw": float(np.asarray(error_delta)[mask].mean()),
                            "ci_low_mw": float(lo), "ci_high_mw": float(hi),
                            "bootstrap_valid": int(valid.sum()),
                            "origin_day_clusters": int((counts > 0).sum())}
    rare_num, rare_den = draws["rare"]
    ordinary_num, ordinary_den = draws["ordinary"]
    valid = (rare_den > 0) & (ordinary_den > 0)
    values = rare_num[valid] / rare_den[valid] - ordinary_num[valid] / ordinary_den[valid]
    if len(values):
        lo, hi = np.quantile(values, [0.025, 0.975])
        result["rare_minus_ordinary"] = {
            "delta_mae_mw": result["rare"]["delta_mae_mw"] - result["ordinary"]["delta_mae_mw"],
            "ci_low_mw": float(lo), "ci_high_mw": float(hi), "bootstrap_valid": int(valid.sum()),
            "origin_day_clusters": int(len(labels))}
    return result


def fixed_ablation_params(model, iteration: int, seed: int, threads: int) -> dict:
    params = model.get_params(deep=False).copy()
    params.update(n_estimators=iteration, random_state=seed, n_jobs=threads)
    return params


def run_ablations(bundle, splits, masks, config, protocol, output, frozen_prediction):
    models_dir = output / "ablation_models"
    models_dir.mkdir()
    variants = {"full_control": [], **protocol["groups"], **protocol["supplementary_ablations"]}
    test = splits["test"]
    y = test.demand.to_numpy()
    frozen_error = np.abs(y - frozen_prediction)
    records, intervals = [], []
    timing_rows = []
    prediction_table = test[["timestamp", "target_timestamp", "demand"]].reset_index(drop=True).copy()
    prediction_table["frozen_prediction"] = frozen_prediction
    control_errors = {}
    for seed in protocol["training_seeds"]:
        for variant, removed in variants.items():
            selected = [f for f in bundle.feature_names if f not in removed]
            if not selected:
                raise ValueError("Ablation cannot remove every model feature.")
            params = fixed_ablation_params(bundle.model, bundle.model.best_iteration_, seed,
                                           protocol["prediction_threads"])
            model = lgb.LGBMRegressor(**params)
            started = time.perf_counter()
            # No eval_set, early stopping or test-driven parameter search.
            model.fit(splits["train"][selected], splits["train"].demand)
            elapsed = time.perf_counter() - started
            prediction = model.predict(test[selected], num_threads=protocol["prediction_threads"])
            strict_metrics(y, prediction, config["mape_epsilon"])
            model_dir = models_dir / f"{variant}__seed-{seed}"
            model_dir.mkdir()
            joblib.dump(model, model_dir / "model.joblib")
            write_json(model_dir / "feature_names.json", selected)
            write_json(model_dir / "parameters.json", params)
            prediction_table[f"{variant}__seed-{seed}"] = prediction
            error = np.abs(y - prediction)
            if variant == "full_control":
                control_errors[seed] = error
            paired_delta = error - control_errors[seed]
            ci = paired_day_bootstrap(test.timestamp.reset_index(drop=True), paired_delta, masks,
                                      protocol["bootstrap_repeats"], protocol["random_seed"])
            for cohort, mask in masks.items():
                if not mask.any():
                    continue
                metrics = strict_metrics(y[mask], prediction[mask], config["mape_epsilon"])
                records.append({"variant": variant, "seed": seed, "cohort": cohort,
                                "feature_count": len(selected), **metrics,
                                "delta_mae_vs_frozen_mw": float((error - frozen_error)[mask].mean()),
                                "delta_mae_vs_paired_control_mw": float(paired_delta[mask].mean())})
            intervals.extend({"variant": variant, "seed": seed, "cohort": cohort, **values}
                             for cohort, values in ci.items())
            timing_rows.append({"variant": variant, "seed": seed, "fit_seconds": elapsed,
                                "rounds": int(model.booster_.current_iteration()),
                                "feature_count": len(selected)})
            print(f"Ablation {variant}, seed {seed}: {elapsed:.1f}s fit", flush=True)
            pd.DataFrame(records).to_csv(output / "ablation_metrics.csv", index=False)
            pd.DataFrame(intervals).to_csv(output / "ablation_day_bootstrap.csv", index=False)
            pd.DataFrame(timing_rows).to_csv(output / "training_timings.csv", index=False)
    prediction_table.to_csv(output / "ablation_predictions.csv", index=False)
    return pd.DataFrame(records), pd.DataFrame(intervals), pd.DataFrame(timing_rows)


def run(data: Path, training_config: Path, artifacts: Path, grid: Path,
        protocol_path: Path, output: Path, preflight_only: bool = False) -> dict:
    if output.exists():
        raise ValueError("Output directory already exists; use a fresh path to prevent overwrites.")
    if artifacts.resolve() == output.resolve() or artifacts.resolve() in output.resolve().parents:
        raise ValueError("RQ2 outputs must not be placed inside frozen artifacts.")
    protocol = load_config(protocol_path)
    if protocol["full_test_permutation_repeats"] < 2 or protocol["bootstrap_repeats"] < 100:
        raise ValueError("Need >=2 permutations and >=100 bootstrap repeats.")
    seeds = protocol["training_seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be nonempty and unique.")
    guarded = [data, training_config, grid, *sorted(artifacts.glob("*"))]
    guarded = [p for p in guarded if p.is_file()]
    hashes = {str(p.resolve()): sha256(p) for p in guarded}
    bundle, config, splits, rare, outcome, runs, rankings, prediction, baseline = validate_context(
        data, training_config, artifacts, grid, protocol)
    masks = cohort_masks(splits["test"], rare)
    if not masks["rare"].any() or not masks["ordinary"].any():
        raise ValueError("Both rare and ordinary test cohorts are required.")
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "lightgbm": lgb.__version__, "scikit_learn": sklearn.__version__,
                "scipy": scipy.__version__, "joblib": joblib.__version__}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    metadata = {"protocol": protocol, "protocol_sha256": sha256(protocol_path),
                "source_sha256": sha256(Path(__file__)), "git_head": commit,
                "versions": versions, "input_sha256": hashes, "frozen_test_metrics": baseline,
                "splits": _split_metadata(splits), "feature_names": bundle.feature_names,
                "duplicate_features": duplicate_features(splits["train"], bundle.feature_names),
                "rare_event_definition": rare.to_dict(), "outcome_definition": outcome.to_dict(),
                "cohort_rows": {key: int(mask.sum()) for key, mask in masks.items()},
                "cohort_origin_days": {key: int(splits["test"].loc[mask, "timestamp"].dt.normalize().nunique())
                                       for key, mask in masks.items()},
                "archive_limitations": "SHAP archive lacks model/data content hashes and row IDs. Sampling reconstructed from current shared code and seed offset; original sampled rows cannot be independently proven identical.",
                "interpretation": "Exploratory post-hoc reliability checks. Existing test set is reused for inspection, not a new independent final evaluation. PFI/SHAP measure different estimands; neither is ground truth."}
    output.mkdir(parents=True)
    write_json(output / "run_manifest.json", metadata)
    print("Preflight passed", json.dumps({"baseline": baseline, "cohort_rows": metadata["cohort_rows"]}), flush=True)
    if preflight_only:
        return metadata
    started = time.perf_counter()
    pi = full_test_permutation(bundle, splits["test"], masks, config, protocol, output)
    agreement = matched_permutation(bundle, splits["test"], rare, outcome, runs, rankings,
                                    config, protocol, output)
    ablation, intervals, timings = run_ablations(bundle, splits, masks, config, protocol, output, prediction)
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise AssertionError(f"An immutable input changed: {path}")
    metadata["inputs_unchanged"] = True
    metadata["wall_seconds"] = time.perf_counter() - started
    metadata["status"] = "completed"
    write_json(output / "run_manifest.json", metadata)
    write_report(output, metadata, pi, agreement, ablation, intervals, timings)
    return metadata


def write_report(output, metadata, pi, agreement, ablation, intervals, timings):
    lines = ["RQ2 Reliability Checks: Initial Real-Data Analysis", "",
             "1. Scope and reproducibility",
             f"Frozen test MAE: {metadata['frozen_test_metrics']['mae']:.6f} MW.",
             "Exact feature order, split boundaries and stored baseline metrics verified before inspection.",
             "All input/source-archive/frozen-artifact hashes unchanged after computation.",
             f"Runtime: {metadata['wall_seconds']/60:.2f} minutes. Versions and hashes: run_manifest.json.",
             "No SHAP grid was rerun. No original model was refitted or overwritten.", "",
             "2. Protocol",
             "Primary groups partition all 26 features: temperature, irradiance, recent demand, seasonal demand, rolling demand and calendar.",
             "Exact duplicate lag pairs are kept together in the seasonal-demand group.",
             "Full-test PFI uses joint donor rows within a group and unrestricted permutation across all test rows.",
             f"{metadata['protocol']['full_test_permutation_repeats']} repeats; report the mean and random-permutation SD, not a population confidence interval.",
             "A supplementary temperature permutation restricts donors to the original rare/ordinary cohort.",
             "Rare/ordinary membership always comes from unmodified origin-time data, not corrupted features or target demand.",
             "Matched PFI reconstructs each evaluation method/size/seed with the shared sampler and +100000 evaluation seed offset.",
             "One donor permutation per sampled dataset; the 30-seed variability combines sample selection and donor randomness.",
             "PFI is averaged before ranking and compared to the corresponding mean-|SHAP| consensus; signed PFI is not clipped.",
             "SHAP values and MAE increases are different quantities even though both have MW units.",
             "Ablations fit training rows only, reuse selected hyperparameters and 678 rounds, and do not retune on validation/test.",
             "Three training seeds have seed-matched full-feature controls; primary deltas compare with these controls, not a different-seed frozen model.",
             "Bootstrap resamples whole origin-date clusters (2000 draws), retaining paired model errors.",
             "The intervals are conditional on each fitted model pair; they do not include training-data uncertainty or longer-than-day dependence.",
             "This protocol was set before RQ2 computations but after RQ1 observations. It is exploratory, not preregistered confirmation.", "",
             "3. Cohort coverage"]
    for key, rows in metadata["cohort_rows"].items():
        lines.append(f"{key}: {rows} rows across {metadata['cohort_origin_days'][key]} origin dates.")
    overlap = metadata["cohort_rows"]["public_holiday"] + metadata["cohort_rows"]["extreme_temperature"] - metadata["cohort_rows"]["rare"]
    lines += [f"Holiday/extreme-temperature overlap: {overlap} rows. These masks can overlap in general; neither is an independent replicate.", "",
              "4. Full-test grouped permutation (mean increase in MAE, MW)"]
    group_pi = pi[(pi.scope == "full_test") & pi.target.str.startswith("group:")]
    lines.append(group_pi.pivot(index="target", columns="cohort", values="mean").round(4).to_string())
    lines += ["", "5. SHAP versus matched permutation ranking (descriptive mean over configurations)",
              agreement.groupby("background_method")[["spearman", "kendall", "jaccard_top5", "jaccard_top10"]].mean().round(4).to_string(),
              "This pooling is descriptive. Background methods are compared against the SAME PFI vector for each evaluation method/size.",
              "Configurations share data/seeds; no independence-based p-value is computed.", "",
              "6. Ablation outcomes (mean across three training seeds)",
              ablation.groupby(["variant", "cohort"])[["mae", "delta_mae_vs_paired_control_mw"]].mean().round(4).to_string(),
              "", "7. Temperature contrast: rare minus ordinary increase in MAE",
              intervals[(intervals.variant == "temperature") & (intervals.cohort == "rare_minus_ordinary")].round(4).to_string(index=False),
              "A positive point contrast is directionally consistent with greater rare-event utility. Intervals spanning zero do not establish that difference.",
              "Ablation evaluates substitutability after refitting. PFI evaluates reliance of the fixed model. Neither establishes causality or SHAP correctness.",
              "", "8. Retraining timings", timings.groupby("variant").fit_seconds.agg(["mean", "min", "max"]).round(3).to_string(),
              "Times refer to fixed 678-round fits, not a full hyperparameter search. Hardware/thread configuration matters.", "",
              "9. Limitations and reporting cautions",
              metadata["archive_limitations"],
              "Unrestricted row permutations break temporal and cross-feature dependence and can create off-manifold rows. Grouping protects within-group dependence only.",
              "Daily temperature broadcasts/hourly irradiance are not genuine 5-minute weather. Origin alignment does not by itself prove daily min/max was available intraday.",
              "Group ablations change effective colsample_bytree counts. Seed-matched controls help, but training randomness and feature substitution remain.",
              "The reused test period and small number of rare dates limit generalisation. No hold-out claim is made for these post-hoc analyses.",
              "Additivity warnings in the supplied SHAP grid remain unresolved by RQ2 agreement. Agreement with PFI is not proof they are harmless.",
              "Reference SHAP is not ground truth. Do not select a background method just because it produces a preferred narrative.",
              "", "10. Method sources",
              "https://scikit-learn.org/stable/modules/permutation_importance.html",
              "https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRegressor.html"]
    (output / "RQ2_Report_EN.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/training_real"))
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/rq2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.data, args.config, args.artifacts, args.grid, args.protocol, args.output, args.preflight_only)


if __name__ == "__main__":
    main()
