"""
Reproduce one grid run exactly and show which evaluation rows cause its
additivity gap, so the cause can be identified from real data rather than
guessed. Defaults reproduce the seed-2087 run that warned at 109.55 MW.

Usage
-----
    python scripts/diagnose_additivity.py --data data/interim/merged_full_frozen.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from load_model import load_frozen_model  # noqa: E402
from sampling import construct_sample, define_outcome_demand_events, define_rare_events  # noqa: E402
from shap_pilot import compute_global_shap_ranking  # noqa: E402
from training import chronological_split, load_config, prepare_frame  # noqa: E402

EVALUATION_SEED_OFFSET = 100_000  # must match shap_experiment.py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/training_real"))
    parser.add_argument("--background-method", default="uniform")
    parser.add_argument("--background-size", type=int, default=50)
    parser.add_argument("--evaluation-method", default="uniform")
    parser.add_argument("--evaluation-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2087)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    bundle = load_frozen_model(args.artifacts)
    config = load_config(args.config)
    frame, _ = prepare_frame(args.data, config)
    splits = chronological_split(frame, config)
    rare = define_rare_events(splits["train"])
    outcome = define_outcome_demand_events(splits["train"])

    background = construct_sample(splits["train"], bundle.feature_names, args.background_size,
                                  args.seed, args.background_method, rare, outcome)
    evaluation = construct_sample(splits["test"], bundle.feature_names, args.evaluation_size,
                                  args.seed + EVALUATION_SEED_OFFSET, args.evaluation_method, rare, outcome)

    _, explainer, shap_values = compute_global_shap_ranking(bundle, background, evaluation)
    predictions = bundle.model.predict(evaluation)
    reconstructed = explainer.expected_value + shap_values.sum(axis=1)
    errors = np.abs(predictions - reconstructed)

    booster = bundle.model.booster_
    print(f"Trees stored in model: {booster.num_trees()} | best_iteration_: {bundle.model.best_iteration_}")
    print(f"Max error: {errors.max():.4f} MW | rows with error > 1 MW: {(errors > 1).sum()} of {len(errors)}")
    print(f"Mean |SHAP| total shift this could cause: <= {errors.sum() / len(errors):.4f} MW\n")

    worst = np.argsort(errors)[::-1][: args.top]
    report = evaluation.iloc[worst].copy()
    report.insert(0, "error_mw", errors[worst])
    report.insert(1, "prediction", predictions[worst])
    report.insert(2, "has_nan", evaluation.iloc[worst].isna().any(axis=1).to_numpy())
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(f"Top {args.top} offending rows:")
        print(report.round(3).to_string())

    # Compare offending rows' feature values against the typical (median) row
    # to spot anything unusual: extreme values, NaNs, out-of-range inputs.
    print("\nFeatures where the worst row is furthest from the evaluation median (in IQRs):")
    q1, q3, med = evaluation.quantile(0.25), evaluation.quantile(0.75), evaluation.median()
    iqr = (q3 - q1).replace(0, np.nan)
    deviation = ((evaluation.iloc[worst[0]] - med).abs() / iqr).sort_values(ascending=False)
    print(deviation.dropna().head(6).round(2).to_string())

    # Threshold check: if a feature value sits exactly on a tree split
    # threshold, LightGBM and SHAP's copy of the tree can route the row down
    # different branches. A tiny nudge to that one feature then changes the
    # native prediction by roughly the additivity error.
    print(f"\nThreshold check on the worst row (additivity error {errors[worst[0]]:.4f} MW):")
    row = evaluation.iloc[[worst[0]]].astype(float)
    base = bundle.model.predict(row)[0]
    found = False
    for col in row.columns:
        value = row[col].iloc[0]
        if np.isnan(value):
            continue
        for delta in (-1e-6, 1e-6):
            nudged = row.copy()
            nudged[col] = value + delta * max(1.0, abs(value))
            jump = abs(bundle.model.predict(nudged)[0] - base)
            if jump > 0.01:
                found = True
                print(f"  {col} = {value!r}: nudging by {delta:+.0e} (relative) changes "
                      f"the prediction by {jump:.4f} MW")
    if not found:
        print("  No feature sits on a split threshold; the cause is something else.")


if __name__ == "__main__":
    main()
