"""
Load the frozen LightGBM model and its feature list from a training
artifacts directory, for use in Week 7's TreeSHAP pilot.

This is deliberately a thin, read-only loader -- it does not retrain or
modify the model in any way. Per the project's frozen-model rule (see the
README's Week 5-6 section), only background/evaluation sample selection
may vary from here on; the model, features, hyperparameters, and split
boundaries are fixed.

Usage
-----
    python src/load_model.py --artifacts artifacts/training

    # or import directly:
    from load_model import load_frozen_model
    bundle = load_frozen_model(Path("artifacts/training"))
    bundle.model.predict(some_dataframe[bundle.feature_names])
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class FrozenModel:
    model: object              # the trained LGBMRegressor (sklearn-style)
    feature_names: list[str]   # exact ordered columns the model expects
    best_params: dict          # hyperparameters + strategy metadata
    metrics: dict               # validation/test metrics from training
    artifacts_dir: Path


def load_frozen_model(artifacts_dir: Path) -> FrozenModel:
    artifacts_dir = Path(artifacts_dir)

    model_path = artifacts_dir / "model.joblib"
    features_path = artifacts_dir / "feature_names.json"
    params_path = artifacts_dir / "best_params.json"
    metrics_path = artifacts_dir / "metrics.json"

    missing = [p for p in (model_path, features_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact file(s): {[str(p) for p in missing]}. "
            f"Expected them inside {artifacts_dir} -- run the Week 5-6 training "
            "pipeline first (scripts/train_model.py) if this directory doesn't exist yet."
        )

    model = joblib.load(model_path)
    feature_names = json.loads(features_path.read_text(encoding="utf-8"))

    if not hasattr(model, "predict"):
        raise ValueError(
            f"Loaded object from {model_path} has no .predict method -- "
            "is this actually a fitted LightGBM/sklearn-style model?"
        )
    if not isinstance(feature_names, list) or not all(isinstance(f, str) for f in feature_names):
        raise ValueError(f"{features_path} did not contain a plain list of column-name strings.")

    best_params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    return FrozenModel(
        model=model,
        feature_names=feature_names,
        best_params=best_params,
        metrics=metrics,
        artifacts_dir=artifacts_dir,
    )


def sanity_check(bundle: FrozenModel) -> None:
    """
    Cheap end-to-end check: build one dummy row of zeros with the model's
    exact expected columns, and confirm .predict() runs and returns a
    finite number. Catches column-order/type mismatches early, before any
    SHAP code is built on top of this loader.
    """
    dummy = pd.DataFrame([[0.0] * len(bundle.feature_names)], columns=bundle.feature_names)
    prediction = bundle.model.predict(dummy)
    value = float(prediction[0])
    if not (value == value):  # NaN check without importing numpy just for this
        raise ValueError("Model produced a NaN prediction on a trivial all-zero input row.")
    print(f"Sanity check passed: predict() on a dummy row returned {value:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/training"),
                         help="Path to the training artifacts directory (default: artifacts/training)")
    args = parser.parse_args()

    bundle = load_frozen_model(args.artifacts)

    print(f"Loaded model from {bundle.artifacts_dir / 'model.joblib'}")
    print(f"Feature count: {len(bundle.feature_names)}")
    print(f"Feature names: {bundle.feature_names}")
    if bundle.best_params:
        print(f"Best params / strategy: {bundle.best_params.get('strategy', '(not recorded)')}")
    if bundle.metrics:
        print(f"Recorded metrics: {json.dumps(bundle.metrics, indent=2)}")

    sanity_check(bundle)


if __name__ == "__main__":
    main()
