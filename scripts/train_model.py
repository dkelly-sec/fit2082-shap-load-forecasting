"""Command-line entry point for the Week 5–6 LightGBM pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "interim" / "merged_full.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "training.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "training")
    args = parser.parse_args()
    result = run_training(args.data, args.config, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
