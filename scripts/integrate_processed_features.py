"""Combine teammate-provided feature splits without trusting their split boundaries.

The supplied files contain target-time columns and contiguous, non-purged
train/validation/test boundaries. This script restores one chronological
origin-time table and removes every target-derived column. The training
pipeline then rebuilds the 24-hour target and purged splits itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        args.input_dir / "train_features.csv",
        args.input_dir / "val_features.csv",
        args.input_dir / "test_features.csv",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing supplied feature files: {missing}")

    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="raise")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    if combined["timestamp"].duplicated().any():
        raise ValueError("Supplied files contain duplicate forecast-origin timestamps.")
    expected_grid = pd.date_range(combined["timestamp"].iloc[0], combined["timestamp"].iloc[-1], freq="5min")
    if len(expected_grid) != len(combined) or not combined["timestamp"].equals(pd.Series(expected_grid)):
        raise ValueError("Supplied files do not form one complete five-minute origin-time grid.")

    target_columns = [column for column in combined.columns if column == "target_timestamp" or column.startswith("target_")]
    combined = combined.drop(columns=target_columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined):,} origin-time rows to {args.output}")
    print(f"Removed target-derived columns: {target_columns}")


if __name__ == "__main__":
    main()
