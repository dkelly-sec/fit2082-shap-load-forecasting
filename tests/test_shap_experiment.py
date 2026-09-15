import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shap_experiment import ranking_stability, summarise_stability


def test_identical_rankings_are_perfectly_stable():
    ranking = pd.Series([3.0, 2.0, 1.0], index=["a", "b", "c"])
    metrics = ranking_stability(ranking, ranking, top_k=2)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["kendall"] == pytest.approx(1.0)
    assert metrics["top_k_overlap"] == pytest.approx(1.0)


def test_reversed_rankings_are_unstable():
    left = pd.Series([3.0, 2.0, 1.0], index=["a", "b", "c"])
    right = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    metrics = ranking_stability(left, right, top_k=1)
    assert metrics["spearman"] == pytest.approx(-1.0)
    assert metrics["kendall"] == pytest.approx(-1.0)
    assert metrics["top_k_overlap"] == 0.0


def test_feature_mismatch_fails():
    with pytest.raises(ValueError, match="same features"):
        ranking_stability(
            pd.Series([2.0, 1.0], index=["a", "b"]),
            pd.Series([2.0, 1.0], index=["a", "c"]),
        )


def test_summary_reports_means_and_minima():
    pairs = pd.DataFrame({
        "spearman": [1.0, 0.5], "kendall": [0.9, 0.4],
        "top_k_overlap": [1.0, 0.6],
    })
    summary = summarise_stability(pairs)
    assert summary["pair_count"] == 2
    assert summary["spearman_mean"] == pytest.approx(0.75)
    assert summary["kendall_min"] == pytest.approx(0.4)
    assert summary["top_k_overlap_mean"] == pytest.approx(0.8)
