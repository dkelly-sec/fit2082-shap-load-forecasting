"""Read-only regression checks for the published RQ2 result snapshots."""

import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rq2.summarise_results import compute, csv_text, main, records, unique  # noqa: E402

RQ2 = ROOT / "rq2"


def test_committed_summaries_reproduce_without_models(capsys):
    main([])
    assert '"model_refits": 0' in capsys.readouterr().out


def test_sample_size_and_top5_regression_values():
    summaries = compute(RQ2 / "results")
    sizes = {(r["background_size"], r["evaluation_size"]): r
             for r in summaries["agreement_by_both_sizes.csv"]}
    assert len(sizes) == 8 and all(r["configurations"] == 15 for r in sizes.values())
    for size, rho in [("50", .8460626780626779), ("100", .847977207977208), ("200", .8481139601139599)]:
        assert sizes[size, "500"]["mean_spearman"] == pytest.approx(rho)
        assert sizes[size, "500"]["mean_top10_jaccard"] == pytest.approx(.7575757575757576)
    pairs = Counter((r["shap_only"], r["permutation_only"])
                    for r in summaries["top5_membership_by_configuration.csv"])
    assert pairs == {("day_of_year", "temp_max"): 96,
                     ("day_of_week", "lag_1week"): 12, ("day_of_year", "lag_1week"): 12}


def test_same_seed_rmse_and_mae_deltas():
    data = compute(RQ2 / "results")["ablation_paired_summary.csv"]
    rows = {r["variant"]: r for r in data if r["cohort"] == "all"}
    assert rows["temperature"]["mean_delta_mae_mw"] == pytest.approx(44.03403565220109)
    assert rows["temperature"]["mean_delta_rmse_mw"] == pytest.approx(76.6404975219579)
    assert all(r["training_seeds"] == 3 for r in rows.values())


def test_modified_evidence_is_rejected(tmp_path):
    shutil.copytree(RQ2 / "results", tmp_path / "results")
    shutil.copyfile(RQ2 / "provenance.json", tmp_path / "provenance.json")
    file = tmp_path / "results/ablation_metrics.csv"
    file.write_bytes(file.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        compute(tmp_path / "results")


def test_summary_rebuild_refuses_overwrite(tmp_path):
    output = tmp_path / "new_summary"
    main(["--output", str(output)])
    expected = compute(RQ2 / "results")
    for name, rows in expected.items():
        assert (output / name).read_text() == csv_text(rows)
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])


def test_duplicate_keys_and_malformed_csv_fail(tmp_path):
    with pytest.raises(ValueError, match="Duplicate keys"):
        unique([{"id": "same"}, {"id": "same"}], ["id"])
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1\n")
    with pytest.raises(ValueError, match="Malformed"):
        records(bad)
