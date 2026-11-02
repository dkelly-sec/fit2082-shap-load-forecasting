# RQ2: permutation importance and retrained ablation

This folder supplies the code and saved output tables behind Sections 3.5,
4.2 and the permutation-agreement breakdown in Section 4.3 of the report.
The real-data analysis ran on **7 October 2026**. Packaging these files does
not rerun SHAP, refit models, retune parameters or alter the final report.

## Quick check: no model, raw data or third-party packages needed

From the repository root, with Python 3.11 or newer:

```bash
python rq2/summarise_results.py
```

This read-only command checks the archived table hashes, record counts,
120-condition coverage, signed-PFI ranking comparisons, same-seed ablation
deltas and the committed summary tables. It does not import LightGBM or
load a model. To rebuild the summaries into a **new** directory instead:

```bash
python rq2/summarise_results.py --output artifacts/rq2_summary_rebuilt
```

An existing output directory is rejected. Original results are never overwritten.
`summary/` contains the resulting background/evaluation-size averages,
paired MAE/RMSE changes, and the configuration-level top-five differences.
The size averages are descriptive, not independent replications or equivalence tests.

## Files and their roles

- `reliability.py`: the RQ2 implementation archived in local commit `b294bf5`. Its
  `full_test_permutation()` and `matched_permutation()` functions calculate
  frozen-model PFI; `run_ablations()` performs paired removal/retraining.
  `paired_day_bootstrap()` computes uncertainty conditional on each fitted pair.
- `run.py` / `python -m rq2`: launcher that sets repository imports and the
  default `protocol.json` path. No statistical method changes.
- `protocol.json`: feature-group definitions, random seed, ten full-test
  permutation repeats, 2,000 bootstrap draws and three training seeds.
- `provenance.json`: portable run metadata, original external-input hashes,
  runtime versions, exact feature order and per-table hashes. It excludes
  the original machine's absolute paths. The embedded post-run audit describes
  the historical verification, not a new retraining run. The packaged source
  hash differs from the historical run/audit hashes, even with LF line endings;
  this package does not claim an exact historical executable snapshot. The
  original output tables are byte-identical and checked independently.
- `summarise_results.py`: standard-library checks and deterministic report tables.
- `tests/`: analysis invariants plus snapshot/summary regression tests.

### Saved evidence in `results/`

These eight RQ2 CSVs are byte-for-byte copies of the completed outputs:

| File | Rows | Meaning |
| --- | ---: | --- |
| `permutation_full_test_runs.csv` | 1,620 | Feature/group, repeat and fixed cohort; includes within-cohort temperature sensitivity |
| `permutation_full_test_summary.csv` | 162 | Mean, SD, range and repeats for each perturbation/cohort |
| `permutation_matched_runs.csv` | 7,800 | 26 feature permutations on each of 300 reconstructed samples |
| `permutation_matched_summary.csv` | 260 | Mean signed PFI over 30 sampling seeds for each evaluation design and feature |
| `shap_permutation_agreement.csv` | 120 | One SHAP/PFI consensus-ranking comparison per grid configuration |
| `ablation_metrics.csv` | 120 | Eight variants x three training seeds x five cohorts |
| `ablation_day_bootstrap.csv` | 144 | Eight variants x three seeds x six cohort/contrast summaries |
| `training_timings.csv` | 24 | One record per fitted variant/seed |

`shap_consensus_importance.csv` is an additional byte-for-byte copy of
`consensus_rankings.csv` from the supplied `shap_week10_grid.zip`. Its 26
rows and 120 configuration columns make the top-five membership checks
reproducible without publishing or recomputing the full SHAP grid.

CSV hashes protect the original bytes, including line endings. The local
`.gitattributes` prevents Git from changing those evidence files on Windows.

## Full analysis reproduction (optional and computational)

Reading/validating the saved tables above is sufficient to inspect the reported
numbers. The following commands are for explicitly reproducing the original
analysis; **they are not required to inspect this handoff**.

```bash
python -m pip install -r rq2/requirements.txt
```

Place the original, separately shared inputs locally:

1. Exact integrated 26-feature training input CSV. Its SHA-256 is
   `e48f9870c9924a80556787fb82994569dcb23391605d23e9d7da7634442349e4`.
   The original local filename was `merged_full.csv`; a file called
   `merged_full_frozen.csv` is suitable only if its content hash matches.
   A newly regenerated CSV can contain different rows/features.
2. `artifacts/training_real/`: the exact frozen `model.joblib`,
   `feature_names.json`, `best_params.json`, `metrics.json`, and
   **`split_metadata.json`**. Keep all supplied frozen artifacts together.
3. The original `shap_week10_grid.zip` (120 configurations x 30 seeds).
   The archive hash is recorded in `provenance.json`.
4. The matching `configs/training.json` from this repository.

Example commands below work in PowerShell and POSIX shells as single lines.
Substitute your local input and archive paths. Use a new output directory each time.

Validation only (reconstructs the input/splits and predicts the saved baseline,
but does not permute, retrain or compute SHAP):

```bash
python rq2/run.py --data data/interim/merged_full_frozen.csv --config configs/training.json --artifacts artifacts/training_real --grid shap_week10_grid.zip --output artifacts/rq2_preflight --preflight-only
```

Full reproduction (runs permutations and 24 LightGBM fits, but **no SHAP grid**):

```bash
python rq2/run.py --data data/interim/merged_full_frozen.csv --config configs/training.json --artifacts artifacts/training_real --grid shap_week10_grid.zip --output artifacts/rq2_reproduction
```

The launcher uses `rq2/protocol.json` unless `--protocol` is supplied. The
analysis refuses an existing output directory, guards the frozen artifact
directory, validates baseline metrics/features/splits and checks input hashes
after a full run. It saves new ablation models separately.

Run tests from the repository root:

```bash
python -m pytest rq2/tests -q
python -m pytest tests rq2/tests -q
```

`requirements.txt` specifies compatible dependencies; exact versions used in
the completed analysis appear in `provenance.json`. Different numerical
libraries/platforms can change freshly fitted models. The committed CSVs,
not a fresh run on changed inputs, are the evidence for the existing report.

## Interpretation and limitations

- The forecasting horizon remains 24 hours. The model and 26-column order
  stay frozen. Full-test PFI measures reliance of that model, while retrained
  ablation measures incremental utility under a fixed training procedure.
- The six groups partition all features. The two duplicate seasonal-lag pairs
  remain together. Ablations include six group removals, `temp_max` alone,
  and a full control at seeds 2082/2083/2084, each using 678 rounds.
- Positive deltas mean higher error. Ablation deltas use the **same-seed full
  control**, not a separately seeded frozen baseline. Negative values remain.
- Rare events use public holidays OR training-set 5th/95th temperature
  cut-offs. The test period includes only seven rare origin dates.
- The rare-minus-ordinary temperature contrast is exploratory. Each per-fit
  day-bootstrap interval includes zero; these are not pooled training intervals.
- Group ablation is **not** the unperformed configuration-specific
  top-five/bottom-five ablation. Do not claim it validates a minimum reliable
  SHAP configuration or proves causal importance.
- Original SHAP row identifiers/model-data hashes were unavailable. Samples
  follow the shared sampler and +100000 evaluation-seed offset. Compatibility
  checks cannot certify original sampled-row identity, and PFI agreement does
  not independently resolve SHAP additivity warnings.
- Daily weather alignment does not establish intraday availability of daily
  temperature summaries. No new claim of operational information availability
  follows from this packaging.

No raw input CSV, API token, binary model, per-row demand/prediction file,
sample-row-ID archive, presentation or report modification is part of this package.
