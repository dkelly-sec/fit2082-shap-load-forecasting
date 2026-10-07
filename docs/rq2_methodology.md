# RQ2 reliability checks

RQ1 measures repeatability and sampling sensitivity. RQ2 asks whether the
explanations agree with independently measured predictive reliance and with
the incremental utility of feature groups after retraining. Agreement is
supporting evidence, not an explanation ground truth or a causal claim.

## Reproduce

Use a Python environment with the project's requirements, plus `pytest` for
tests. The RQ2 code does not require a new SHAP run.

```bash
python src/reliability.py \
  --data data/interim/merged_full.csv \
  --config configs/training.json \
  --artifacts artifacts/training_real \
  --grid /path/to/shap_week10_grid.zip \
  --protocol configs/rq2.json \
  --output artifacts/rq2_initial

python -m pytest -q tests/test_reliability.py
```

The CSV must be the exact supplied, feature-engineered training input, not a
freshly generated reconstruction. `--preflight-only` verifies compatibility
and writes a manifest without permutations or fits. Use a new output path
for each invocation. The program refuses an existing path or a path within
the frozen artifacts directory. It does not implement resume.

Before computing results it verifies the exact 26-column order against the
fitted model, the saved chronological origin/target split metadata, the
archive's row counts, the training-derived rare/outcome thresholds and the
frozen test metrics. It records content hashes, library versions, the source
hash, and the Git base revision. Uncommitted implementation is identified
by the source hash, not by the base revision alone. All source data and
frozen artifacts are hashed again after computation.

## Protocol and feature groups

`configs/rq2.json` fixes this initial exploratory analysis before RQ2
outcomes are computed. It was designed after the RQ1 findings and is not a
prospectively preregistered confirmatory experiment.

| Group | Columns |
| --- | --- |
| Temperature | `temp_min`, `temp_max`, `temperature`, `irr_temperature` |
| Irradiance | `irr_electricity`, `irr_irradiance_direct`, `irr_irradiance_diffuse` |
| Recent demand | `lag_5min`, `lag_10min`, `lag_15min` |
| Seasonal demand | `lag_1day`, `lag_1week`, `demand_lag_288`, `demand_lag_2016` |
| Rolling demand | `roll_mean_1h`, `roll_std_1h`, `roll_mean_24h`, `roll_std_24h` |
| Calendar | `day_of_week`, `is_weekend`, `is_public_holiday`, `origin_hour_of_day`, `hour`, `minute`, `month`, `day_of_year` |

The groups form a disjoint partition of all 26 features. Exact duplicate
lag pairs remain together in the seasonal-demand group. `temp_max_only`
is a supplementary ablation, not an alternative group partition.

## Frozen-model permutation

For every individual feature and primary group, run 10 unrestricted donor
permutations on the full held-out test set. A grouped permutation uses the
same donor row for all columns in the group, preserving their internal
relationships, including exact duplicates. The original frozen model is
never fitted. Positive importance is the increase in MAE, in MW. Negative
values are retained, not clipped or converted to absolute values.

Summarise full-test, rare, ordinary, extreme-temperature and public-holiday
errors using cohort membership from the original, unperturbed rows. The
primary rare cohort is public holiday OR origin temperature outside the
training 5th/95th percentile cut-offs. The realised high-target-demand
cohort remains a separate retrospective evaluation design.

A supplementary joint temperature permutation uses donors only within the
original rare/ordinary cohort. This answers a different, conditional
perturbation question; it should not be silently substituted for the main
unrestricted result. Permutation SD describes donor randomisation on this
fixed dataset, not a population confidence interval.

## SHAP and permutation comparisons

For every evaluation method, size and seed in the archive, reconstruct the
sample using the shared `construct_sample()` function and the grid's
`seed + 100000` convention. Retain positional row IDs to select the correct
targets and verify returned feature values against source rows. Run one
feature-wise permutation on each reconstructed evaluation sample. Average
the signed MAE increases over the same seeds before ranking them.

Compare each background configuration's mean-|SHAP| consensus against the
matching evaluation method/size permutation vector. Background methods
therefore share the same permutation reference, rather than receiving
different random references. Save Spearman, Kendall and top-5/top-10
Jaccard. These are descriptive agreement statistics, not independent
replicate-based tests. SHAP attribution magnitude and permutation error
increase measure different quantities, even though both use MW.

The archive lacks original sampled-row IDs, input/model hashes and the
original runtime versions. Reconstruction follows the available shared
code but cannot prove the original samples/model byte-identical. This
provenance limitation is recorded, not hidden behind a compatibility check.

## Retrained group ablations

Fit each reduced-feature model and a full-feature control on the same
purged training rows. Use the frozen model's selected parameters and fixed
678-round budget, without early stopping, a new parameter search or any
test-set selection. Use three seeds, 2082–2084, with a seed-matched
full-feature control for each ablation. Save these new models separately.

Primary deltas compare a reduced model with its seed-matched full-feature
control. Additional deltas against the frozen model are labelled separately.
This matters because a seed change alone may change predictive performance.

Evaluate MAE, RMSE and MAPE on fixed test cohorts. For paired MAE deltas,
resample complete origin-date clusters 2,000 times. Compute percentile
intervals per fitted model pair and the rare-minus-ordinary delta contrast
using shared bootstrap draws. Replicates with an empty relevant cohort are
excluded and counted. Do not average interval endpoints and call that a
pooled confidence interval across training seeds.

These intervals are conditional on the fitted models. They do not include
training-data uncertainty, correct dependence spanning multiple days or
support broad extreme-event generalisation from very few rare dates.

## Interpretation limits

- PFI measures reliance of the fixed model; ablation measures substitutability
  after retraining. Neither establishes feature causality or SHAP correctness.
- Single-feature importance can be misleading with correlated/duplicate
  features. Grouping retains within-group dependence but does not repair all
  cross-group dependence or off-manifold perturbations.
- Identical `colsample_bytree` fractions imply different effective column
  counts after ablation. This and training randomness affect comparisons.
- The test set has already been evaluated and is reused for post-hoc
  inspection. Do not call this a fresh independent final test or retune the
  production/frozen forecasting model using these results.
- Daily BOM broadcasts and hourly irradiance are not genuine five-minute
  observations. Origin timestamp alignment does not prove daily min/max
  observations were available at that time.
- Agreement with PFI does not resolve the SHAP additivity warnings.

## Outputs

- `run_manifest.json`: exact protocol, versions, hashes, baseline and cohorts.
- `permutation_full_test_runs.csv` and `permutation_full_test_summary.csv`.
- `permutation_matched_runs.csv`, `permutation_matched_summary.csv` and
  `evaluation_samples.json`.
- `shap_permutation_agreement.csv`: one row per SHAP configuration.
- `ablation_metrics.csv`, `ablation_day_bootstrap.csv`, `training_timings.csv`
  and `ablation_predictions.csv`.
- `ablation_models/`: independent fitted controls and reduced models.
- `RQ2_Report_EN.txt`: English initial-analysis report and limitations.

Method sources: [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
and [LightGBM estimator documentation](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMRegressor.html).
