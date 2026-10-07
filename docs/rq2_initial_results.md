# RQ2: initial real-data results

Analysis completed on 7 October 2026. This is an exploratory reliability
check, not a new forecasting model selection or a proof of explanation
correctness. The original frozen model and supplied Week 10 SHAP archive
were not changed. No SHAP grid was recomputed.

## Data and protocol

- Exact supplied training CSV and frozen feature order: 26 columns.
- Frozen test MAE reproduced: **397.492181 MW**, on 30,799 rows.
- Full-test permutation: 26 individual features and six groups, ten repeats.
- Matched permutation: 300 reconstructed evaluation samples (five methods,
  two sizes, 30 seeds), compared with 120 SHAP configuration consensuses.
- Ablation: six groups plus `temp_max` alone and a full-feature control;
  three training seeds, 24 independent fits in total.
- Fixed selected hyperparameters and 678 boosting rounds; training rows only,
  no new early stopping or tuning. All reduced models are saved separately.
- Paired origin-day bootstrap: 2,000 draws per fitted model pair. Intervals
  condition on those models, not on the entire training procedure.

Rare events are origin-time public holidays OR temperature below/above the
training-derived 5th/95th percentiles (9.45 and 26.35). Test coverage is only
seven rare dates: three extreme-temperature dates and four public holidays,
with no overlap in this test period. Ordinary coverage is 100 dates.

## Permutation and ablation answer different questions

Positive values below mean an increase in MAE. Permutation alters inputs
to the unchanged frozen model; ablation removes inputs and retrains.

| Removed/permuted group | Frozen-model PFI, all rows (MW) | Ablation, all rows (MW) |
| --- | ---: | ---: |
| Recent demand | 255.03 | -0.15 |
| Calendar | 123.74 | 45.45 |
| Temperature | 51.06 | 44.03 |
| Seasonal demand | 49.96 | 8.13 |
| Irradiance | 39.88 | 0.21 |
| Rolling demand | 7.89 | -14.73 |

PFI values are means of ten permutations; ablation values are means of
three seed-matched control comparisons. These are not pooled confidence
intervals. Recent-demand features can strongly influence the frozen model
while other correlated features substitute for them after retraining.
The negative rolling-demand ablation delta is an observed improvement in
this comparison, not a justification for changing the frozen model after
inspecting its test set.

## Does temperature help more on rare days?

Removing the four-column temperature group increases MAE by **159.11 MW**
on rare rows and **35.97 MW** on ordinary rows, averaged over training seeds.
The mean difference is 123.14 MW. However, the per-model paired bootstrap
intervals for that difference are wide and all cross zero:

| Training seed | Rare-minus-ordinary delta (MW) | 95% origin-day bootstrap interval (MW) |
| --- | ---: | ---: |
| 2082 | 121.93 | [-71.44, 368.11] |
| 2083 | 129.90 | [-51.23, 355.05] |
| 2084 | 117.58 | [-67.23, 339.48] |

The point estimates are consistent with stronger utility on rare days,
but do **not establish** that difference in this limited period. Removing
`temp_max` alone gives much smaller, similar changes: 9.78 MW on rare rows
and 9.48 MW on ordinary rows. Do not attribute the whole group result to
`temp_max` alone. Both restricted and unrestricted temperature permutation
also show positive rare-row reliance, but neither is a causal intervention.

## SHAP versus matched permutation rankings

| Background method | Mean Spearman | Mean top-5 Jaccard | Mean top-10 Jaccard |
| --- | ---: | ---: | ---: |
| Uniform | 0.8341 | 0.6667 | 0.7727 |
| Time-stratified | 0.8336 | 0.6667 | 0.7727 |
| K-means | 0.8298 | 0.6667 | 0.7235 |

These are descriptive means over 40 configurations per background method.
Every top-five comparison shares four features, not five. The differences
in mean Spearman are small, and the conditions share data/seeds. This is
not an independence-based significance test or proof of a winning method.
The same PFI reference is used for all backgrounds with the same evaluation
method and size. SHAP attribution and permutation loss increase are different
estimands; neither is explanation ground truth.

## Runtime and reproducibility limits

The run took approximately **6.58 minutes** on the local environment.
Each fixed-budget fit took 4.86–6.38 seconds using four threads; this is
not the runtime of a full hyperparameter search.

The manifest records input hashes, versions, feature order and baseline
metrics. The original SHAP zip lacks model/data hashes, runtime versions
and sampled row IDs. Reconstructed samples follow the available shared
sampler and `seed + 100000`; exact identity to the original sampled rows
cannot be independently certified.

Unrestricted permutations can create implausible combinations and break
temporal/cross-group dependence. Duplicate seasonal lags are kept together
in group analyses, but individual PFI remains sensitive to correlation.
Only seven rare dates limit inference. Daily temperature broadcasts and
hourly irradiance are not native five-minute weather, and timestamp alignment
alone does not establish intraday availability of daily min/max observations.
RQ2 agreement does not resolve the grid's additivity warnings.

See [the methodology and reproduction instructions](rq2_methodology.md).
