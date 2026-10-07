# FIT2082 Data Pipeline

Data acquisition, cleaning, model training, and explainability pipeline for
*Sampling Sensitivity of Global SHAP Feature Rankings in Short-Term
Electricity Demand Forecasting*.

Produces a clean, chronologically-split dataset combining AEMO Victorian
electricity demand (native 5-minute resolution, per supervisor direction),
BOM daily temperature, renewables.ninja hourly solar irradiance, and
derived calendar features — trains and validates a LightGBM forecaster
against a seasonal-naive baseline — wires up a TreeSHAP explainer verified
mathematically — and runs the full repeated-sampling SHAP grid that
measures how stable global feature rankings are across sampling designs.

## Project status (running log, most recent first)

- **Week 10:** Grid extended on supervisor advice by adding background
  sizes 25 and 100, after first timing one seed of each new
  configuration: 120 configurations × 30 seeds = **3,600 runs** (about
  3.7 more hours of SHAP compute). New `src/analyse_stability.py`
  regenerates every RQ1/RQ3 table and figure from the grid output,
  including Kruskal–Wallis/Dunn tests, paired size comparisons and
  bootstrap confidence intervals. Findings: background choices drive how
  repeatable the whole ranking is; evaluation method drives which features
  reach the top. Rare-event evaluation changes the top 5 in 27.6% of runs
  (against 1–2% for uniform, time- and season-stratified), because
  `temp_max` rises from 8th to 6th under rare conditions. For the top 5,
  a 50-row background matches 200 at about a quarter of the cost; the top
  10 needs 100. Final report drafting under way; RQ2 (reliability) in
  progress.
- **Week 9:** Full RQ1 sampling grid completed: 60 configurations
  (3 background methods × 2 background sizes × 5 evaluation methods ×
  2 evaluation sizes) × 30 seeds = **1,800 SHAP runs** against the frozen
  model, in about 6.6 hours of SHAP compute on a laptop (no HPC needed).
  `src/shap_experiment.py` extended to run the whole grid in one command,
  to report within-condition stability separately from between-condition
  agreement (the Week 8 pilot pooled the two), to add Kendall's W and
  top-5/top-10 Jaccard, and to resume after interruption.
  `season_stratified` evaluation sampling added. k-means sampling fixed
  for NaN features. The real cause of additivity warnings was identified
  (feature values lying exactly on split thresholds), replacing the
  incorrect Week 7 explanation. Pooled within-condition stability is high
  (mean Spearman 0.987; Kendall's W ≥ 0.973 for every configuration), but
  top-5 feature membership can still shift between seeds (minimum top-5
  Jaccard 0.43). Per-configuration and between-condition analysis is
  Week 10's work.
- **Week 8:** Controlled sample construction implemented in
  `src/sampling.py`: uniform random, k-means representatives,
  month/time-of-day stratified, and rare-event-stratified sampling.
  Background samples now come only from the purged training split and
  evaluation samples only from the held-out purged test split. Rare-event
  thresholds are pre-registered from training data. Primary rare events
  are public holidays or origin-time temperature outside the training
  5th-95th percentile range. Realised high target demand is kept as a
  separate, explicitly retrospective evaluation cohort. A 12-run real-data
  pilot grid validated the design and runtime.
- **Week 7:** TreeSHAP pilot wired up (`src/shap_pilot.py`) against the
  frozen model, using `interventional` perturbation mode. Verified via a
  mathematical additivity check, not just eyeballed. Issues caught and
  fixed in the process: SHAP's silent 100-row background cap, and a
  frozen-model provenance mismatch (see below). The additivity gaps seen
  in Week 7 were originally misattributed; the real cause was found in
  Week 9.
- **Weeks 5-6:** LightGBM forecaster trained and tuned via time-series CV
  with a purged chronological split, validated against a seasonal-naive
  baseline (~24.1% lower test MAE). Model, features, hyperparameters, and
  split boundaries are now **frozen** for Week 7 onward.
- **Week 4:** Full data pipeline complete — AEMO demand (native 5-min),
  BOM temperature, renewables.ninja irradiance, calendar features,
  chronological train/val/test split. EDA found no anomalies requiring
  `EXCLUDED_PERIODS`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Renewables.ninja API token

Sign up for a free account at <https://www.renewables.ninja/register>,
find your API token on your account page, then set it as an environment
variable — **never commit it to git**:

```bash
# Windows PowerShell
$env:RENEWABLES_NINJA_TOKEN = "your_token_here"

# Mac/Linux
export RENEWABLES_NINJA_TOKEN="your_token_here"
```

## Configure

Edit `src/config.py`:

- `START_DATE` / `END_DATE` — your 2-year window (currently 2024-01-01 to 2025-12-31)
- `REGION` — NEM region of interest (currently VIC1)
- `LATITUDE` / `LONGITUDE` — location for temperature and irradiance (currently Melbourne, Olympic Park)
- `EXCLUDED_PERIODS` — any anomaly windows identified during EDA (currently none — see Week 4 status above)

## Run: Data pipeline (Week 4)

### 1. AEMO demand data (native 5-minute resolution)

```bash
python src/fetch_aemo.py
```

Uses [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS) to pull
`DISPATCHREGIONSUM` and keeps its native 5-minute resolution — no
resampling. Drops AEMO's intervention-pricing-run duplicate rows (keeps
only `INTERVENTION == 0`, the physical dispatch run).

### 2. BOM temperature data (daily)

BOM's free download only offers daily temperature — see the docstring in
`src/load_bom.py` for the exact download steps (Climate Data Online, min
and max temperature, per year or all years, station 086338).

Place the downloaded folders (e.g. `IDCJAC0011_086338_2024/`,
`IDCJAC0010_086338_2024/`) directly in `data/raw/`, then:

```bash
python src/load_bom.py
```

### 3. Renewables.ninja irradiance data (hourly)

```bash
python src/fetch_irradiance.py
```

Pulls hourly solar irradiance/PV simulation data via the renewables.ninja
API, one calendar year at a time (free-tier rate limits). Uses the
`merra2` dataset — the higher-resolution `sarah` dataset does **not**
cover Australia, only Europe/Africa/Middle East.

### 4. Merge, add features, split

```bash
python src/clean_merge.py
```

This:
- Reindexes demand to a complete 5-minute grid and interpolates gaps up to
  2 hours (longer gaps left as NaN and reported)
- Adds calendar features (`day_of_week`, `is_weekend`, `is_public_holiday`,
  `season`)
- Broadcasts each day's temperature and each hour's irradiance across the
  relevant 5-minute demand rows
- Applies any `EXCLUDED_PERIODS`
- Writes a chronological 70/15/15 train/val/test split plus
  `split_manifest.json`

Outputs:

```
data/interim/merged_full.csv
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/split_manifest.json
```

### Data pipeline tests

```bash
python tests/test_fetch_aemo_native.py
python tests/test_merge_multi_resolution.py
```

Cover region-filtering, intervention-duplicate handling, native-resolution
preservation, and the two-tier broadcast merge (daily temperature, hourly
irradiance, both onto 5-minute demand) — synthetic data, no network needed.

## Run: LightGBM training baseline (Weeks 5-6)

```bash
python scripts/train_model.py \
  --data data/interim/merged_full_frozen.csv \
  --config configs/training.json \
  --output artifacts/training_real
```

**Use `merged_full_frozen.csv`, not `merged_full.csv`.** The frozen model
was trained on `merged_full_frozen.csv` (26 feature columns). A freshly
regenerated `merged_full.csv` from `clean_merge.py` alone has only 16,
because it skips the legacy feature-engineering step that produced the
other 10 — see "Frozen-model provenance mismatch" below.

**Note on output folder naming:** `artifacts/training_real` is the
authoritative, frozen model referenced throughout this document (Test
LightGBM MAE 397.49 MW, ~24.1% improvement over baseline). An earlier,
independently-regenerated model briefly existed under `artifacts/training`
during Week 7. `artifacts/training` should not be treated as
authoritative; use `artifacts/training_real` for all Week 7+ work.

### Forecast-row semantics

The configured horizon is 288 five-minute intervals (24 hours). Every
training row is timestamped by its **forecast origin** `t`:

- `timestamp` = forecast origin `t`
- `target_timestamp` = `t + 24h`
- `demand` = target demand `y_(t+24h)`
- `demand_at_origin` = `y_t`, and is the 24-hour seasonal-naive prediction
- `demand_lag_288` = `y_(t-24h)` (origin-relative)
- `demand_lag_2016` = `y_(t-7d)` (origin-relative)

The structural boundary loss is 2,304 rows: 2,016 required for the longest
origin-relative lag, plus 288 trailing rows required to construct the
future target.

The pipeline:
- requires sorted, duplicate-free timestamps and writes `data_quality.json`
- uses a 70/15/15 chronological split with a 288-row purge between train
  and validation, and between validation and test
- asserts both origin-time ordering and that earlier-split target
  timestamps end before the next split's origins begin
- compares LightGBM against the 24-hour seasonal-naive baseline `y_t`
- selects a small, fixed parameter grid using validation MAE and LightGBM
  early stopping; the test set is evaluated only after selection
- keeps the train-fitted, validation-selected model for Week 7 TreeSHAP

### Weather availability

The current configuration uses only weather observations attached to the
forecast-origin row (`weather_feature_timing: origin`). It does **not**
treat actual observations at `t+24h` as information available at origin —
this was a deliberate methodological choice to avoid treating future
weather as known, which real day-ahead forecasting can't assume.

**Caveat (found in Week 10):** daily temperature values are copied onto
every 5-minute row of their day, and a BOM daily maximum covers 9am to
9am the next morning. An origin early in a day therefore sees that day's
maximum temperature before it occurs: a small same-day look-ahead. The
model is frozen, so this is documented as a limitation rather than fixed.
It may inflate the measured importance of the daily temperature features,
but does not affect the stability comparisons, since every configuration
explains the same model.

### Training outputs

Outputs under `artifacts/training_real/` are:

```text
model.joblib                 reloadable sklearn-style model
model.txt                    LightGBM native model
feature_names.json           exact ordered model columns
best_params.json             selected parameters, seed and final strategy
metrics.json                 validation/test model and baseline metrics
data_quality.json            columns, ordering, duplicates, weather gaps, boundary loss
split_metadata.json          exact time ranges and row counts
tuning_results.json          validation-only search results
predictions.csv              timestamp, actual, prediction, split, model
actual_vs_predicted.png      first test-week comparison
residual_distribution.png    test residual histogram
```

### Formal real-data run (results)

The formal run used ~208,223 supplied origin-time rows; after structural
boundary loss and a small number of missing-irradiance rows, ~205,900 rows
remained before purged splitting.

| Split | Model | MAE (MW) | RMSE (MW) | MAPE | n |
|---|---|---:|---:|---:|---:|
| Validation | LightGBM | 367.91 | 515.58 | 6.81% | 30,797 |
| Validation | Seasonal naive | 491.40 | 678.44 | 9.14% | 30,797 |
| Test | LightGBM | 397.49 | 554.50 | 10.14% | 30,799 |
| Test | Seasonal naive | 524.05 | 746.44 | 13.38% | 30,799 |

Test MAE is approximately **24.1% lower** than the seasonal-naive
baseline. Selected LightGBM candidate: learning rate 0.03, 63 leaves.

**After the model was approved: dataset, split boundaries, feature
definitions and order, hyperparameters, seed, and trained model were
frozen.** All later SHAP experiments reuse them; only background and
evaluation sample selection may change.

### Training tests

```bash
python -m pytest -q tests/test_training.py tests/test_fetch_aemo_native.py tests/test_merge_multi_resolution.py
```

## Run: Week 7 — TreeSHAP pilot

### 1. Load the frozen model

```bash
python src/load_model.py --artifacts artifacts/training_real
```

Reads `model.joblib` and `feature_names.json`, prints the feature list and
recorded training metrics, and runs a cheap sanity check (predicting on a
dummy all-zero row) to confirm the model is genuinely loadable and callable
before any SHAP code is built on top of it.

### 2. Run a single SHAP configuration

```bash
python src/shap_pilot.py \
  --data data/interim/merged_full_frozen.csv \
  --config configs/training.json \
  --artifacts artifacts/training_real \
  --background-size 200 \
  --evaluation-size 100 \
  --background-method uniform \
  --evaluation-method time_stratified \
  --seed 2082 \
  --output artifacts/shap_pilot_week8
```

This:
- Reuses the training pipeline's exact `prepare_frame()` logic, so feature
  columns are guaranteed to line up with what the frozen model expects
  (and fails loudly if they don't)
- Draws the background from the purged training split and the evaluation
  sample from the held-out test split, using the methods in
  `src/sampling.py`
- Wires up `shap.TreeExplainer` in **interventional** mode — the mode that
  takes an explicit background dataset, which is the actual variable this
  project's research question manipulates
- Verifies the result mathematically via an **additivity check**:
  `prediction == base_value + sum(SHAP values)` for every explained row,
  within a documented tolerance (see "Additivity gaps" below)
- Aggregates to a global ranking (mean absolute SHAP value per feature),
  saves it as a CSV and a bar chart

Outputs under the selected output directory:

```text
pilot_global_ranking.csv        feature, mean_abs_shap
pilot_global_ranking.png        bar chart, top 15 features
rare_event_definition.json      training-only preregistered thresholds
outcome_demand_definition.json  separate retrospective target-demand threshold
run_metadata.json               methods, sizes, seed, sample pools and runtime
```

### Sample construction methods (`src/sampling.py`)

- `uniform`: reproducible simple random sampling
- `kmeans`: distinct real rows nearest MiniBatchKMeans centres after
  standardisation. The frozen dataset contains genuine NaNs (e.g.
  `temp_max` on the two incomplete BOM days); these are median-filled
  **only for the clustering distance calculation**, and the returned rows
  are the real, unmodified rows with NaNs intact, so the model sees
  exactly what it saw in training and the candidate pool matches every
  other method
- `time_stratified`: proportional coverage across calendar month and four
  six-hour time-of-day blocks (up to 48 strata)
- `season_stratified`: proportional coverage across the four seasons.
  Evaluation samples come from the test split, whose forecast origins run
  from 14 September to 29 December 2025, so only spring and early summer
  are present. In practice it balances two seasons (roughly 73% spring,
  27% summer) and behaves almost like uniform sampling
- `rare_event_stratified`: 50% rare-event rows and 50% ordinary rows by
  default. Primary rare events are public holidays and extreme origin-time
  temperatures, using temperature thresholds fixed from training data
- `outcome_demand_stratified`: a separate retrospective evaluation cohort
  based on realised target demand at `t+24h`. It is never described as
  information available at forecast origin

Background and evaluation pools are deliberately separated: background
rows come from the purged training split, evaluation rows from the
held-out purged test split. Method roles are explicit and enforced:

| Role | Methods |
|---|---|
| Background | `uniform`, `kmeans`, `time_stratified` |
| Evaluation | `uniform`, `time_stratified`, `season_stratified`, `rare_event_stratified`, `outcome_demand_stratified` |

### Two SHAP wiring issues found and fixed

**1. SHAP silently caps background data at 100 rows.** `shap.TreeExplainer`
has an undocumented default that subsamples any larger background dataset
down to 100 rows internally. Left unfixed, this would have quietly
corrupted every background-size experiment — the core variable of this
project's research question. Fixed by explicitly constructing a
`shap.maskers.Independent(background, max_samples=len(background))`
rather than passing the raw background data directly.

**2. Additivity gaps: real cause found in Week 9 (the Week 7 explanation
was wrong).** Most evaluation rows reconstruct to within ~0.001 MW, but a
small number show gaps of up to ~110 MW. Week 7 attributed this to
LightGBM's `regression_l1` objective, based on a comparison across
objectives. That explanation does not hold: a plain `regression_l1` model
reconstructs exactly in isolated tests, and the objective comparison was
misleading because each objective trains entirely different trees, with
split thresholds in different places.

The actual cause, confirmed on the frozen model with
`scripts/diagnose_additivity.py`: when a feature value sits exactly on a
tree split threshold, LightGBM's native prediction and SHAP's internal
copy of the tree route the row down different branches, and the gap
equals the difference between the two leaves. For the worst row of the
seed-2087 run, nudging `temperature` (10.9) by one part in a million
changes the native prediction by 109.5519 MW, exactly matching the
additivity error; three of the four worst rows in that run share the
value 10.90. This is more frequent here because daily temperature and
hourly irradiance are broadcast across 5-minute rows, so exact values
repeat many times, and LightGBM places split thresholds at observed
values.

The impact on global rankings is bounded: a row's error can shift a run's
mean |SHAP| totals by at most error / evaluation size (≤ 1.15 MW in the
diagnosed run, against inter-feature gaps of tens of MW). The check keeps
a 1% warning threshold and a 5% hard limit, and every run's maximum error
is recorded in `runs.csv`. Across the Week 9 grid, 534 of 1,800 runs
raised a warning (more often at evaluation size 500, since larger samples
are more likely to contain a threshold row); none reached the hard limit.

To diagnose any specific run:

```bash
python scripts/diagnose_additivity.py --data data/interim/merged_full_frozen.csv \
  --background-method uniform --background-size 50 \
  --evaluation-method uniform --evaluation-size 100 --seed 2087
```

### Frozen-model provenance mismatch (found and resolved)

Partway through Week 7, a reproducibility issue surfaced, distinct from
the wiring issues above.

`artifacts/` is gitignored (correctly — it's large, regenerable output),
which meant only training *code* was ever shared via GitHub between team
members, never the actual trained `model.joblib`. When the Week 7 pilot
was first built, a model was regenerated locally by running
`scripts/train_model.py` directly against a freshly-generated
`merged_full.csv`. This produced a **different model** than the one
actually used for the Week 6 presentation's reported results (397.49 MW
test MAE), without that being obvious at first — both models used the
same code, the same config, and the same random seed, and produced
metrics close enough (402.84 vs 397.49 MW) to plausibly look like ordinary
run-to-run noise.

The actual cause was a genuinely different, larger feature set: the
formal `training_real` model has **26 features**, not 16. The extra 10 —
`lag_5min`, `lag_10min`, `lag_15min`, `lag_1day`, `lag_1week`,
`roll_mean_1h`, `roll_std_1h`, `roll_mean_24h`, `roll_std_24h`,
`origin_hour_of_day` — come from an earlier, Week 5 feature-engineering
script (`src/feature_engineering.py`, now superseded by `training.py`'s
own `prepare_frame()`), whose output survived
`scripts/integrate_processed_features.py`'s target-column stripping (it
only removes `target_*`-prefixed columns, not these) and ended up folded
into the formal training run.

**This was caught, not guessed:** comparing `feature_names.json` from both
models showed the exact column-set difference, and running the SHAP pilot
against the mismatched pair triggered the deliberate feature-mismatch
guard in `run_pilot()`.

**Resolution:** the exact input CSV used for the formal run was shared
(`data/interim/merged_full_frozen.csv` — 208,223 rows, 23 source columns,
reproducing the documented 26-feature set exactly) alongside the four
artifact files. Re-running the pilot against this exact pairing confirmed
a clean feature-set match (26/26) and a passing additivity check.

**Lesson for the team:** a frozen model is only meaningfully frozen if its
*exact* input data is also pinned and shared, not just its training code.

### Pilot results (frozen model, background n=200, evaluation n=100)

| Feature | Mean \|SHAP\| |
|---|---:|
| `lag_5min` (5 min ago) | 240.7 |
| `day_of_week` | 154.7 |
| `day_of_year` | 128.4 |
| `origin_hour_of_day` | 123.4 |
| `lag_10min` (10 min ago) | 97.8 |
| `lag_1week` | 78.2 |
| `roll_mean_24h` | 66.0 |
| `temp_max` | 58.5 |
| `irr_irradiance_diffuse` | 49.2 |
| `lag_1day` | 36.9 |

A single-configuration sanity check, not the RQ1 experiment. The very
short-term lags dominate, plausibly acting as a proxy for the current
demand "regime" (weather, season, overall load level) that carries
forward reasonably well even 24 hours ahead; `day_of_week` and
`origin_hour_of_day` capture calendar structure; weather features
contribute at a real but comparatively lower level.

### Week 8 small-grid pilot

A 12-run pilot (background `uniform`/`time_stratified` × sizes 50/200 ×
3 seeds, fixed 100-row time-stratified evaluation) validated the design
and runtime. It reported mean Spearman 0.967, Kendall 0.895 and Top-10
overlap 0.929 across 66 run pairs.

**Note:** those 66 pairs included comparisons between *different*
configurations, so the pilot figures mix within-condition stability with
between-condition agreement. The Week 9 grid reports these separately;
use the Week 9 figures for RQ1.

## Run: Week 9 — full sampling grid (RQ1)

```bash
python src/shap_experiment.py \
  --data data/interim/merged_full_frozen.csv \
  --artifacts artifacts/training_real \
  --background-methods uniform,kmeans,time_stratified \
  --background-sizes 50,200 \
  --evaluation-methods uniform,time_stratified,season_stratified,rare_event_stratified,outcome_demand_stratified \
  --evaluation-sizes 100,500 \
  --n-seeds 30 \
  --output artifacts/shap_week9_grid
```

Every combination of background method × background size × evaluation
method × evaluation size × seed is run against the frozen model: 60
configurations × 30 seeds = 1,800 runs. Seeds are 2082–2111; each
evaluation sample uses `seed + 100000` so background and evaluation draws
are independent.

### Two levels of stability

- **Within-condition stability:** seeds are compared only against other
  seeds of the *same* configuration. This answers "if I repeat this exact
  sampling design, do I get the same ranking?" — the core RQ1
  measurement. Reported as pairwise Spearman, Kendall tau, top-5 and
  top-10 Jaccard, and Kendall's coefficient of concordance (W) across all
  30 seeds of each configuration.
- **Between-condition agreement:** each configuration's *consensus*
  ranking (mean |SHAP| across its seeds) is compared against the
  consensus ranking of a reference configuration — by default the largest
  uniform/uniform configuration (`bg-uniform_n-200__eval-uniform_n-500`),
  following the spec's treatment of the largest sample as the stable
  reference. This answers "does changing the sampling design change the
  answer?"

Note on metrics: `top_k_overlap` is shared features / k, which is **not**
Jaccard. Jaccard is shared / union, reported separately as `jaccard_top5`
and `jaccard_top10`. The spec specifies Jaccard.

### Running safely

- **Preflight:** one sample per method and size is constructed before any
  SHAP runs, so problems such as too few rare-event rows for the largest
  evaluation size fail in seconds rather than hours in.
- **Resume:** each run is saved to `runs/` as soon as it finishes.
  Re-running the same command into the same output directory skips
  completed runs, so an interrupted grid continues rather than restarts.
  Adding seeds later only computes the new ones.
- **Manifest guard:** `experiment_manifest.json` records the model and
  data the directory was created from; the script refuses to resume into
  a directory produced from a different model or dataset.
- **Runtime:** ~13 s per run on average (about 3 s at evaluation size 100,
  up to ~35 s at size 500 with a 200-row background). `runtime_seconds_*`
  in the summary times SHAP computation only, not sample construction
  (k-means refits clusters on ~144,000 rows per run), so real wall-clock
  time is longer. For reporting, measure wall-clock time from the
  timestamps of the first and last files in `runs/`.

### Outputs

Under `artifacts/shap_week9_grid/`:

```text
runs/                             one ranking CSV + metadata JSON per run
runs.csv                          one row per run: config, seed, runtime, additivity
rankings_long.csv                 run_id, config, seed, feature, mean_abs_shap, rank
rankings_wide.csv                 features x run_id
pairwise_within_condition.csv     seed-vs-seed comparisons within each configuration
within_condition_stability.csv    one row per configuration: stability summary + Kendall's W
consensus_rankings.csv            features x configuration, mean |SHAP| across seeds
between_condition_agreement.csv   each configuration's consensus vs the reference
experiment_summary.json           headline results and definitions
experiment_manifest.json          model/data identity guard for resuming
results_schema.json               column listing for every output file
```

`pairwise_stability.csv` from Week 8 is no longer written, since it mixed
the two levels of stability.

`artifacts/` is gitignored, so these results exist only where the grid
was run. **Back up `artifacts/shap_week9_grid` and share that exact folder
between team members** rather than re-running, per the provenance lesson
above.

### Week 9 results (pooled across all 60 configurations)

| Within-condition metric | Mean | Minimum |
|---|---:|---:|
| Spearman rank correlation | 0.987 | 0.920 |
| Kendall tau | 0.938 | 0.785 |
| Kendall's W (per configuration) | 0.987 | 0.973 |
| Top-10 Jaccard | 0.918 | 0.667 |
| Top-5 Jaccard | 0.969 | 0.429 |

26,100 within-condition pairs (60 configurations × 435 seed pairs).

Whole-ranking metrics show rankings are highly repeatable: even the least
stable configuration has W = 0.973. But the top-5 Jaccard minimum of
0.429 means that for at least one pair of seeds, only 3 of the top-5
features were shared. The overall ordering barely moves while the top of
the ranking occasionally does: Spearman and W are dominated by the many
lower-ranked features whose order rarely changes, which can mask
instability at the top, where feature-importance decisions are actually
made.

These are pooled figures. How stability varies *with* sample size and
construction method, and how much the sampling design changes the
answer, is in `within_condition_stability.csv` and
`between_condition_agreement.csv`, and is Week 10's analysis.

## Run: Week 10 — extended grid and analysis (RQ1, RQ3)

### Extending the grid

Copy the Week 9 results first, so the original run is kept as a record and
the extension only computes new runs (resume skips everything already in
the folder):

```powershell
Copy-Item -Recurse artifacts/shap_week9_grid artifacts/shap_week10_grid
```

Time one seed of each new configuration before committing to the full run:

```bash
python src/shap_experiment.py \
  --data data/interim/merged_full_frozen.csv \
  --artifacts artifacts/training_real \
  --background-methods uniform,kmeans,time_stratified \
  --background-sizes 25,50,100,200 \
  --evaluation-methods uniform,time_stratified,season_stratified,rare_event_stratified,outcome_demand_stratified \
  --evaluation-sizes 100,500 \
  --n-seeds 1 \
  --output artifacts/shap_week10_grid
```

Then rerun the same command with `--n-seeds 30`. Mean SHAP seconds per run
across the full grid:

| Background size | Evaluation 100 | Evaluation 500 |
|---|---:|---:|
| 25 | 2.6 | 6.1 |
| 50 | 2.9 | 8.8 |
| 100 | 4.5 | 17.2 |
| 200 | 7.8 | 33.3 |

### Analysis

```bash
python src/analyse_stability.py \
  --grid artifacts/shap_week10_grid \
  --output artifacts/analysis_week10
```

Takes about a minute and writes every table and figure used in the report:

```text
factor_effects_within.csv          mean stability by each design factor
factor_effects_between.csv         mean agreement with the reference by factor
kruskal_wallis.csv                 H and Holm-adjusted p per factor x metric
dunn_posthoc.csv                   pairwise Dunn tests, Holm-adjusted
size_tradeoff.csv                  stability and runtime per background x evaluation size
size_paired_tests.csv              each size vs the largest, matched on all other factors
size_regression.csv                metric ~ log2(background size)
top5_boundary.csv                  features ranked 4-6 per configuration, with gaps
feature_importance_by_method.csv   median importance and rank per evaluation method
topk_changes_vs_reference.csv      features entering / leaving the top 10
top5_swaps_by_run.csv              per run: top-5 features differing from the reference
top5_swap_summary.csv              swap rates and features involved, per evaluation method
bootstrap_ci.csv                   95% bootstrap CI on each feature's mean |SHAP|
stability_vs_background_size.png
top5_by_evaluation_method.png
temperature_importance_by_method.png
cost_vs_stability.png
summary.json                       headline numbers
```

Tests use configuration-level values (pairs within a configuration are not
independent). Size comparisons pair each configuration with the one
identical except for size; a non-significant result is not proof of
equivalence, so median differences are reported alongside p-values.

### Week 10 results

**Whole-ranking repeatability rises with background size** (Kruskal–Wallis
H = 44.0, p < 0.001):

| Background size | Kendall's W | Top-5 Jaccard |
|---|---:|---:|
| 25 | 0.977 | 0.924 |
| 50 | 0.984 | 0.966 |
| 100 | 0.987 | 0.971 |
| 200 | 0.991 | 0.971 |

**Top-5 stability depends on the evaluation method** (H = 47.2,
p < 0.001), not the background method (p = 0.98):

| Evaluation method | Top-5 Jaccard | Runs with a different top 5 |
|---|---:|---:|
| Uniform | 0.992 | 1.2% |
| Season-stratified | 0.991 | 1.4% |
| Time-stratified | 0.988 | 1.8% |
| Outcome-demand-stratified | 0.943 | 10.0% |
| Rare-event-stratified | 0.876 | 27.6% |

Under rare-event evaluation `temp_max` gains about a third in importance
(median across configurations of mean |SHAP|, 68.9 → 91.5 MW) and moves from 8th to 6th, entering the top 5
in 186 of the 199 rare-event runs whose top 5 differed. Averaged over 30
seeds, every configuration's consensus top 5 matches the reference, except
that k-means backgrounds with rare-event evaluation leave 5th place a
genuine tie (bootstrap intervals overlap).

**RQ3, stability side:** against a 200-row background, 50 rows show no
detectable top-5 loss (median difference 0.000, p = 0.77) at about a
quarter of the cost; the top 10 needs 100 rows (p = 0.33). Kendall's W is
slightly lower at every smaller size. Rare-event evaluation needs a
100-row background. The final RQ3 answer awaits the RQ2 reliability
results.

## Tests

```bash
python -m pytest -q tests/
```

Or by area:

```bash
# data pipeline
python -m pytest -q tests/test_fetch_aemo_native.py tests/test_merge_multi_resolution.py
# training
python -m pytest -q tests/test_training.py
# frozen model loader and single-configuration SHAP
python -m pytest -q tests/test_load_model.py tests/test_shap_pilot.py
# sampling methods and the full grid
python -m pytest -q tests/test_sampling.py tests/test_shap_experiment.py tests/test_shap_experiment_grid.py
# analysis (RQ1, RQ3)
python -m pytest -q tests/test_analyse_stability.py
```

65 tests in total.

The grid tests cover: Jaccard and Kendall's W correctness (including the
identity linking W to mean Spearman, and ties at zero importance); that
within-condition pairs never cross configurations; resume after an
interrupted run; extending seeds without recomputing; the manifest guard;
and preflight failing before any SHAP runs. The sampling tests include
k-means tolerating NaN features while returning real, unmodified rows.
The analysis tests check Holm and Dunn against hand-calculated answers,
the bootstrap intervals, matched pairing in the size tests, and run the
full analysis on a synthetic grid built with `shap_experiment.py`'s own
functions, so they test exactly the file formats the real grid produces.

**Note:** if you hit a `_tkinter.TclError` about a missing `init.tcl` file
when running tests on Windows, this is a known issue with Python installs
from the Microsoft Store (their Tcl/Tk bundling can be incomplete inside
the Store's sandboxed environment) — unrelated to this project's code.
The tests already set `matplotlib.use("Agg")` to avoid needing a working
Tkinter at all; if you still hit this in your own scripts, add the same
line before importing `matplotlib.pyplot`.

## Before the final report (due Monday, Week 14)

- [ ] RQ2: permutation importance and ablation on the reference and
      candidate configurations (background 50 and 100 with evaluation
      500), including temperature on rare-event vs ordinary days
- [ ] Final RQ3 answer once reliability results are in
- [ ] Abstract and conclusion, written last
- [ ] Full read-through of the report and a test export to PDF/DOCX
- [ ] Correct the additivity explanation with Zeehan, if not already done
      (the L1-objective account appeared on the Week 7 slide)
- [ ] Back up `artifacts/shap_week10_grid` and share it with the team

## Completed checklist (Weeks 4-10)

- [x] All three data sources (AEMO, BOM, renewables.ninja) fetched,
      cleaned, and merged
- [x] EDA complete; no anomalies found requiring `EXCLUDED_PERIODS`
- [x] Feature engineering — superseded by the training pipeline's own
      `prepare_frame()`, which handles this more rigorously (explicit
      origin/target semantics, purged splits)
- [x] LightGBM trained, tuned, and validated against baseline
- [x] Model, features, hyperparameters, and split frozen for Week 7+
      (`artifacts/training_real/`, input data `merged_full_frozen.csv`)
- [x] TreeSHAP wired up in interventional mode, verified via additivity
- [x] SHAP background cap and frozen-model provenance mismatch found and
      fixed
- [x] Sample construction methods implemented, including
      `season_stratified`; k-means NaN handling fixed
- [x] Full 1,800-run sampling grid completed with within- and
      between-condition stability separated
- [x] Real cause of additivity gaps identified and documented (split
      threshold coincidence with broadcast weather values)
- [x] Grid extended to 120 configurations (3,600 runs), timed first
- [x] Analysis script with statistical tests, paired size comparisons and
      bootstrap intervals; RQ1 and the stability side of RQ3 analysed
