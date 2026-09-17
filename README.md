# FIT2082 Data Pipeline

Data acquisition, cleaning, model training, and explainability pipeline for
*Sampling Sensitivity of Global SHAP Feature Rankings in Short-Term
Electricity Demand Forecasting*.

Produces a clean, chronologically-split dataset combining AEMO Victorian
electricity demand (native 5-minute resolution, per supervisor direction),
BOM daily temperature, renewables.ninja hourly solar irradiance, and
derived calendar features — trains and validates a LightGBM forecaster
against a seasonal-naive baseline — and wires up a TreeSHAP explainer,
verified mathematically, ready for the full background/evaluation sampling
experiment grid in Weeks 8-9.

## Project status (running log, most recent first)

- **Week 8:** Controlled sample construction implemented in
  `src/sampling.py`: uniform random, k-means representatives,
  month/time-of-day stratified, and rare-event-stratified sampling.
  Background samples now come only from the purged training split and
  evaluation samples only from the held-out purged test split. Rare-event
  thresholds are pre-registered from training data. Primary rare events
  are public holidays or origin-time temperature outside the training
  5th-95th percentile range. Realised high target demand is kept as a
  separate, explicitly retrospective evaluation cohort.
  A 12-run real-data pilot grid completed successfully across two
  background methods, two background sizes and three seeds. Mean pairwise
  ranking stability was Spearman 0.967, Kendall 0.895 and Top-10 overlap
  92.9%; every run records its sampling design, runtime and additivity
  diagnostic.
- **Week 7:** TreeSHAP pilot wired up (`src/shap_pilot.py`) against the
  frozen model, using `interventional` perturbation mode. Verified via a
  mathematical additivity check, not just eyeballed. Two real bugs caught
  and fixed in the process (see "Week 7: TreeSHAP pilot" below). Pilot
  ranking on real data matches EDA intuition (recent demand, calendar/time
  features and weekly demand history dominate; weather signals are lower).
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

The training entry point consumes the Week 4 merged dataset directly. It
does not download data or implement the later SHAP sampling experiment.

```bash
python scripts/train_model.py \
  --data data/interim/merged_full.csv \
  --config configs/training.json \
  --output artifacts/training_real
```

The formal real-data artifacts are stored under
`artifacts/training_real`. The Week 7 commands below reuse this exact
directory so that the SHAP pilot loads the approved real-data model rather
than a smoke-test or separately generated model.

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
this was a deliberate methodological choice (see the training pipeline's
docstrings) to avoid treating future weather as known, which real
day-ahead forecasting can't assume.

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
frozen.** Week 7 TreeSHAP experiments reuse them; only background and
evaluation sample selection may change from here on.

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

### 2. Run the TreeSHAP pilot

```bash
python src/shap_pilot.py \
  --data data/interim/merged_full.csv \
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
- Draws a plain random background and evaluation sample (a placeholder for
  Weeks 8-9's actual experimental variable — background/evaluation size
  *and* construction method: uniform, k-means, time-stratified, rare-event
  -stratified)
- Wires up `shap.TreeExplainer` in **interventional** mode — the mode that
  takes an explicit background dataset, which is the actual variable this
  project's whole research question manipulates
- Verifies the result mathematically via an **additivity check**:
  `prediction == base_value + sum(SHAP values)` must hold for every
  explained row, within a documented tolerance (see below) — not just
  assumed to be correct because nothing crashed
- Aggregates to a global ranking (mean absolute SHAP value per feature),
  saves it as a CSV and a bar chart

Outputs under the selected pilot output directory (for the example above,
`artifacts/shap_pilot_week8/`):

```text
pilot_global_ranking.csv     feature, mean_abs_shap
pilot_global_ranking.png     bar chart, top 15 features
rare_event_definition.json   training-only preregistered thresholds
outcome_demand_definition.json  separate retrospective target-demand threshold
run_metadata.json            methods, sizes, seed, sample pools and runtime
```

### Week 8 controlled sampling design

The sample-construction methods are implemented in `src/sampling.py`:

- `uniform`: reproducible simple random sampling
- `kmeans`: distinct real rows nearest MiniBatchKMeans centres after
  standardisation
- `time_stratified`: proportional coverage across calendar month and four
  six-hour time-of-day blocks
- `rare_event_stratified`: 50% rare-event rows and 50% ordinary rows by
  default. Primary rare events are public holidays and extreme origin-time
  temperatures, using temperature thresholds fixed from training data
- `outcome_demand_stratified`: a separate retrospective evaluation cohort
  based on realised target demand at `t+24h`. It is never described as
  information available at forecast origin

Background and evaluation pools are deliberately separated. Background
rows are drawn from the purged training split; evaluation rows are drawn
from the held-out purged test split. This prevents the Week 7 placeholder
behaviour of drawing both samples from the complete prepared frame.
The implementation is shared, but method roles are explicit: background
methods are `uniform`, `kmeans`, and `time_stratified`; evaluation methods
are `uniform`, `time_stratified`, `rare_event_stratified`, and
`outcome_demand_stratified`. `season_stratified` is intentionally left for
David's follow-up contribution.

The initial controlled grid should remain a pilot rather than the complete
30-seed factorial experiment requested later in the project:

| Variable | Initial pilot values |
|---|---|
| Background method | uniform, kmeans, time_stratified |
| Background size | 50, 200 |
| Evaluation method | uniform, time_stratified, rare_event_stratified, outcome_demand_stratified |
| Evaluation size | 100, 500 |
| Seeds | 2082, 2083, 2084 |

Use the pilot runtimes and rank-stability results to decide whether the full
size grid (for example 25, 50, 100, 250 and 500) and approximately 30 seeds
require the Monash M3 cluster. Do not change the frozen model, feature order,
training data or split boundaries between runs.

Run the Week 8 sampling tests with:

```bash
python -m pytest -q tests/test_sampling.py tests/test_shap_pilot.py tests/test_shap_experiment.py
```

### Two real bugs found and fixed this week

**1. SHAP silently caps background data at 100 rows.** `shap.TreeExplainer`
has an undocumented default that subsamples any larger background dataset
down to 100 rows internally. Left unfixed, this would have quietly
corrupted every background-size experiment in Weeks 8-9 — the literal core
variable of this project's research question, since the intended sample
size would never actually reach the explainer. Fixed by explicitly
constructing a `shap.maskers.Independent(background, max_samples=len(background))`
rather than passing the raw background data directly.

**2. The additivity check initially failed by ~1-13 MW.** Diagnosed
empirically (not assumed) by testing the same setup across different
LightGBM objectives: `regression_l1` (MAE, used by this project's frozen
model) computes leaf values via an iterative approximation rather than an
exact closed-form value, unlike `regression` (L2) or `huber`, which showed
the gap shrink by roughly 5x and 10,000x respectively when swapped in on
the same data. This isolates the cause to the L1 objective's leaf-fitting
procedure — a small, bounded, known characteristic, not a wiring bug. The
additivity check now uses `max(5.0 MW, 1% of median prediction)` as a
diagnostic warning threshold and `max(5.0 MW, 5% of median prediction)`
as a hard failure limit. This records sample-dependent L1 reconstruction
variation rather than hiding it by repeatedly changing one tolerance. A
genuinely corrupted explanation still fails the hard guard.

### Pilot results (real data, background n=200, evaluation n=100)

| Feature | Mean \|SHAP\| |
|---|---:|
| `lag_5min` | 240.7 |
| `day_of_week` | 154.7 |
| `day_of_year` | 128.4 |
| `origin_hour_of_day` | 123.4 |
| `lag_10min` | 97.8 |
| `lag_1week` | 78.2 |
| `roll_mean_24h` | 66.0 |
| `temp_max` | 58.5 |
| `irr_irradiance_diffuse` | 49.2 |
| `lag_1day` | 36.9 |

This is a **pilot sanity check, not the formal RQ1 experiment** — it
confirms the explainer is wired correctly and the ranking is plausible.
The table is from the frozen `artifacts/training_real` model, not the
separately regenerated `artifacts/training` model. The broader
background/evaluation sampling experiment remains Weeks 8-9's work.

### Week 8 small-grid result

The completed 12-run grid varied background method (`uniform` and
`time_stratified`), background size (50 and 200) and seed (2082-2084),
while holding a 100-row time-stratified test evaluation sample design.
Across all 66 run pairs, mean/minimum stability was:

| Metric | Mean | Minimum |
|---|---:|---:|
| Spearman rank correlation | 0.967 | 0.936 |
| Kendall rank correlation | 0.895 | 0.828 |
| Top-10 overlap | 0.929 | 0.900 |

These are pilot results used to validate the design and estimate runtime,
not the final factorial experiment or final research conclusion.

The experiment writes `runs.csv`, `pairwise_stability.csv`,
`rankings_wide.csv`, `rankings_long.csv`, and `results_schema.json`.
The long table contains `feature`, `run_id`, `mean_abs_shap`, and `rank`,
so later analysis can append runs without parsing per-run files.

### Week 7 tests

```bash
python -m pytest -q tests/test_load_model.py tests/test_shap_pilot.py
```

Covers: the loader against a genuinely (fast-)trained model; the
additivity check both passing correctly and correctly detecting a
deliberately broken explainer (not just a rubber-stamp check); a
zero-variance feature receiving exactly zero SHAP importance; and a
feature-mismatch guard that fails loudly before wasting time computing
SHAP values against the wrong columns.

**Note:** if you hit a `_tkinter.TclError` about a missing `init.tcl` file
when running tests on Windows, this is a known issue with Python installs
from the Microsoft Store (their Tcl/Tk bundling can be incomplete inside
the Store's sandboxed environment) — unrelated to this project's code.
The tests already set `matplotlib.use("Agg")` to avoid needing a working
Tkinter at all; if you still hit this in your own scripts, add the same
line before importing `matplotlib.pyplot`.

## Before Week 8

- [x] Design the actual background/evaluation sample construction module
      (uniform random, k-means summarisation, time-stratified, rare-event
      -stratified) — `shap_pilot.py`'s `draw_sample()` is a placeholder
      this should replace
- [x] Confirm the 1-day forecast horizon with Zeehan (fixed 24-hour-ahead
      point forecast)
- [x] Define a small 3-seed pilot design before considering the full
      approximately 30-seed experiment
- [x] Define the initial background/evaluation size grid; use measured
      pilot runtime and stability to decide whether Monash M3 is needed

## Completed checklist (Weeks 4-7)

- [x] All three data sources (AEMO, BOM, renewables.ninja) fetched,
      cleaned, and merged
- [x] EDA complete; no anomalies found requiring `EXCLUDED_PERIODS`
- [x] Feature engineering (lags, rolling stats, target-aligned
      weather/calendar) — superseded by the training pipeline's own
      `prepare_frame()`, which handles this more rigorously (explicit
      origin/target semantics, purged splits)
- [x] LightGBM trained, tuned, and validated against baseline
- [x] Model, features, hyperparameters, and split frozen for Week 7+
- [x] TreeSHAP wired up in interventional mode, verified via additivity
- [x] Two real bugs found and fixed before they could affect Weeks 8-9
