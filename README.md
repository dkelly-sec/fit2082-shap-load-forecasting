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

- **Week 7:** TreeSHAP pilot wired up (`src/shap_pilot.py`) against the
  frozen model, using `interventional` perturbation mode. Verified via a
  mathematical additivity check, not just eyeballed. Two real bugs caught
  and fixed in the process (see "Week 7: TreeSHAP pilot" below). Pilot
  ranking on real data matches EDA intuition (hour-of-day and demand lags
  dominate; weather signals cluster mid-ranking).
- **Weeks 5-6:** LightGBM forecaster trained and tuned via time-series CV
  with a purged chronological split, validated against a seasonal-naive
  baseline (~23.6% lower test MAE). Model, features, hyperparameters, and
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
  --output artifacts/training
```

**Note on output folder naming:** the folder name after `--output` is
arbitrary — it isn't referenced anywhere in the code itself, only in this
README's prose. Earlier documentation here referred to a
`artifacts/training_real` folder for the formal run specifically (to
distinguish it from smoke-test runs); this has been standardised back to
`artifacts/training` throughout this document. Use whichever name you
like locally, just be consistent when pointing `--artifacts` at it in
Week 7's scripts below.

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

Outputs under `artifacts/training/` are:

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
| Validation | LightGBM | 400.05 | 545.36 | 7.40% | 31,143 |
| Validation | Seasonal naive | 488.01 | 675.52 | 9.08% | 31,143 |
| Test | LightGBM | 402.84 | 573.19 | 10.14% | 31,144 |
| Test | Seasonal naive | 527.55 | 750.15 | 13.52% | 31,144 |

Test MAE is approximately **23.6% lower** than the seasonal-naive
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
python src/load_model.py --artifacts artifacts/training
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
  --artifacts artifacts/training \
  --background-size 200 \
  --evaluation-size 100
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

Outputs under `artifacts/shap_pilot/`:

```text
pilot_global_ranking.csv     feature, mean_abs_shap
pilot_global_ranking.png     bar chart, top 15 features
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
additivity check's tolerance is now scaled to prediction magnitude
(`max(5.0 MW, 0.3% of median prediction)`) rather than a fixed constant,
with the full reasoning documented in `check_additivity()`'s docstring. A
genuine wiring bug would produce errors orders of magnitude larger than
this (hundreds/thousands of MW), so the check remains meaningful.

### Pilot results (real data, background n=200, evaluation n=100)

| Feature | Mean \|SHAP\| |
|---|---:|
| `hour` | 244.0 |
| `demand_lag_288` (yesterday) | 221.1 |
| `demand_lag_2016` (last week) | 214.6 |
| `day_of_week` | 172.4 |
| `day_of_year` | 105.2 |
| `temperature` | 102.3 |
| `irr_temperature` | 83.9 |
| `temp_max` | 81.3 |
| `irr_electricity` | 76.1 |
| `irr_irradiance_diffuse` | 67.2 |

This is a **pilot sanity check, not the formal RQ1 experiment** — it
confirms the explainer is wired correctly and the ranking is plausible
(time-of-day and demand lags dominate, matching the EDA's hourly/duck-curve
findings; weather signals cluster together mid-ranking rather than any
one dominating). The actual background/evaluation sampling experiment
grid is Weeks 8-9's work.

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

- [ ] Design the actual background/evaluation sample construction module
      (uniform random, k-means summarisation, time-stratified, rare-event
      -stratified) — `shap_pilot.py`'s `draw_sample()` is a placeholder
      this should replace
- [ ] Confirm the 1-day forecast horizon with Zeehan (raised as an open
      question in the Week 7 progress update)
- [ ] Plan the 30-seed repeated-sampling design and whether Monash's M3
      HPC cluster is needed given compute cost
- [ ] Decide the exact background/evaluation size grid to test (spec
      suggests e.g. 25, 50, 100, 250, 500 for background; 10%-100% of test
      set for evaluation)

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
