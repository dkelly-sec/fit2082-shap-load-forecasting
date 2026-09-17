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
  mathematical additivity check, not just eyeballed. Three real issues
  caught and fixed in the process (two wiring bugs, plus a frozen-model
  provenance mismatch — see "Week 7: TreeSHAP pilot" below). Pilot ranking
  on the correct, verified frozen model matches EDA intuition (short-term
  demand momentum and day-of-week dominate; weather signals contribute at
  a lower but non-trivial level).
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
  --output artifacts/training
```

**Note on output folder naming:** `artifacts/training_real` is the
authoritative, frozen model referenced throughout this document (Test
LightGBM MAE 397.49 MW, ~24.1% improvement over baseline). An earlier,
independently-regenerated model briefly existed under `artifacts/training`
during Week 7 — see "Frozen-model provenance mismatch" below for what
happened and how it was resolved. `artifacts/training` should not be
treated as authoritative; use `artifacts/training_real` for all Week 7+
work.

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
python src/load_model.py --artifacts artifacts/training_real
```

Reads `model.joblib` and `feature_names.json`, prints the feature list and
recorded training metrics, and runs a cheap sanity check (predicting on a
dummy all-zero row) to confirm the model is genuinely loadable and callable
before any SHAP code is built on top of it.

### 2. Run the TreeSHAP pilot

```bash
python src/shap_pilot.py \
  --data data/interim/merged_full_frozen.csv \
  --config configs/training.json \
  --artifacts artifacts/training_real \
  --background-size 200 \
  --evaluation-size 100
```

**Important:** use `merged_full_frozen.csv`, not `merged_full.csv`. These
are two different files with two different sets of columns — see "Frozen-
model provenance mismatch" below for exactly why, and never confuse the
two going forward. `merged_full_frozen.csv` is the exact input file the
`training_real` model was trained on (26 feature columns); a freshly
regenerated `merged_full.csv` from `clean_merge.py` alone only has 16,
since it skips the legacy `feature_engineering.py` step that produced the
other 10.

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

### Frozen-model provenance mismatch (found and resolved)

Partway through Week 7, a third issue surfaced — distinct from the two
wiring bugs above, and arguably more important to document clearly since
it's a reproducibility issue, not a code bug.

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
script (`src/feature_engineering.py`, now superseded by
`training.py`'s own `prepare_frame()`), whose output survived
`scripts/integrate_processed_features.py`'s target-column stripping (it
only removes `target_*`-prefixed columns, not these) and ended up folded
into the formal training run. A freshly regenerated `merged_full.csv`
skips that legacy script entirely, so those 10 columns never exist in it.

**This was caught, not guessed:** comparing `feature_names.json` from both
models directly showed the exact column-set difference, and running the
SHAP pilot against the mismatched pair triggered the deliberate
feature-mismatch guard in `run_pilot()` — exactly the failure mode that
guard exists to catch.

**Resolution:** the team member who ran the formal training shared the
exact input CSV used (`data/interim/merged_full_frozen.csv` in this repo —
208,223 rows, 23 source columns, confirmed to reproduce the documented
26-feature set exactly) alongside the four artifact files
(`model.joblib`, `feature_names.json`, `best_params.json`,
`metrics.json`). Re-running the pilot against this exact pairing confirmed
a clean feature-set match (26/26) and a passing additivity check.

**Lesson for the team:** a frozen model is only meaningfully frozen if its
*exact* input data is also pinned and shared, not just its training code —
identical code with two different (but similarly-shaped) input files can
silently produce two different models with deceptively similar metrics.

### Pilot results (correct frozen model, background n=200, evaluation n=100)

| Feature | Mean \|SHAP\| |
|---|---:|
| `lag_5min` (5 min ago) | 296.2 |
| `day_of_week` | 144.4 |
| `lag_10min` (10 min ago) | 118.6 |
| `lag_1week` | 104.2 |
| `origin_hour_of_day` | 90.3 |
| `day_of_year` | 86.2 |
| `temp_max` | 74.4 |
| `roll_mean_24h` | 66.7 |
| `temperature` | 60.4 |
| `irr_irradiance_diffuse` | 42.9 |

This is a **pilot sanity check, not the formal RQ1 experiment** — it
confirms the explainer is wired correctly and the ranking is plausible.
The very short-term lags (`lag_5min`, `lag_10min`) dominate, plausibly
acting as a proxy for the current demand "regime" (weather, season,
overall load level) that carries forward reasonably well even 24 hours
ahead; `day_of_week` and `origin_hour_of_day` capture calendar structure;
weather features contribute at a real but comparatively lower level. The
actual background/evaluation sampling experiment grid is Weeks 8-9's work.

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
      (`artifacts/training_real/`, input data `merged_full_frozen.csv`)
- [x] TreeSHAP wired up in interventional mode, verified via additivity
- [x] Three real issues found and fixed before they could affect Weeks
      8-9: two SHAP wiring bugs, plus a frozen-model provenance mismatch
      (see "Frozen-model provenance mismatch" above)
