# FIT2082 Data Pipeline

Data acquisition and cleaning pipeline for *Sampling Sensitivity of Global
SHAP Feature Rankings in Short-Term Electricity Demand Forecasting*.

Produces a clean, chronologically-split dataset combining AEMO Victorian
electricity demand (native 5-minute resolution, per supervisor direction),
BOM daily temperature, renewables.ninja hourly solar irradiance, and
derived calendar features (day-of-week, public holiday flag, season).

## Project status

Per Zeehan's Week 4 feedback email and follow-up meeting direction:
- AEMO demand stays at its **native 5-minute resolution** — no resampling.
- Weather now includes **both temperature (BOM) and irradiance
  (renewables.ninja)**, not temperature alone.

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
- `EXCLUDED_PERIODS` — any anomaly windows you identify during EDA

## Run

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

## Tests

```bash
python tests/test_fetch_aemo_native.py
python tests/test_merge_multi_resolution.py
```

Cover region-filtering, intervention-duplicate handling, native-resolution
preservation, and the two-tier broadcast merge (daily temperature, hourly
irradiance, both onto 5-minute demand) — synthetic data, no network needed.

## Week 5–6: LightGBM training baseline

The training entry point consumes the Week 4 merged dataset directly. It does
not download data or implement the later SHAP sampling experiment.

```bash
python scripts/train_model.py \
  --data data/interim/merged_full.csv \
  --config configs/training.json \
  --output artifacts/training
```

The pipeline:

- sorts and validates the `timestamp` column, then uses the existing 70/15/15
  chronological split with strict non-overlap assertions;
- adds calendar fields plus causal demand lags at 288 and 2,016 five-minute
  intervals (the same time on the previous day/week);
- compares LightGBM with a previous-day seasonal-naive baseline;
- selects a small, fixed parameter grid using validation MAE and LightGBM
  early stopping; the test set is evaluated only after selection;
- safely reports MAE, RMSE and MAPE using a configurable denominator floor;
- keeps the train-fitted, validation-selected model for Week 7 TreeSHAP.

Outputs under `artifacts/training/` are:

```text
model.joblib                 reloadable sklearn-style model
model.txt                    LightGBM native model
feature_names.json           exact ordered model columns
best_params.json             selected parameters, seed and final strategy
metrics.json                 validation/test model and baseline metrics
split_metadata.json          exact time ranges and row counts
tuning_results.json          validation-only search results
predictions.csv              timestamp, actual, prediction, split, model
actual_vs_predicted.png      first test-week comparison
residual_distribution.png    test residual histogram
```

Run the training-related tests from the repository root:

```bash
python -m pytest -q tests/test_training.py tests/test_fetch_aemo_native.py tests/test_merge_multi_resolution.py
```

Generated CSV files are intentionally gitignored. If
`data/interim/merged_full.csv` is absent, restore or rerun the Week 4 data
pipeline before formal training. Do not report smoke-test metrics as project
results.

## Before Week 5

- [x] Sign up for renewables.ninja and set `RENEWABLES_NINJA_TOKEN`
- [x] Ran all three fetch scripts against live data — AEMO 210,527 rows
      (native 5-min, 2024–2025), BOM temperature, renewables.ninja
      irradiance (17,533 hourly rows)
- [x] Confirmed irradiance feature columns: `electricity`,
      `irradiance_direct`, `irradiance_diffuse`, `temperature` (this last
      one is MERRA-2's *modeled* temperature, kept as `irr_temperature` in
      the merged output — distinct from BOM's real station reading in the
      `temperature` column, not a duplicate/error)
- [x] Ran `clean_merge.py` — 210,527 rows merged, chronological split
      147,368 / 31,579 / 31,580 (~70/15/15)
- [x] **Known, documented gap:** 155 rows (0.07%) at the very start of the
      dataset (2024-01-01, before ~11am local time) have no matching
      irradiance value. This is a UTC/local-time boundary effect, not a
      real data gap: renewables.ninja's API request starts at
      `2024-01-01 00:00 UTC`, which is `2024-01-01 11:00` in Melbourne
      (AEDT, +11:00) — so there's no irradiance data available for local
      times before that on the very first day, since it would require
      data from before the requested range began. Left in place rather
      than dropped; worth a one-line mention in the report's data-cleaning
      section.
- [ ] Identify and log any other anomaly windows in `EXCLUDED_PERIODS`
- [ ] Prepare the Week 5 progress PowerPoint for Zeehan
