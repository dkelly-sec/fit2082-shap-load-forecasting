# FIT2082 Data Pipeline — Week 4

Data acquisition and cleaning pipeline for *Sampling Sensitivity of Global
SHAP Feature Rankings in Short-Term Electricity Demand Forecasting*.

Produces a clean, half-hourly, chronologically-split dataset combining AEMO
Victorian electricity demand, BOM weather (temperature, humidity), and
derived calendar features (day-of-week, public holiday flag, season) —
ready to hand to the LightGBM forecaster in Week 5–6.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure

Edit `src/config.py` first:

- `START_DATE` / `END_DATE` — your 2-year window (currently 2024-01-01 to 2025-12-31)
- `BOM_STATION_NAME` / `BOM_STATION_ID` — your chosen weather station
- `EXCLUDED_PERIODS` — any anomaly windows you identify during EDA

## Run

### 1. AEMO demand data

```bash
python src/fetch_aemo.py
```

This uses [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS), a maintained
Python package built specifically for downloading historical AEMO/NEM
data — it handles the split between AEMO's "Current" NEMWEB directory and
the older MMS Data Model Archive automatically, and caches downloaded
files under `data/raw/nemosis_cache/` so repeated runs don't re-download.

It pulls `DISPATCHREGIONSUM` (5-minute resolution) and resamples to
half-hourly by taking the mean, which is standard practice for this kind
of demand-forecasting work.

Notes:
- The first run can be slow and will use noticeable disk space, since
  NEMOSIS downloads full monthly files before filtering to VIC1.
- This was written against NEMOSIS's documented API but **not run against
  the live AEMO site** in the environment this was built in (no network
  access to AEMO's domain there). Run it yourself and sanity-check the
  first few rows before trusting a full 2-year pull.
- If it fails, check whether NEMOSIS needs updating
  (`pip install --upgrade nemosis`) before assuming the pipeline code is
  wrong — AEMO's site structure changes occasionally and NEMOSIS is
  actively maintained to track it.

### 2. BOM weather data

BOM's Climate Data Online doesn't offer free bulk sub-daily downloads via a
simple API — get it manually:

1. Find your station: <https://www.bom.gov.au/climate/data-services/station-data.shtml>
2. Download temperature + humidity for your date range as CSV from
   <http://www.bom.gov.au/climate/data/>
3. Save it as `data/raw/bom_weather_raw.csv`

Then run:

```bash
python src/load_bom.py
```

If it can't find the temperature/humidity/timestamp columns, it'll print
the actual column names it found — update `COLUMN_KEYWORDS` in
`src/load_bom.py` to match your file.

### 3. Merge, add features, split

```bash
python src/clean_merge.py
```

This merges demand + weather on timestamp, adds `day_of_week`,
`is_weekend`, `is_public_holiday` (Victorian holidays via the `holidays`
package), and `season`; reindexes to a complete half-hourly grid and
linearly interpolates gaps up to 2 hours (longer gaps are left as NaN and
reported, not silently filled); applies any `EXCLUDED_PERIODS`; and writes
a chronological 70/15/15 train/val/test split plus a `split_manifest.json`
recording the exact date cutoffs used, so every later experiment can
reference the same split.

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
python tests/test_fetch_aemo_resample.py
```

Covers the 5-minute → half-hourly resampling and region-filtering logic in
`fetch_aemo.py` using synthetic data, since it's the part most likely to
have a subtle bug and doesn't need network access to verify.

## Before Week 5

- [ ] Confirm `fetch_aemo.py` runs cleanly against the live site (NEMOSIS
      may prompt you to accept AEMO's terms/agree to caching on first run)
      and spot-check a few known dates (a public holiday, a heatwave day)
- [ ] Confirm the BOM export's actual column headers match `load_bom.py`
- [ ] Identify and log any anomaly windows (missing data, sensor outages)
      in `EXCLUDED_PERIODS`
- [ ] Sanity-check `split_manifest.json` cutoff dates look right
- [ ] Confirm `data/processed/` and `data/raw/nemosis_cache/` are *not* in
      git (see `.gitignore`) but are reproducible by anyone who clones the
      repo and runs the three scripts in order
