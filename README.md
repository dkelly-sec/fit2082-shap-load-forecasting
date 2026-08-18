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

- `START_DATE` / `END_DATE` — your 2-year window
- `BOM_STATION_NAME` / `BOM_STATION_ID` — your chosen weather station
- `EXCLUDED_PERIODS` — any anomaly windows you identify during EDA (e.g. a
  documented COVID-disruption range, sensor outages)

## Run

### 1. AEMO demand data

```bash
python src/fetch_aemo.py
```

This was written and unit-tested against synthetic AEMO-format data but
**has not been run against the live AEMO site** (its domain isn't reachable
from the environment this was built in). Before trusting a full pull:

- Run it once and manually open the first downloaded file to confirm the
  column mapping in `fetch_aemo.py` / `clean_merge.py` matches what AEMO
  actually returns — table schemas do drift between MMSDM releases.
- AEMO's "Current" NEMWEB directory only retains ~13 months of history. If
  `START_DATE` is older than that, you'll need a second pass against the
  MMS Data Model Archive (different URL structure) — see the docstring in
  `fetch_aemo.py`.
- AEMO migrated its NEMWEB base URL on 30 Apr 2026. If listing/downloads
  fail outright, check the current address on AEMO's site and update
  `NEMWEB_BASE_URL` in `config.py`.

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
python tests/test_aemo_cidf.py
```

Covers the AEMO row-dispatch (`C,`/`I,`/`D,`/`F,`) CSV parser using
synthetic data, since it's the fiddliest and most failure-prone part of the
pipeline and doesn't need network access to verify.

## Before Week 5

- [ ] Confirm `fetch_aemo.py` runs cleanly against the live site and spot-
      check a few known dates (a public holiday, a heatwave day) by hand
- [ ] Confirm the BOM export's actual column headers match `load_bom.py`
- [ ] Identify and log any anomaly windows (missing data, COVID period if
      in range) in `EXCLUDED_PERIODS`
- [ ] Sanity-check `split_manifest.json` cutoff dates look right
- [ ] Commit `data/processed/` outputs are *not* in git (see `.gitignore`)
      but are reproducible by anyone who clones the repo and runs the
      three scripts in order
