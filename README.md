# FIT2082 Data Pipeline — Week 4

Data acquisition and cleaning pipeline for *Sampling Sensitivity of Global
SHAP Feature Rankings in Short-Term Electricity Demand Forecasting*.

Produces a clean, half-hourly, chronologically-split dataset combining AEMO
Victorian electricity demand, BOM daily temperature (min/max/mean), and
derived calendar features (day-of-week, public holiday flag, season) —
ready to hand to the LightGBM forecaster in Week 5–6.

## Status: Week 4 complete ✅

Pipeline has been run end-to-end against real data:

| Stage | Result |
|---|---|
| AEMO demand (`fetch_aemo.py`) | 35,088 half-hourly rows, 2024-01-01 → 2025-12-31, region VIC1, 0 gaps |
| BOM weather (`load_bom.py`) | 731 daily rows, station 086338 (Melbourne, Olympic Park); 2 days had an incomplete min/max pair, filled from the nearest complete day rather than derived from a single reading |
| Merge + split (`clean_merge.py`) | 35,088 rows merged with no loss; chronological split: train 24,561 / val 5,263 / test 5,264 (~70/15/15) |

Next up: Week 5–6, EDA and LightGBM training via time-series CV.

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

Uses [NEMOSIS](https://github.com/UNSW-CEEM/NEMOSIS), a maintained Python
package for downloading historical AEMO/NEM data. Pulls `DISPATCHREGIONSUM`
(5-minute resolution) and resamples to half-hourly by taking the mean.
First run can be slow while NEMOSIS builds its local cache under
`data/raw/nemosis_cache/`.

### 2. BOM weather data

BOM's Climate Data Online only offers **daily** temperature for free
download (no free half-hourly/hourly station archive over a multi-year
span — that requires a paid Data Services request). Humidity generally
isn't freely available either, so this project uses daily min/max/mean
temperature only — worth a line in your report noting the substitution
from the original half-hourly weather plan.

To get the data:

1. Go to <http://www.bom.gov.au/climate/data/>
2. Text search → Data about: **Temperature** → Type of data: **Daily** →
   **Minimum temperature**. Search your station (e.g. "Melbourne") and
   select it (e.g. `086338 Melbourne (Olympic Park)`).
3. Click **Get Data** → on the page that opens, click **"1 year of data"**
   (repeat per year) or **"All years of data"** (one click, then the
   pipeline filters to your configured date range automatically).
4. This downloads a zip that your browser/OS likely auto-extracts into a
   folder like `IDCJAC0011_086338_2024/`. Move that folder as-is into
   `data/raw/`.
5. Repeat steps 2–4 with **Maximum temperature** instead of Minimum.

You should end up with folders like this inside `data/raw/`:

```
data/raw/IDCJAC0011_086338_2024/   (minimum temperature)
data/raw/IDCJAC0011_086338_2025/
data/raw/IDCJAC0010_086338_2024/   (maximum temperature)
data/raw/IDCJAC0010_086338_2025/
```

`load_bom.py` finds these automatically by folder-name pattern — you don't
need to list exact filenames anywhere. Then run:

```bash
python src/load_bom.py
```

This combines all matching min/max files, computes a daily mean
temperature, and filters to your configured date range.

### 3. Merge, add features, split

```bash
python src/clean_merge.py
```

This:
- Reindexes demand to a complete half-hourly grid and linearly
  interpolates demand gaps up to 2 hours (longer gaps are left as NaN and
  reported, not silently filled)
- Adds `day_of_week`, `is_weekend`, `is_public_holiday` (Victorian
  holidays via the `holidays` package), and `season`
- Merges in daily weather, broadcasting each day's temperature values
  across that day's 48 half-hourly demand rows
- Applies any `EXCLUDED_PERIODS`
- Writes a chronological 70/15/15 train/val/test split plus a
  `split_manifest.json` recording the exact date cutoffs used

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
python tests/test_bom_merge.py
```

Cover the 5-minute→half-hourly resampling, BOM CSV parsing, the
missing-reading-does-not-silently-average-to-one-value fix, and the
daily-to-half-hourly broadcast merge — all using synthetic data, no
network access needed.

## Before Week 5

- [x] `fetch_aemo.py` run against live AEMO data (35,088 rows — matches
      exactly 2 years of half-hourly data, leap year included)
- [x] `load_bom.py` run against live BOM data (731 days). 2 days had an
      incomplete min/max pair — `temperature` for those is filled from the
      nearest complete day rather than silently derived from a single
      reading (see the comment in `load_bom_daily()` if you want to know
      why that distinction matters)
- [x] `clean_merge.py` run end-to-end — 35,088 rows merged with no loss,
      chronological 70/15/15 split written, `split_manifest.json` produced
- [ ] Identify and log any anomaly windows (missing data, sensor outages)
      in `EXCLUDED_PERIODS` — none identified yet; revisit during EDA
- [ ] Sanity-check `split_manifest.json` cutoff dates look right
- [x] Note the temperature-only (no humidity, daily not half-hourly)
      weather scope in your report/EDA as a documented deviation from the
      original proposal, with the reasoning (BOM's free-tier limitations)
      — see the "BOM weather data" section above, reuse that wording
- [x] Confirmed `data/processed/` and `data/raw/nemosis_cache/` are *not*
      in git (see `.gitignore`) but are reproducible by anyone who clones
      the repo and runs the three scripts in order
