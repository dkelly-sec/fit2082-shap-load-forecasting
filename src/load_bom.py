"""
Load and standardise Bureau of Meteorology weather data.

BOM's Climate Data Online does not offer a simple free bulk-download API
for sub-daily station data -- you request/download it interactively:

    1. Go to http://www.bom.gov.au/climate/data/
       (or use the Weather Station Directory to find your station first:
       https://www.bom.gov.au/climate/data-services/station-data.shtml)
    2. Search for your chosen station (config.BOM_STATION_NAME /
       config.BOM_STATION_ID), select temperature + humidity, and the
       finest available sub-daily resolution.
    3. Set the date range to match config.START_DATE / config.END_DATE.
    4. Download the CSV and save it to the path in config.BOM_RAW_CSV
       (data/raw/bom_weather_raw.csv by default).

This script then loads that file, standardises column names/timestamp
format, and writes a cleaned version to data/interim/.

BOM's exported CSVs vary a bit in exact column naming depending on the
product, so this loader is deliberately tolerant: it looks for columns by
keyword rather than assuming an exact header, and will raise a clear error
telling you what it found if it can't confidently match temperature/
humidity/timestamp columns. Check the printed column mapping against your
actual file the first time you run it.
"""

from __future__ import annotations
import re
import sys

import pandas as pd

import config

# keyword -> standardised column name
COLUMN_KEYWORDS = {
    "timestamp": ["date", "time", "local"],
    "temperature": ["air temp", "temperature", "temp"],
    "humidity": ["relative humidity", "humidity", "rh"],
}


def _find_column(columns: list[str], keywords: list[str]) -> str | None:
    lowered = {c: c.lower() for c in columns}
    for kw in keywords:
        for original, low in lowered.items():
            if kw in low:
                return original
    return None


def load_bom_csv(path=None) -> pd.DataFrame:
    path = path or config.BOM_RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the BOM CSV manually first -- see the "
            "module docstring in load_bom.py for the exact steps -- and save it "
            "to that path (or pass a different path to load_bom_csv())."
        )

    raw = pd.read_csv(path)
    print(f"Loaded {path} with columns: {list(raw.columns)}")

    col_map = {}
    for std_name, keywords in COLUMN_KEYWORDS.items():
        found = _find_column(list(raw.columns), keywords)
        if found is None:
            raise ValueError(
                f"Could not find a column matching '{std_name}' (looked for "
                f"keywords {keywords}) in {list(raw.columns)}. Update "
                "COLUMN_KEYWORDS in load_bom.py to match your file's actual "
                "headers."
            )
        col_map[found] = std_name
        print(f"  mapped '{found}' -> '{std_name}'")

    df = raw[list(col_map.keys())].rename(columns=col_map)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)
    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        print(f"  warning: {n_bad} rows had unparseable timestamps and will be dropped")
    df = df.dropna(subset=["timestamp"])

    for col in ("temperature", "humidity"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")

    out_path = config.INTERIM_DIR / "bom_weather_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    try:
        load_bom_csv()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        sys.exit(1)
