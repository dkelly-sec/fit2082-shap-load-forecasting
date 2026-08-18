"""
Fetch AEMO NEMWEB half-hourly operational demand data for a single region
and date range, and save a tidy CSV to data/raw/.

Usage
-----
    python src/fetch_aemo.py

What this does
---------------
1. Lists the NEMWEB directory for half-hourly operational demand reports.
2. Downloads the zip files covering the configured date range (config.py).
3. Unzips and parses each file with aemo_cidf.parse_cidf().
4. Filters to the configured REGION.
5. Concatenates everything and writes data/raw/aemo_demand_raw.csv.

Important notes
----------------
* AEMO NEMWEB directory layout and retention policy can change without
  notice, and the base URL migrated on 30 Apr 2026 (the old endpoint was
  decommissioned 7 Apr 2026). If listing/downloads start failing, check
  https://www.aemo.com.au/energy-systems/electricity/national-electricity-
  market-nem/data-nem/market-data-nemweb for the current address and
  update NEMWEB_BASE_URL / the report paths in config.py.
* NEMWEB's "Current" directory typically only retains ~13 months of data.
  For anything older, AEMO's data is instead in the MMS Data Model
  Archive, which has a different structure (monthly zips of zips) --
  you will likely need a second pass with a similar approach if your
  START_DATE is more than ~13 months back from today.
* This script does not run in this environment (AEMO's domain is not on
  the sandbox's network allowlist) -- it has been written and reviewed
  for correctness but not executed end-to-end. Run it locally / on your
  own machine, and sanity-check the first few downloaded files by hand
  before trusting the full pull.
"""

from __future__ import annotations
import io
import re
import zipfile
from datetime import date
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config
from aemo_cidf import parse_cidf

TABLE_NAME = "ACTUAL_HH"  # matches operational demand half-hourly actual table


def list_report_files(base_url: str, report_path: str) -> list[str]:
    """Return absolute URLs of all files linked from a NEMWEB directory listing."""
    listing_url = urljoin(base_url, report_path)
    resp = requests.get(listing_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]
    # keep only zip/csv report files, drop navigation links like "../"
    files = [h for h in hrefs if re.search(r"\.(zip|csv)$", h, re.IGNORECASE)]
    return [urljoin(listing_url, f) for f in files]


def date_from_filename(url: str) -> date | None:
    """
    Best-effort extraction of a YYYYMMDD date from an AEMO filename so we
    can filter the file list down to the configured date range before
    downloading everything.
    """
    m = re.search(r"(20\d{6})", url)
    if not m:
        return None
    s = m.group(1)
    try:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except ValueError:
        return None


def download_and_parse(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    if url.lower().endswith(".zip"):
        frames = []
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                with zf.open(name) as f:
                    text = f.read().decode("utf-8", errors="replace")
                frames.append(parse_cidf(text, table_name=TABLE_NAME))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    else:
        text = resp.content.decode("utf-8", errors="replace")
        return parse_cidf(text, table_name=TABLE_NAME)


def main():
    print(f"Listing NEMWEB current reports at {config.NEMWEB_CURRENT_REPORT_PATH} ...")
    all_files = list_report_files(config.NEMWEB_BASE_URL, config.NEMWEB_CURRENT_REPORT_PATH)
    print(f"Found {len(all_files)} candidate files.")

    in_range = []
    for url in all_files:
        d = date_from_filename(url)
        if d is not None and config.START_DATE <= d <= config.END_DATE:
            in_range.append(url)
    print(f"{len(in_range)} files fall inside {config.START_DATE} .. {config.END_DATE}.")

    if not in_range:
        print(
            "No files matched the configured date range in the 'Current' directory. "
            "If your START_DATE is more than ~13 months old, you likely need to pull "
            "from the MMS Data Model Archive instead -- see the module docstring."
        )

    frames = []
    for i, url in enumerate(in_range, 1):
        print(f"[{i}/{len(in_range)}] downloading {url}")
        try:
            df = download_and_parse(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")
            continue
        if df.empty:
            continue
        if "REGIONID" in df.columns:
            df = df[df["REGIONID"] == config.REGION]
        frames.append(df)

    if not frames:
        raise SystemExit("No data collected -- check network access and NEMWEB paths.")

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates()

    out_path = config.RAW_DIR / "aemo_demand_raw.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
