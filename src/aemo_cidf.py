"""
Parser for AEMO's proprietary "row-dispatch" CSV format.

AEMO NEMWEB CSVs are not plain tabular CSVs. Each physical file can contain
several logical tables interleaved, and every row starts with a one-letter
record type:

    C,  comment / file header line (usually first and last line)
    I,  a column-header line, introducing a new table/schema block
    D,  a data row belonging to the most recently seen I, block
    F,  footer line (rare)

A single file can contain multiple I,/D,... blocks for different tables.
This module extracts just the block(s) matching a given table name (e.g.
"DEMANDOPERATIONALACTUAL" or "ACTUAL_HH") into a pandas DataFrame.

Reference: this quirk is well documented by third-party AEMO data tooling,
e.g. https://github.com/charlescoverdale/aemo
"""

from __future__ import annotations
import csv
import io
from typing import Optional
import pandas as pd


def parse_cidf(raw_text: str, table_name: Optional[str] = None) -> pd.DataFrame:
    """
    Parse AEMO C/I/D/F formatted text into a single DataFrame.

    Parameters
    ----------
    raw_text : the full decoded contents of one NEMWEB CSV file.
    table_name : if given, only rows whose I-record table name contains
        this substring (case-insensitive) are kept. If multiple tables
        match, they are concatenated (only sensible when they share a
        schema) -- for anything else, filter after the fact yourself.

    Returns
    -------
    pandas.DataFrame with columns taken from the matching I, row(s).
    """
    reader = csv.reader(io.StringIO(raw_text))

    current_columns: list[str] | None = None
    current_table: str | None = None
    frames: list[pd.DataFrame] = []
    rows_for_current_block: list[list[str]] = []

    def flush():
        if current_columns is not None and rows_for_current_block:
            df = pd.DataFrame(rows_for_current_block, columns=current_columns)
            frames.append(df)

    for row in reader:
        if not row:
            continue
        record_type = row[0].strip()

        if record_type == "I":
            # starting a new table block -- flush the previous one first
            flush()
            rows_for_current_block = []
            # AEMO I, rows look like: I,<REPORT>,<TABLE>,<VERSION>,<col1>,<col2>,...
            # The first 4 fields are metadata, not column names.
            current_table = row[2] if len(row) > 2 else None
            current_columns = row[4:] if len(row) > 4 else None
            # Only bother collecting this block if it matches the requested table
            if table_name is not None and (
                current_table is None or table_name.lower() not in current_table.lower()
            ):
                current_columns = None  # signal "skip this block"

        elif record_type == "D":
            if current_columns is not None:
                data_fields = row[4:]  # drop D,<report>,<table>,<version>
                # pad/truncate defensively in case of trailing commas etc.
                if len(data_fields) < len(current_columns):
                    data_fields = data_fields + [""] * (len(current_columns) - len(data_fields))
                elif len(data_fields) > len(current_columns):
                    data_fields = data_fields[: len(current_columns)]
                rows_for_current_block.append(data_fields)

        elif record_type in ("C", "F"):
            # comment/header or footer -- not data, ignore
            continue

    flush()

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    return result
