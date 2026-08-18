"""
Unit test for aemo_cidf.parse_cidf using a small synthetic file that
mimics AEMO's row-dispatch format. No network access needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemo_cidf import parse_cidf  # noqa: E402

SAMPLE = (
    'C,NEMP.WORLD,ACTUAL_HH,AEMO,PUBLIC,2024/01/01,00:00:00,,,\n'
    'I,OPERATIONAL_DEMAND,ACTUAL_HH,1,SETTLEMENTDATE,REGIONID,OPERATIONAL_DEMAND\n'
    'D,OPERATIONAL_DEMAND,ACTUAL_HH,1,"2024/01/01 00:30:00",VIC1,5123.4\n'
    'D,OPERATIONAL_DEMAND,ACTUAL_HH,1,"2024/01/01 01:00:00",VIC1,5001.2\n'
    'D,OPERATIONAL_DEMAND,ACTUAL_HH,1,"2024/01/01 00:30:00",NSW1,7890.1\n'
    'I,SOME_OTHER_TABLE,OTHER,1,COL_A,COL_B\n'
    'D,SOME_OTHER_TABLE,OTHER,1,foo,bar\n'
    'C,END OF REPORT\n'
)


def test_parse_extracts_matching_table_only():
    df = parse_cidf(SAMPLE, table_name="ACTUAL_HH")
    assert len(df) == 3
    assert "REGIONID" in df.columns
    assert set(df["REGIONID"]) == {"VIC1", "NSW1"}


def test_parse_can_filter_region_after():
    df = parse_cidf(SAMPLE, table_name="ACTUAL_HH")
    vic = df[df["REGIONID"] == "VIC1"]
    assert len(vic) == 2
    assert vic["OPERATIONAL_DEMAND"].astype(float).tolist() == [5123.4, 5001.2]


def test_parse_ignores_unrelated_tables():
    df = parse_cidf(SAMPLE, table_name="ACTUAL_HH")
    assert "COL_A" not in df.columns


def test_parse_all_tables_when_no_filter():
    df = parse_cidf(SAMPLE, table_name=None)
    # both blocks come back concatenated with NaNs padded where columns differ
    assert len(df) >= 3


if __name__ == "__main__":
    test_parse_extracts_matching_table_only()
    test_parse_can_filter_region_after()
    test_parse_ignores_unrelated_tables()
    test_parse_all_tables_when_no_filter()
    print("All aemo_cidf tests passed.")
