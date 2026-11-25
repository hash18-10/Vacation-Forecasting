"""Tests for the holiday calendar schema, date parsing, location scoping and deduplication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vacation_forecasting.holiday_analysis import (
    applicable_holidays,
    holiday_counts_by_month,
    load_holidays,
)

REAL_CALENDAR = Path(__file__).resolve().parents[1] / "data" / "raw" / "holidays.csv"
HEADER = "date,holiday_name,holiday_type,location,source_file,year\n"


def write_calendar(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "holidays.csv"
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_real_calendar_has_expected_scope() -> None:
    holidays = load_holidays(REAL_CALENDAR)
    assert set(holidays["year"].unique()) == {2023, 2024, 2025, 2026}
    assert not holidays["holiday_name"].str.contains("Boxing", case=False).any()
    christmas_2025 = holidays.loc[holidays["holiday_name"].eq("Christmas Holiday") & holidays["year"].eq(2025)]
    assert len(christmas_2025) == 1


def test_real_calendar_keeps_both_holiday_types() -> None:
    holidays = load_holidays(REAL_CALENDAR)
    assert set(holidays["holiday_type"]) == {"Mandatory", "Restricted/Optional"}


def test_india_wide_holiday_counts_for_any_location(tmp_path: Path) -> None:
    path = write_calendar(tmp_path, ["2025-01-26,Republic Day,Mandatory,India,test.pdf,2025"])
    holidays = load_holidays(path)
    months = pd.DatetimeIndex(["2025-01-01"])
    assert holiday_counts_by_month(holidays, months, ["Hyderabad"]).iloc[0] == 1
    assert holiday_counts_by_month(holidays, months, []).iloc[0] == 1


def test_location_specific_holiday_counts_only_for_its_location(tmp_path: Path) -> None:
    path = write_calendar(
        tmp_path, ["2025-09-16,Anant Chaturdashi (Ganesh Visarjan),Restricted/Optional,Mumbai,test.pdf,2025"]
    )
    holidays = load_holidays(path)
    months = pd.DatetimeIndex(["2025-09-01"])
    assert holiday_counts_by_month(holidays, months, ["Mumbai"]).iloc[0] == 1
    assert holiday_counts_by_month(holidays, months, ["Hyderabad"]).iloc[0] == 0


def test_mandatory_and_restricted_types_are_preserved(tmp_path: Path) -> None:
    path = write_calendar(
        tmp_path,
        [
            "2025-08-15,Independence Day,Mandatory,India,test.pdf,2025",
            "2025-10-21,Diwali / Deepavali,Restricted/Optional,India,test.pdf,2025",
        ],
    )
    holidays = load_holidays(path).set_index("holiday_name")
    assert holidays.loc["Independence Day", "holiday_type"] == "Mandatory"
    assert holidays.loc["Diwali / Deepavali", "holiday_type"] == "Restricted/Optional"


def test_exact_duplicate_rows_are_removed(tmp_path: Path) -> None:
    row = "2025-08-15,Independence Day,Mandatory,India,test.pdf,2025"
    holidays = load_holidays(write_calendar(tmp_path, [row, row]))
    assert len(holidays) == 1


def test_same_holiday_for_india_and_location_is_counted_once(tmp_path: Path) -> None:
    path = write_calendar(
        tmp_path,
        [
            "2025-09-16,Anant Chaturdashi,Restricted/Optional,India,test.pdf,2025",
            "2025-09-16,Anant Chaturdashi,Restricted/Optional,Mumbai,test.pdf,2025",
        ],
    )
    holidays = load_holidays(path)
    assert len(applicable_holidays(holidays, ["Mumbai"])) == 1


def test_invalid_date_format_raises(tmp_path: Path) -> None:
    path = write_calendar(tmp_path, ["25-Dec-25,Christmas Holiday,Restricted/Optional,India,test.pdf,2025"])
    with pytest.raises(ValueError):
        load_holidays(path)


def test_year_must_match_date(tmp_path: Path) -> None:
    path = write_calendar(tmp_path, ["2025-12-25,Christmas Holiday,Restricted/Optional,India,test.pdf,2024"])
    with pytest.raises(ValueError, match="year"):
        load_holidays(path)


def test_unknown_holiday_type_raises(tmp_path: Path) -> None:
    path = write_calendar(tmp_path, ["2025-12-25,Christmas Holiday,Optional,India,test.pdf,2025"])
    with pytest.raises(ValueError, match="holiday_type"):
        load_holidays(path)
