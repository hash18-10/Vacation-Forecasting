"""Tests for value normalization, file discovery, header mapping and preprocessing."""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType

import pandas as pd
import pytest

from vacation_forecasting.data_loader import (
    discover_timesheet_files,
    load_raw_timesheets,
    load_timesheet_file,
    read_workbook,
    resolve_columns,
)
from vacation_forecasting.normalization import (
    financial_year_end,
    fy_month_index,
    normalize_identifier_series,
    normalize_pag_series,
    parse_transaction_dates,
)
from vacation_forecasting.preprocessing import preprocess_timesheets


@pytest.mark.parametrize(
    ("calamine_available", "expected_engine"),
    [(True, "calamine"), (False, "pyxlsb")],
)
def test_xlsb_reader_selects_available_engine(
    monkeypatch, tmp_path, calamine_available: bool, expected_engine: str
) -> None:
    path = tmp_path / "timesheet.xlsb"
    path.touch()
    monkeypatch.setitem(
        sys.modules,
        "python_calamine",
        ModuleType("python_calamine") if calamine_available else None,
    )
    monkeypatch.setitem(sys.modules, "pyxlsb", ModuleType("pyxlsb"))
    calls = {}

    def fake_read_excel(*args, **kwargs):
        calls.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    read_workbook(path, "Timesheet", 1)

    assert calls == {"engine": expected_engine, "sheet_name": "Timesheet", "header": 1}


def test_xlsb_layout_is_detected_for_unrecognized_filename(monkeypatch, tmp_path, settings_for) -> None:
    path = tmp_path / "renamed-timesheet.xlsb"
    path.touch()
    monkeypatch.setitem(sys.modules, "python_calamine", ModuleType("python_calamine"))

    class FakeExcelFile:
        sheet_names = ["Cover", "Memberwise Timesheet"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def parse(self, sheet_name, header, nrows=None):
            columns = ["Employee ID", "Transaction Date", "PAG", "TRC Code", "Hours"]
            if nrows == 0:
                if sheet_name == "Cover" and header >= 2:
                    raise ValueError("Header row exceeds worksheet length")
                if sheet_name == "Memberwise Timesheet" and header == 1:
                    return pd.DataFrame(columns=columns)
                return pd.DataFrame(columns=["Cover page"])
            return pd.DataFrame([["001", "2025-01-31", "Solvay", "105", 9]], columns=columns)

    monkeypatch.setattr(pd, "ExcelFile", lambda *args, **kwargs: FakeExcelFile())

    frame, mapping = load_timesheet_file(path, replace(settings_for, sheet_name="Cover"))

    assert mapping["member_id"] == "Employee ID"
    assert frame["source_sheet"].iloc[0] == "Memberwise Timesheet"
    assert frame["source_row"].iloc[0] == 3


def test_pag_aliases_map_to_canonical_name() -> None:
    values = pd.Series(["Solvay SA", "solvay s.a.", "SOLVAY GROUP", "Other Co", None])
    result = normalize_pag_series(values)
    assert result.tolist()[:3] == ["Solvay", "Solvay", "Solvay"]
    assert result.iloc[3] == "Other Co"
    assert pd.isna(result.iloc[4])


def test_preprocess_preserves_order_for_combined_rejection_reasons(settings_for) -> None:
    raw = pd.DataFrame(
        {
            "member_id": [None],
            "transaction_date": [None],
            "pag": [None],
            "trc_code": [None],
            "hours": ["not numeric"],
        }
    )

    result = preprocess_timesheets(raw, settings_for)

    assert result.rejected["rejection_reason"].iloc[0] == (
        "missing_date;missing_member_id;missing_pag;missing_trc_code;invalid_hours"
    )


def test_trc_code_numeric_and_text_compare_equal() -> None:
    values = pd.Series([105, 105.0, " 105 ", "105.0", None])
    result = normalize_identifier_series(values)
    assert result.tolist()[:4] == ["105", "105", "105", "105"]
    assert result.iloc[4] is None or pd.isna(result.iloc[4])


def test_excel_serial_dates_parse_correctly() -> None:
    result = parse_transaction_dates(pd.Series([45292, "2024-01-15", "not a date"]))
    assert result.iloc[0] == pd.Timestamp("2024-01-01")
    assert result.iloc[1] == pd.Timestamp("2024-01-15")
    assert pd.isna(result.iloc[2])


@pytest.mark.parametrize(
    ("date", "fy_end", "month_index"),
    [
        ("2023-10-01", 2024, 1),
        ("2024-09-30", 2024, 12),
        ("2024-01-15", 2024, 4),
    ],
)
def test_financial_year_is_october_to_september(date: str, fy_end: int, month_index: int) -> None:
    dates = pd.Series([pd.Timestamp(date)])
    assert financial_year_end(dates).iloc[0] == fy_end
    assert fy_month_index(dates).iloc[0] == month_index


def test_resolve_columns_accepts_variant_headers() -> None:
    mapping = resolve_columns(
        ["Emp ID", "Trans Date", "Account Name", "TRC", "Hrs"], "file.csv"
    )
    assert mapping["member_id"] == "Emp ID"
    assert mapping["pag"] == "Account Name"


def test_resolve_columns_reports_missing_required() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        resolve_columns(["Employee ID", "Date"], "bad.csv")


def test_discovery_skips_holiday_file_and_lock_files(settings_for) -> None:
    raw = settings_for.raw_dir
    (raw / "holidays.csv").write_text("date,holiday_name,location\n")
    (raw / "~$FY2024.xlsb").write_text("lock")
    (raw / "FY2024.csv").write_text("x")
    files = discover_timesheet_files(raw, exclude=settings_for.holidays_csv)
    assert [path.name for path in files] == ["FY2024.csv"]


def test_load_and_preprocess_csv_keeps_source_and_rejects_bad_rows(
    settings_for, synthetic_raw: pd.DataFrame
) -> None:
    raw = synthetic_raw.copy()
    raw.loc[0, "Hours"] = None
    raw.loc[1, "Employee ID"] = None
    raw.to_csv(settings_for.raw_dir / "FY2024.csv", index=False)
    (settings_for.raw_dir / "holidays.csv").write_text("date,holiday_name,location\n")

    loaded = load_raw_timesheets(settings_for)
    assert len(loaded) == len(synthetic_raw)
    assert set(loaded["source_file"]) == {"FY2024.csv"}

    result = preprocess_timesheets(loaded, settings_for)
    assert len(result.rejected) == 2
    assert set(result.rejected["rejection_reason"]) == {"invalid_hours", "missing_member_id"}
    assert len(result.clean) == len(synthetic_raw) - 2
    assert set(result.clean["pag"]) == {"Solvay"}
    assert "month" in result.clean.columns
    assert "financial_year_label" in result.clean.columns
