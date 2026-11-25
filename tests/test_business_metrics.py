"""Tests for the business metrics: working days, person-days, vacation % and availability %."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from vacation_forecasting.business_metrics import (
    PERSON_MONTH_COLUMNS,
    check_monthly_table,
    days_from_hours,
    monthly_business_table,
    partial_months,
    person_months_from_forecast,
    person_months_from_rows,
    safe_pct,
    summarise_period,
    weekday_counts,
    working_days_by_location,
)
from vacation_forecasting.config import INDIA_LOCATION
from vacation_forecasting.vacation_analysis import available_pags, select_pag

MARCH_2025 = pd.DatetimeIndex([pd.Timestamp("2025-03-01")])
FEB_2025 = pd.DatetimeIndex([pd.Timestamp("2025-02-01")])
NO_HOLIDAYS = pd.DataFrame()


def _holidays(*dates: str, location: str = INDIA_LOCATION) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(list(dates)),
            "holiday_name": ["Test holiday"] * len(dates),
            "holiday_type": ["Mandatory"] * len(dates),
            "location": [location] * len(dates),
            "source_file": ["test"] * len(dates),
            "year": [pd.Timestamp(d).year for d in dates],
        }
    )


def test_weekday_counts_exclude_weekends():
    assert weekday_counts(MARCH_2025).iloc[0] == 21  # March 2025: 31 days, starts on a Saturday
    assert weekday_counts(FEB_2025).iloc[0] == 20  # February 2025: 28 days, starts on a Saturday


def test_holiday_exclusion_is_off_by_default_and_removes_only_weekday_holidays():
    holidays = _holidays("2025-03-03", "2025-03-01")  # Monday (weekday) and Saturday (weekend)
    off = working_days_by_location(MARCH_2025, [INDIA_LOCATION], holidays, exclude_holidays=False)
    on = working_days_by_location(MARCH_2025, [INDIA_LOCATION], holidays, exclude_holidays=True)
    assert off.iloc[0, 0] == 21
    assert on.iloc[0, 0] == 20  # the Saturday holiday does not reduce the count


def test_days_from_hours_uses_divisor_and_rejects_non_positive():
    assert days_from_hours(9.0, 9.0) == 1.0
    assert days_from_hours(4.5, 9.0) == 0.5
    with pytest.raises(ValueError):
        days_from_hours(9.0, 0.0)


def test_safe_pct_returns_nan_for_zero_denominator():
    assert math.isnan(safe_pct(5.0, 0.0))
    assert safe_pct(2.0, 40.0) == pytest.approx(5.0)


def test_person_months_from_rows_counts_only_vacation_hours():
    rows = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-03-01"] * 3),
            "member_id": ["A", "A", "B"],
            "location": [INDIA_LOCATION, INDIA_LOCATION, INDIA_LOCATION],
            "hours": [9.0, 4.5, 8.0],
            "is_vacation": [True, True, False],
        }
    )
    out = person_months_from_rows(rows, hours_per_day=9.0)
    assert list(out.columns) == PERSON_MONTH_COLUMNS
    out = out.set_index("member_id")
    assert out.loc["A", "vacation_days"] == pytest.approx(1.5)
    assert out.loc["B", "vacation_days"] == pytest.approx(0.0)


def test_person_months_from_forecast_converts_hours_to_days():
    employees = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "member_id": ["A", "B"],
            "location": [INDIA_LOCATION, INDIA_LOCATION],
            "forecast": [9.0, 4.5],
        }
    )
    out = person_months_from_forecast(employees, hours_per_day=9.0)
    assert out["vacation_days"].tolist() == pytest.approx([1.0, 0.5])


def test_monthly_table_percentages_sum_to_100_and_average_per_employee():
    person_months = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "member_id": ["A", "B"],
            "location": [INDIA_LOCATION, INDIA_LOCATION],
            "vacation_days": [2.0, 0.0],
        }
    )
    table = monthly_business_table(person_months, MARCH_2025, NO_HOLIDAYS, exclude_holidays=False)
    row = table.iloc[0]
    assert row["employees"] == 2
    assert row["available_person_days"] == pytest.approx(42.0)  # 2 employees x 21 weekdays
    assert row["vacation_days"] == pytest.approx(2.0)
    assert row["vacation_pct"] == pytest.approx(2.0 / 42.0 * 100)
    assert row["vacation_pct"] + row["availability_pct"] == pytest.approx(100.0)
    assert row["avg_vacation_days_per_employee"] == pytest.approx(1.0)
    assert check_monthly_table(table) == []


def test_holiday_flag_reduces_available_person_days():
    person_months = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-03-01"]),
            "member_id": ["A"],
            "location": [INDIA_LOCATION],
            "vacation_days": [0.0],
        }
    )
    holidays = _holidays("2025-03-03")
    off = monthly_business_table(person_months, MARCH_2025, holidays, exclude_holidays=False)
    on = monthly_business_table(person_months, MARCH_2025, holidays, exclude_holidays=True)
    assert off.iloc[0]["available_person_days"] == pytest.approx(21.0)
    assert on.iloc[0]["available_person_days"] == pytest.approx(20.0)


def test_zero_employees_gives_nan_percentages_not_errors():
    empty = pd.DataFrame(columns=PERSON_MONTH_COLUMNS)
    table = monthly_business_table(empty, MARCH_2025, NO_HOLIDAYS, exclude_holidays=False)
    row = table.iloc[0]
    assert row["employees"] == 0
    assert math.isnan(row["vacation_pct"])
    assert math.isnan(row["availability_pct"])
    assert math.isnan(row["avg_vacation_days_per_employee"])


def test_summarise_period_uses_summed_days_over_summed_person_days():
    monthly = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "employees": [10, 1],
            "weekdays": [20, 5],
            "available_person_days": [40.0, 10.0],
            "vacation_days": [4.0, 5.0],
        }
    )
    summary = summarise_period(monthly, unique_employees=10)
    # Summed days / summed person-days = 9 / 50 = 18%. A mean of monthly percentages would give 30%.
    assert summary["vacation_pct"] == pytest.approx(18.0)
    assert summary["availability_pct"] == pytest.approx(82.0)
    assert summary["avg_vacation_days_per_employee"] == pytest.approx(0.9)


def test_partial_months_flags_incomplete_ends_only():
    partial = partial_months(pd.Series(pd.to_datetime(["2025-03-15", "2025-05-20"])))
    assert partial == [pd.Timestamp("2025-03-01"), pd.Timestamp("2025-05-01")]
    full = partial_months(pd.Series(pd.to_datetime(["2025-03-01", "2025-05-31"])))
    assert full == []


def test_check_monthly_table_flags_inconsistent_percentages():
    table = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-03-01"]),
            "employees": [2],
            "weekdays": [21],
            "available_person_days": [42.0],
            "vacation_days": [2.0],
            "vacation_pct": [5.0],
            "availability_pct": [90.0],
            "avg_vacation_days_per_employee": [1.0],
        }
    )
    issues = check_monthly_table(table)
    assert any("do not sum to 100" in issue for issue in issues)


def test_ten_employee_example_from_requirement():
    monthly = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-03-01"]),
            "employees": [10],
            "weekdays": [10],
            "available_person_days": [100.0],
            "vacation_days": [10.0],
        }
    )
    summary = summarise_period(monthly, unique_employees=10)
    assert summary["vacation_pct"] == pytest.approx(10.0)
    assert summary["availability_pct"] == pytest.approx(90.0)
    assert summary["avg_vacation_days_per_employee"] == pytest.approx(1.0)


def test_monthly_forecast_percentage_from_forecast_hours():
    employees = pd.DataFrame(
        {
            "month": pd.to_datetime(["2025-03-01", "2025-03-01"]),
            "member_id": ["A", "B"],
            "location": [INDIA_LOCATION, INDIA_LOCATION],
            "forecast": [45.0, 0.0],  # 45 h / 9 = 5 vacation days
        }
    )
    person_months = person_months_from_forecast(employees, hours_per_day=9.0)
    table = monthly_business_table(person_months, MARCH_2025, NO_HOLIDAYS, exclude_holidays=False)
    row = table.iloc[0]
    assert row["vacation_days"] == pytest.approx(5.0)
    assert row["vacation_pct"] == pytest.approx(5.0 / 42.0 * 100)
    assert row["vacation_pct"] + row["availability_pct"] == pytest.approx(100.0)


def test_pag_selection_uses_only_the_selected_pag():
    rows = pd.DataFrame({"pag": ["Solvay", "Solvay", "Other"], "member_id": ["A", "B", "C"]})
    assert available_pags(rows) == ["Other", "Solvay"]
    solvay = select_pag(rows, "Solvay")
    assert solvay["member_id"].tolist() == ["A", "B"]
    with pytest.raises(ValueError):
        select_pag(rows, "Missing PAG")


def test_partial_last_month_is_flagged_but_full_months_are_not():
    dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-02-28", "2025-03-10"]))
    assert partial_months(dates) == [pd.Timestamp("2025-03-01")]
