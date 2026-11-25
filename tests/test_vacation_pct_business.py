"""Business calculation for the Solvay KPIs: Vacation %, Availability % and average vacation days per employee."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vacation_forecasting.business_metrics import (
    check_monthly_table,
    monthly_business_table,
    partial_months,
    summarise_period,
    weekday_counts,
)
from vacation_forecasting.employee_forecasting import seasonal_naive_future

NO_HOLIDAYS = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "location": pd.Series(dtype="object")})


def _person_months(members: list[str], month: str, vacation_days: float | list[float]) -> pd.DataFrame:
    """One row per member for one month. `vacation_days` is a scalar or one value per member."""
    values = [vacation_days] * len(members) if np.isscalar(vacation_days) else list(vacation_days)
    return pd.DataFrame(
        {
            "month": pd.Timestamp(month),
            "member_id": members,
            "location": "DE",
            "vacation_days": values,
        }
    )


def _monthly(members: list[str], months: list[str], vacation_days: float) -> pd.DataFrame:
    frames = [_person_months(members, month, vacation_days) for month in months]
    pm = pd.concat(frames, ignore_index=True)
    return monthly_business_table(pm, pd.DatetimeIndex(months), NO_HOLIDAYS, exclude_holidays=False)


def test_ten_employees_ten_working_days_ten_vacation_days_is_ten_percent():
    monthly = pd.DataFrame(
        {
            "month": [pd.Timestamp("2025-01-01")],
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


def test_weekends_are_excluded_from_working_days():
    # February 2025 starts on a Saturday and has 20 Monday to Friday days.
    assert weekday_counts(pd.DatetimeIndex(["2025-02-01"])).iat[0] == 20

    monthly = _monthly(["E1"], ["2025-02-01"], vacation_days=0.0)
    assert monthly["weekdays"].iat[0] == 20
    assert monthly["available_person_days"].iat[0] == 20


@pytest.mark.parametrize("employee_count", [1, 3, 7])
def test_available_person_days_scale_with_employee_count(employee_count):
    members = [f"E{i}" for i in range(employee_count)]
    monthly = _monthly(members, ["2025-01-01"], vacation_days=1.0)

    weekdays = monthly["weekdays"].iat[0]
    assert monthly["employees"].iat[0] == employee_count
    assert monthly["available_person_days"].iat[0] == employee_count * weekdays
    assert monthly["vacation_pct"].iat[0] == pytest.approx(100.0 / (employee_count * weekdays) * employee_count)


@pytest.mark.parametrize("months", [3, 6, 12])
def test_period_percent_uses_summed_days_not_average_of_monthly_percentages(months):
    month_list = [str(pd.Timestamp("2025-01-01") + pd.DateOffset(months=i))[:10] for i in range(months)]
    members = ["E1", "E2", "E3", "E4"]
    # Different vacation volumes per month, so an average of monthly % would differ from the summed ratio.
    frames = []
    for i, month in enumerate(month_list):
        frames.append(_person_months(members, month, vacation_days=float(i)))
    pm = pd.concat(frames, ignore_index=True)
    monthly = monthly_business_table(pm, pd.DatetimeIndex(month_list), NO_HOLIDAYS, exclude_holidays=False)

    summary = summarise_period(monthly, unique_employees=len(members))

    assert summary["months"] == months
    expected = monthly["vacation_days"].sum() / monthly["available_person_days"].sum() * 100
    assert summary["vacation_pct"] == pytest.approx(expected)
    assert summary["availability_pct"] == pytest.approx(100 - expected)


def test_partial_first_month_is_flagged():
    dates = pd.Series(pd.to_datetime(["2024-10-15", "2024-11-03", "2025-08-28"]))
    partial = partial_months(dates)

    assert pd.Timestamp("2024-10-01") in partial
    assert pd.Timestamp("2025-08-01") in partial
    assert pd.Timestamp("2024-11-01") not in partial


def test_zero_vacation_gives_zero_percent_and_full_availability():
    monthly = _monthly(["E1", "E2"], ["2025-03-01"], vacation_days=0.0)
    summary = summarise_period(monthly, unique_employees=2)

    assert summary["vacation_pct"] == pytest.approx(0.0)
    assert summary["availability_pct"] == pytest.approx(100.0)
    assert summary["avg_vacation_days_per_employee"] == pytest.approx(0.0)
    assert check_monthly_table(monthly) == []


def test_member_with_no_history_rows_is_not_counted():
    # E3 has no timesheet row in this month, so it adds no employees and no available person-days.
    monthly = _monthly(["E1", "E2"], ["2025-01-01"], vacation_days=2.0)
    weekdays = monthly["weekdays"].iat[0]

    assert monthly["employees"].iat[0] == 2
    assert monthly["available_person_days"].iat[0] == 2 * weekdays
    assert monthly["avg_vacation_days_per_employee"].iat[0] == pytest.approx(2.0)


def test_month_with_no_employees_gives_nan_percentages():
    empty = pd.DataFrame(columns=["month", "member_id", "location", "vacation_days"])
    monthly = monthly_business_table(empty, pd.DatetimeIndex(["2025-01-01"]), NO_HOLIDAYS, exclude_holidays=False)

    assert monthly["employees"].iat[0] == 0
    assert np.isnan(monthly["vacation_pct"].iat[0])
    assert np.isnan(monthly["avg_vacation_days_per_employee"].iat[0])


def test_seasonal_naive_forecast_is_zero_for_member_without_last_year_row():
    # A has a row in Jan 2024 (9 h = 1 day). B has no Jan 2024 row, so the forecast for Jan 2025 is 0.
    history_months = pd.date_range("2024-01-01", periods=12, freq="MS")
    rows = []
    for month in history_months:
        rows.append({"member_id": "A", "location": "DE", "month": month, "vacation_hours": 9.0 if month.month == 1 else 0.0})
        if month.month != 1:
            rows.append({"member_id": "B", "location": "DE", "month": month, "vacation_hours": 0.0})
    frame = pd.DataFrame(rows)
    frame["pag"] = "Solvay"
    frame["vacation_occurrences"] = (frame["vacation_hours"] > 0).astype(float)
    frame["actual_event"] = (frame["vacation_hours"] > 0).astype(float)
    frame["is_future"] = False

    future = pd.DataFrame(
        [
            {"member_id": "A", "location": "DE", "month": pd.Timestamp("2025-01-01")},
            {"member_id": "B", "location": "DE", "month": pd.Timestamp("2025-01-01")},
        ]
    )
    future["pag"] = "Solvay"
    future["vacation_hours"] = np.nan
    future["vacation_occurrences"] = np.nan
    future["actual_event"] = np.nan
    future["is_future"] = True
    combined = pd.concat([frame, future], ignore_index=True)

    result = seasonal_naive_future(combined).set_index("member_id")

    assert result.loc["A", "forecast"] == pytest.approx(9.0)
    assert result.loc["B", "forecast"] == pytest.approx(0.0)


def test_seasonal_naive_rejects_horizon_beyond_twelve_months():
    # 12 history months and 13 future months: one more than the seasonal lookup can cover from actual history.
    months = pd.date_range("2024-01-01", periods=25, freq="MS")
    frame = pd.DataFrame(
        {
            "member_id": ["A"] * 25,
            "location": ["DE"] * 25,
            "month": months,
            "vacation_hours": [0.0] * 25,
            "vacation_occurrences": [0.0] * 25,
            "actual_event": [0.0] * 25,
            "pag": ["Solvay"] * 25,
            "is_future": [False] * 12 + [True] * 13,
        }
    )
    with pytest.raises(ValueError):
        seasonal_naive_future(frame)
