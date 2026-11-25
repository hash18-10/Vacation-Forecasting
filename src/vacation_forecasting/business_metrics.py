"""Business metrics per PAG and month: vacation days, working person-days, vacation % and availability %.

Pure functions only: no file access and no Streamlit.

Definitions
- Vacation day = vacation hours / hours_per_vacation_day (9 h for a full day; half days are 4.5 h).
- Working day = Monday to Friday. Weekends are never counted.
- With exclude_public_holidays=True, weekday public holidays that apply to the employee's location (and
  India-wide holidays) are also removed. With False (default) every weekday is a working day.
- Employees: historical = distinct members with a timesheet row in the month; forecast = members the model forecasts.
- Available person-days = sum over employees of their working days.
- Vacation % = vacation days / available person-days x 100. Availability % = 100 - vacation %.
- Average vacation days per employee = vacation days / employees.
Where a denominator is zero the result is NaN, never an error.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from vacation_forecasting.config import UNKNOWN_LOCATION
from vacation_forecasting.holiday_analysis import applicable_holidays

PERSON_MONTH_COLUMNS = ["month", "member_id", "location", "vacation_days"]
MONTHLY_COLUMNS = [
    "month",
    "employees",
    "weekdays",
    "available_person_days",
    "vacation_days",
    "vacation_pct",
    "availability_pct",
    "avg_vacation_days_per_employee",
]


def days_from_hours(hours: pd.Series | float, hours_per_day: float) -> pd.Series | float:
    """Convert hours to vacation days. The divisor must be positive."""
    if not hours_per_day > 0:
        raise ValueError(f"hours_per_day must be positive, got {hours_per_day}")
    return hours / hours_per_day


def safe_pct(numerator: float, denominator: float) -> float:
    """numerator / denominator x 100, or NaN when the denominator is zero, negative or missing."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator) / float(denominator) * 100.0


def weekday_counts(months: pd.DatetimeIndex) -> pd.Series:
    """Monday to Friday days in each month. Weekends are never counted."""
    months = pd.DatetimeIndex(months)
    counts = [len(pd.bdate_range(month, month + MonthEnd(0))) for month in months]
    return pd.Series(counts, index=months, dtype="int64")


def _weekday_holidays_per_month(holidays: pd.DataFrame, months: pd.DatetimeIndex, location: str) -> pd.Series:
    """Weekday holidays that apply to one location, per month (India-wide holidays included)."""
    relevant = applicable_holidays(holidays, [location])
    if relevant.empty:
        return pd.Series(0, index=months, dtype="int64")
    weekday = relevant.loc[relevant["date"].dt.weekday < 5]
    if weekday.empty:
        return pd.Series(0, index=months, dtype="int64")
    month_start = weekday["date"].dt.to_period("M").dt.to_timestamp()
    return month_start.value_counts().reindex(months, fill_value=0).astype("int64")


def working_days_by_location(
    months: pd.DatetimeIndex, locations: list[str], holidays: pd.DataFrame, exclude_holidays: bool
) -> pd.DataFrame:
    """Working days per month (rows) for each location (columns)."""
    months = pd.DatetimeIndex(months)
    weekdays = weekday_counts(months)
    columns: dict[str, pd.Series] = {}
    for location in locations:
        working = weekdays.copy()
        if exclude_holidays:
            working = working - _weekday_holidays_per_month(holidays, months, location)
        columns[location] = working.clip(lower=0)
    return pd.DataFrame(columns, index=months)


def _most_common(values: pd.Series) -> str:
    modes = values.mode()
    return str(modes.iat[0]) if len(modes) else UNKNOWN_LOCATION


def person_months_from_rows(rows: pd.DataFrame, hours_per_day: float) -> pd.DataFrame:
    """One row per member and month from clean timesheet rows.

    `rows` needs month, member_id, location, hours and is_vacation. Only vacation hours become vacation days.
    """
    work = rows[["month", "member_id", "location", "hours", "is_vacation"]].copy()
    work["location"] = work["location"].fillna(UNKNOWN_LOCATION)
    work["vacation_hours"] = work["hours"].where(work["is_vacation"].astype(bool), 0.0)
    grouped = work.groupby(["month", "member_id"], as_index=False).agg(
        location=("location", _most_common),
        vacation_hours=("vacation_hours", "sum"),
    )
    grouped["vacation_days"] = days_from_hours(grouped["vacation_hours"], hours_per_day)
    return grouped[PERSON_MONTH_COLUMNS]


def person_months_from_forecast(employees: pd.DataFrame, hours_per_day: float) -> pd.DataFrame:
    """Forecast employee-months (predicted vacation hours) converted to vacation days.

    `employees` needs month, member_id, location and forecast (hours).
    """
    out = employees[["month", "member_id", "location", "forecast"]].copy()
    out["vacation_days"] = days_from_hours(out["forecast"], hours_per_day)
    return out[PERSON_MONTH_COLUMNS]


def monthly_business_table(
    person_months: pd.DataFrame,
    months: pd.DatetimeIndex,
    holidays: pd.DataFrame,
    exclude_holidays: bool,
) -> pd.DataFrame:
    """Business metrics per month. Months with no employees give zero employees and NaN percentages."""
    months = pd.DatetimeIndex(months)
    locations = sorted(person_months["location"].unique().tolist())
    if person_months.empty:
        available = pd.Series(0.0, index=months)
        employees = pd.Series(0, index=months, dtype="int64")
        days = pd.Series(0.0, index=months)
    else:
        working = working_days_by_location(months, locations, holidays, exclude_holidays)
        counts = (
            person_months.groupby(["month", "location"]).size().unstack(fill_value=0)
            .reindex(index=months, columns=locations, fill_value=0)
        )
        available = (counts * working).sum(axis=1).astype(float)
        employees = person_months.groupby("month")["member_id"].nunique().reindex(months, fill_value=0)
        days = person_months.groupby("month")["vacation_days"].sum().reindex(months, fill_value=0.0)

    table = pd.DataFrame(
        {
            "month": months,
            "employees": employees.to_numpy(),
            "weekdays": weekday_counts(months).to_numpy(),
            "available_person_days": available.to_numpy(),
            "vacation_days": days.to_numpy(),
        }
    )
    available_safe = table["available_person_days"].where(table["available_person_days"] > 0)
    employees_safe = table["employees"].where(table["employees"] > 0)
    table["vacation_pct"] = table["vacation_days"] / available_safe * 100.0
    table["availability_pct"] = 100.0 - table["vacation_pct"]
    table["avg_vacation_days_per_employee"] = table["vacation_days"] / employees_safe
    return table[MONTHLY_COLUMNS]


def summarise_period(monthly: pd.DataFrame, unique_employees: int) -> dict[str, float | int]:
    """Totals over a period. Percentages use summed days over summed person-days, not an average of monthly %."""
    available = float(monthly["available_person_days"].sum())
    days = float(monthly["vacation_days"].sum())
    vacation_pct = safe_pct(days, available)
    return {
        "months": int(len(monthly)),
        "employees": int(unique_employees),
        "weekdays": int(monthly["weekdays"].sum()),
        "available_person_days": available,
        "vacation_days": days,
        "vacation_pct": vacation_pct,
        "availability_pct": 100.0 - vacation_pct,
        "avg_vacation_days_per_employee": days / unique_employees if unique_employees > 0 else float("nan"),
    }


def partial_months(dates: pd.Series) -> list[pd.Timestamp]:
    """Calendar months at either end of the data that are not fully covered. These are excluded from history."""
    clean = pd.to_datetime(dates).dropna()
    if clean.empty:
        return []
    first, last = clean.min(), clean.max()
    partial: list[pd.Timestamp] = []
    first_month = first.to_period("M").to_timestamp()
    if first.normalize() != first_month:
        partial.append(first_month)
    last_month = last.to_period("M").to_timestamp()
    if last.normalize() < last_month + MonthEnd(0) and last_month not in partial:
        partial.append(last_month)
    return partial


def check_monthly_table(monthly: pd.DataFrame, tolerance: float = 1e-6) -> list[str]:
    """Consistency checks. An empty list means the table passed every check."""
    issues: list[str] = []
    if (monthly["vacation_days"] < -tolerance).any():
        issues.append("Negative vacation days found.")
    no_capacity = (monthly["employees"] > 0) & (monthly["available_person_days"] <= 0)
    for month in monthly.loc[no_capacity, "month"]:
        issues.append(f"{month:%b %Y}: employees present but zero working days.")
    over = monthly["vacation_days"] > monthly["available_person_days"] + tolerance
    for month in monthly.loc[over, "month"]:
        issues.append(f"{month:%b %Y}: vacation days exceed available person-days.")
    total = monthly["vacation_pct"] + monthly["availability_pct"]
    off = total.notna() & ((total - 100.0).abs() > 1e-6)
    for month in monthly.loc[off, "month"]:
        issues.append(f"{month:%b %Y}: vacation % and availability % do not sum to 100.")
    return issues
