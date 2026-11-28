"""Per-employee leave profile by calendar month and a forecast that repeats the average month pattern."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from vacation_forecasting.business_metrics import safe_pct
from vacation_forecasting.config import FULL_DAY_HOURS, LEAVE_TRC_CODES

FORECAST_MONTHS = 12
STANDARD_WORKING_DAYS = 20
VACATION_CODE = "105"
# Vacation is already in the monthly profile, so it is excluded here to avoid double counting.
OCCASION_LEAVE_CODES = frozenset({"109", "115", "610"})


@dataclass(frozen=True)
class LeaveForecast:
    """Average leave days per calendar month, the forecast months and the number of years behind each average."""

    profile: pd.DataFrame
    forecast: pd.DataFrame


def build_leave_forecast(
    rows: pd.DataFrame, member_id: str, latest_data_month: pd.Timestamp, today: date, horizon: int = FORECAST_MONTHS
) -> LeaveForecast:
    """Average each calendar month over the member's complete months, then repeat that pattern forward.

    Months count only when the member has a timesheet row in them. The latest month in the data is excluded
    because it may be partial.
    """
    employee = rows.loc[rows["member_id"].astype(str).eq(str(member_id))]
    active_months = pd.DatetimeIndex(employee["month"].dropna().unique()).sort_values()
    leave_hours = employee.loc[employee["trc_code"].isin(LEAVE_TRC_CODES)].groupby("month")["hours"].sum()
    days = (leave_hours / FULL_DAY_HOURS).reindex(active_months, fill_value=0.0)
    days = days.loc[days.index < latest_data_month]

    profile = (
        days.groupby(days.index.month)
        .agg(avg_leave_days="mean", years_observed="size")
        .reindex(range(1, 13), fill_value=0)
    )
    profile["Month"] = [calendar.month_name[m] for m in profile.index]
    profile.index.name = "calendar_month"

    first_forecast = max(
        pd.Timestamp(today).to_period("M").to_timestamp(),
        latest_data_month + pd.DateOffset(months=1),
    )
    months = pd.date_range(first_forecast, periods=horizon, freq="MS")
    forecast = pd.DataFrame(
        {
            "Month": months,
            "Forecast leave days": profile["avg_leave_days"].reindex(months.month).to_numpy(),
            "Years behind average": profile["years_observed"].reindex(months.month).to_numpy(),
        }
    )
    return LeaveForecast(profile=profile.reset_index(), forecast=forecast)


def occasion_key(name: str) -> str:
    """Match the same occasion across years despite spelling variants, e.g. Sankranti / Sankranthi."""
    first = re.split(r"[/|]", str(name))[0]
    key = re.sub(r"[^a-z]", "", first.lower())
    return key.replace("sankrathi", "sankranti").replace("sankranthi", "sankranti")


def occasion_rates(employee: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Expected holiday-date leave days per occasion next year.

    Every holiday group is divided by the same years (calendar years the member has data for), so a name variant
    seen in one year contributes its share of a year rather than a 100% rate.
    """
    columns = ["holiday_name", "rate", "projected_month"]
    if holidays.empty:
        return pd.DataFrame(columns=columns)
    active_years = set(employee["month"].dropna().dt.year)
    leave_days = set(employee.loc[employee["trc_code"].isin(OCCASION_LEAVE_CODES), "transaction_date"].dt.normalize())
    calendar_days = holidays.assign(day=holidays["date"].dt.normalize())
    calendar_days = calendar_days.assign(occasion=calendar_days["holiday_name"].map(occasion_key))
    years = set(calendar_days["day"].dt.year) & active_years
    if not years:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, group in calendar_days.groupby("occasion"):
        on_leave = {day.year for day in group["day"] if day in leave_days and day.year in years}
        if not on_leave:
            rows.append(None)
            continue
        latest = group.sort_values("day").iloc[-1]
        rows.append(
            {
                "holiday_name": latest["holiday_name"],
                "rate": len(on_leave) / len(years),
                "projected_month": group["day"].max().month,
            }
        )
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows, columns=columns)


def build_employee_monthly_forecast(
    rows: pd.DataFrame,
    member_id: str,
    latest_data_month: pd.Timestamp,
    today: date,
    holidays: pd.DataFrame,
    horizon: int = FORECAST_MONTHS,
) -> pd.DataFrame:
    """Vacation days per calendar month for one member, in the same columns as the business monthly view."""
    employee = rows.loc[rows["member_id"].astype(str).eq(str(member_id))]
    active_months = pd.DatetimeIndex(employee["month"].dropna().unique()).sort_values()
    vacation_hours = employee.loc[employee["trc_code"].eq(VACATION_CODE)].groupby("month")["hours"].sum()
    days = (vacation_hours / FULL_DAY_HOURS).reindex(active_months, fill_value=0.0)
    days = days.loc[days.index < latest_data_month]
    profile = days.groupby(days.index.month).mean().reindex(range(1, 13), fill_value=0.0)

    first_forecast = max(
        pd.Timestamp(today).to_period("M").to_timestamp(),
        latest_data_month + pd.DateOffset(months=1),
    )
    months = pd.date_range(first_forecast, periods=horizon, freq="MS")
    weekdays = np.full(len(months), STANDARD_WORKING_DAYS)
    occasion_days = (
        occasion_rates(employee, holidays).groupby("projected_month")["rate"].sum().reindex(range(1, 13), fill_value=0.0)
    )
    vacation_days = profile.reindex(months.month).to_numpy() + occasion_days.reindex(months.month).to_numpy()
    vacation_pct = [safe_pct(v, w) for v, w in zip(vacation_days, weekdays)]
    return pd.DataFrame(
        {
            "month": months,
            "employees": 1,
            "weekdays": weekdays,
            "vacation_days": vacation_days,
            "vacation_pct": vacation_pct,
            "availability_pct": [100.0 - p if not pd.isna(p) else float("nan") for p in vacation_pct],
        }
    )


def ranked_months(profile: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Calendar months ordered by average leave days, highest first."""
    ranked = profile.loc[profile["avg_leave_days"] > 0, ["Month", "avg_leave_days", "years_observed"]]
    return ranked.sort_values("avg_leave_days", ascending=False).head(top).reset_index(drop=True)
