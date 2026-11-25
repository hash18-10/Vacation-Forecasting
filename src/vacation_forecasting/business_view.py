"""Business view for one PAG: historical and forecast monthly tables built from an employee-level run.

No forecasting happens here. Forecast values come from the run's employee-month rows; this module only converts
hours to vacation days and calculates the business percentages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vacation_forecasting.business_metrics import (
    check_monthly_table,
    monthly_business_table,
    partial_months,
    person_months_from_forecast,
    person_months_from_rows,
    summarise_period,
)
from vacation_forecasting.config import Settings
from vacation_forecasting.employee_forecasting import EmployeeForecastRun
from vacation_forecasting.insights import forecast_insight_lines, historical_trend_lines


@dataclass
class BusinessView:
    pag: str
    hours_per_vacation_day: float
    exclude_public_holidays: bool
    history_monthly: pd.DataFrame
    history_summary: dict
    forecast_monthly: pd.DataFrame
    forecast_summary: dict
    previous_period_summary: dict | None
    prior_period_summary: dict | None
    last_year_summary: dict | None
    excluded_partial_months: list[pd.Timestamp] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    history_lines: list[str] = field(default_factory=list)
    forecast_lines: list[str] = field(default_factory=list)


def _period_summary(table: pd.DataFrame, person_months: pd.DataFrame) -> dict:
    """Summary over the months in `table`. Unique employees are counted across those months only."""
    in_period = person_months.loc[person_months["month"].isin(table["month"])]
    return summarise_period(table, in_period["member_id"].nunique())


def build_business_view(pag_df: pd.DataFrame, holidays: pd.DataFrame, run: EmployeeForecastRun, settings: Settings) -> BusinessView:
    """Assemble the business tables. `pag_df` must be the clean, vacation-flagged rows for the PAG."""
    hours_per_day = settings.hours_per_vacation_day
    exclude = settings.exclude_public_holidays
    pag = settings.pag

    start, end = run.window
    history_months = pd.date_range(start, end, freq="MS")
    window_rows = pag_df.loc[(pag_df["month"] >= start) & (pag_df["month"] <= end)]
    history_pm = person_months_from_rows(window_rows, hours_per_day)
    history_monthly = monthly_business_table(history_pm, history_months, holidays, exclude)
    history_summary = summarise_period(history_monthly, history_pm["member_id"].nunique())

    forecast_months = pd.DatetimeIndex(sorted(run.future_monthly["month"].unique()))
    forecast_pm = person_months_from_forecast(run.future_employees, hours_per_day)
    forecast_monthly = monthly_business_table(forecast_pm, forecast_months, holidays, exclude)
    forecast_summary = summarise_period(forecast_monthly, forecast_pm["member_id"].nunique())

    last_year_summary = None
    if len(forecast_months) > 0:
        last_year_months = forecast_months - pd.DateOffset(months=12)
        last_year_table = history_monthly.loc[history_monthly["month"].isin(last_year_months)]
        if len(last_year_table) == len(forecast_months):
            last_year_summary = _period_summary(last_year_table, history_pm)

    previous_period_summary = None
    if len(history_monthly) >= 12:
        previous_period_summary = _period_summary(history_monthly.tail(12), history_pm)
    prior_period_summary = None
    if len(history_monthly) >= 24:
        prior_period_summary = _period_summary(history_monthly.iloc[-24:-12], history_pm)

    issues = check_monthly_table(history_monthly) + check_monthly_table(forecast_monthly)
    latest_employees = int(history_monthly["employees"].iloc[-1]) if len(history_monthly) else None
    history_lines = historical_trend_lines(history_monthly, history_summary, previous_period_summary, prior_period_summary)
    forecast_lines = forecast_insight_lines(
        forecast_monthly, forecast_summary, previous_period_summary, last_year_summary, latest_employees
    )

    return BusinessView(
        pag=pag,
        hours_per_vacation_day=hours_per_day,
        exclude_public_holidays=exclude,
        history_monthly=history_monthly,
        history_summary=history_summary,
        forecast_monthly=forecast_monthly,
        forecast_summary=forecast_summary,
        previous_period_summary=previous_period_summary,
        prior_period_summary=prior_period_summary,
        last_year_summary=last_year_summary,
        excluded_partial_months=partial_months(pag_df["transaction_date"]),
        issues=issues,
        history_lines=history_lines,
        forecast_lines=forecast_lines,
    )
