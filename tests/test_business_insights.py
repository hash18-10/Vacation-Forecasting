from __future__ import annotations

import pandas as pd

from vacation_forecasting.insights import forecast_insight_lines, historical_trend_lines


def _summary(vacation_pct: float, months: int = 12, avg: float = 1.0) -> dict:
    return {
        "months": months,
        "employees": 10,
        "weekdays": 260,
        "available_person_days": 2600.0,
        "vacation_days": vacation_pct / 100 * 2600.0,
        "vacation_pct": vacation_pct,
        "availability_pct": 100.0 - vacation_pct,
        "avg_vacation_days_per_employee": avg,
    }


def _history_with_december_peak() -> pd.DataFrame:
    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")
    values = [10.0 if m.month == 12 else 2.0 for m in months]
    return pd.DataFrame({"month": months, "vacation_pct": values})


def test_historical_trend_names_peak_calendar_month_and_overall_level():
    history = _history_with_december_peak()
    lines = historical_trend_lines(history, _summary(3.3, months=24), None, None)
    text = " ".join(lines)
    assert "December has the highest average vacation %" in text
    assert "(10.0%)" in text
    assert "Historical vacation % over 24 complete months: 3.3%" in text


def test_historical_trend_reports_change_between_previous_and_prior_periods():
    history = _history_with_december_peak()
    lines = historical_trend_lines(
        history,
        _summary(3.0, months=24),
        previous_period=_summary(4.0),
        prior_period=_summary(2.5),
    )
    assert any("increased compared with the 12 months before that (+1.5 percentage points)" in line for line in lines)


def test_forecast_insights_state_vacation_and_availability_from_summary():
    months = pd.date_range("2026-01-01", periods=3, freq="MS")
    forecast = pd.DataFrame(
        {"month": months, "vacation_pct": [2.0, 4.0, 3.0], "forecast": [18.0, 36.0, 27.0]}
    )
    summary = _summary(3.0, months=3, avg=1.5)
    lines = forecast_insight_lines(forecast, summary, None, None, latest_employees=10)
    text = " ".join(lines)
    assert "The forecasted vacation level is 3.0%." in text
    assert "Expected availability is 97.0%." in text
    assert "average of 1.5 vacation days each" in text
    assert "February 2026 is expected to have the highest vacation activity (4.0%)." in text
    assert "based on historical vacation patterns for this PAG" in text


def test_forecast_insights_empty_when_no_forecast_months():
    empty = pd.DataFrame({"month": pd.to_datetime([]), "vacation_pct": [], "forecast": []})
    assert forecast_insight_lines(empty, _summary(float("nan"), months=0), None, None, None) == []
