"""Unit tests for demo insight statements and feature labels. Uses small hand-written tables, not the real data."""

from __future__ import annotations

import pandas as pd
import pytest

from vacation_forecasting.employee_features import FEATURE_COLUMNS, FEATURE_LABELS
from vacation_forecasting.employee_forecasting import EMPLOYEE_XGBOOST, SEASONAL_NAIVE, feature_weights
from vacation_forecasting.insights import (
    forecast_statements,
    history_statements,
    model_comparison_statements,
    year_over_year_statement,
)


def test_every_feature_has_a_plain_language_label():
    assert set(FEATURE_LABELS) == set(FEATURE_COLUMNS)


def test_history_statements_name_peak_low_and_calendar_levels():
    months = pd.date_range("2023-01-01", periods=12, freq="MS")
    values = [10.0] * 12
    values[1] = 5.0  # February is the lowest month
    values[11] = 100.0  # December is the highest month
    history = pd.DataFrame({"month": months, "vacation_hours": values})
    calendar = pd.DataFrame({"calendar_month": range(1, 13), "avg_vacation_hours": values})

    lines = history_statements(history, calendar)

    assert "Highest vacation month in the window: Dec 2023 (100.0 h)." in lines
    assert "Lowest vacation month in the window: Feb 2023 (5.0 h)." in lines
    assert "Calendar months above the average monthly level: Dec." in lines


def test_year_over_year_statement_compares_last_12_with_previous_12():
    months = pd.date_range("2023-09-01", periods=24, freq="MS")
    values = [10.0] * 12 + [15.0] * 12
    history = pd.DataFrame({"month": months, "vacation_hours": values})

    text = year_over_year_statement(history)

    assert text is not None and "+50.0%" in text


def test_year_over_year_statement_needs_24_months():
    months = pd.date_range("2023-09-01", periods=23, freq="MS")
    history = pd.DataFrame({"month": months, "vacation_hours": [10.0] * 23})
    assert year_over_year_statement(history) is None


def test_forecast_statements_report_total_and_change_vs_last_year():
    forecast = pd.DataFrame(
        {
            "month": pd.date_range("2025-09-01", periods=3, freq="MS"),
            "forecast": [100.0, 200.0, 300.0],
            "same_month_last_year_actual": [150.0, 150.0, 150.0],
        }
    )

    lines = forecast_statements(forecast)

    assert lines[0] == "Forecast total Sep 2025 to Nov 2025: 600.0 h."
    assert any("The forecast is 33.3% higher than that period." in line for line in lines)


def test_model_comparison_states_lowest_mae_without_claiming_xgboost_superiority():
    pag = pd.DataFrame(
        {"model": [SEASONAL_NAIVE, EMPLOYEE_XGBOOST], "n_rows": [6, 6], "MAE": [81.0, 74.0]}
    )
    employee = pd.DataFrame(
        {"model": [SEASONAL_NAIVE, EMPLOYEE_XGBOOST], "n_rows": [791, 791], "MAE": [3.36, 3.46]}
    )

    lines = model_comparison_statements(pag, employee)

    assert "Lowest PAG-level holdout MAE: Employee-month XGBoost (one-step) (74.0 h per month)." in lines
    assert any("seasonal naive has lower MAE than employee-month XGBoost (3.36 vs 3.46 h)" in line for line in lines)


def test_feature_weights_sum_to_100_and_rank_largest_first():
    contributions = pd.DataFrame(0.0, index=[0, 1], columns=list(FEATURE_COLUMNS))
    contributions["holiday_count"] = [3.0, -3.0]

    weights = feature_weights(contributions)

    assert weights.iloc[0]["feature"] == "holiday_count"
    assert weights["share_pct"].sum() == pytest.approx(100.0)
