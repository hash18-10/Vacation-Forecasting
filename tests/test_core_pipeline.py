"""Tests for classification, leakage safety, forecasting, evaluation and insights on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vacation_forecasting.config import ModelSettings
from vacation_forecasting.data_loader import resolve_columns
from vacation_forecasting.evaluation import compute_metrics, compare_models, best_model
from vacation_forecasting.features import (
    FEATURE_COLUMNS,
    TARGET,
    build_features,
    build_member_month_panel,
    extend_panel_for_future,
)
from vacation_forecasting.forecasting import (
    BASELINE_MODELS,
    baseline_one_step_backtest,
    baseline_recursive_forecast,
    xgboost_one_step_backtest,
    xgboost_recursive_forecast,
)
from vacation_forecasting.holiday_analysis import holiday_counts_by_month
from vacation_forecasting.insights import build_insights
from vacation_forecasting.preprocessing import preprocess_timesheets
from vacation_forecasting.vacation_analysis import (
    add_vacation_flag,
    calendar_month_profile,
    fy_vacation_summary,
    monthly_vacation_summary,
)

FAST_MODEL = ModelSettings(n_estimators=40, max_depth=2)


def canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename raw headers to canonical names, as the loader does."""
    mapping = resolve_columns(raw.columns, "synthetic")
    return raw.rename(columns={raw_name: canonical for canonical, raw_name in mapping.items()})


def test_resolve_columns_accepts_real_timesheet_headers() -> None:
    """Headers from the FY23-FY25 memberwise sheets, including HOURS(DAILY), must map to canonical names."""
    headers = [
        "TRANSACTION DATE", "HOURS(DAILY)", "MEMBER ID", "PAG", "TRC CODE", "TRC DESC", "MEMBER NAME",
    ]
    mapping = resolve_columns(headers, "real")
    assert mapping["hours"] == "HOURS(DAILY)"
    assert mapping["transaction_date"] == "TRANSACTION DATE"
    assert mapping["member_id"] == "MEMBER ID"


@pytest.fixture
def clean_df(synthetic_raw, settings_for) -> pd.DataFrame:
    result = preprocess_timesheets(canonicalize(synthetic_raw), settings_for)
    return add_vacation_flag(result.clean)


def test_only_trc_105_is_vacation(clean_df: pd.DataFrame) -> None:
    assert clean_df.loc[clean_df["is_vacation"], "trc_code"].eq("105").all()
    assert not clean_df.loc[~clean_df["is_vacation"], "trc_code"].eq("105").any()


def test_description_does_not_affect_classification(settings_for, synthetic_raw) -> None:
    raw = canonicalize(synthetic_raw)
    raw["trc_description"] = "Vacation"  # mislabelled text must not change the flag
    result = preprocess_timesheets(raw, settings_for)
    flagged = add_vacation_flag(result.clean)
    assert flagged["is_vacation"].eq(flagged["trc_code"].eq("105")).all()


def test_fy_summary_and_monthly_summary_are_consistent(clean_df: pd.DataFrame) -> None:
    monthly = monthly_vacation_summary(clean_df)
    fy = fy_vacation_summary(clean_df)
    assert monthly["vacation_hours"].sum() == pytest.approx(fy["vacation_hours"].sum())
    assert len(calendar_month_profile(clean_df)) == 12


def test_features_do_not_use_current_or_future_target(clean_df: pd.DataFrame) -> None:
    panel = build_member_month_panel(clean_df)
    base = build_features(panel)
    cutoff = pd.Timestamp("2023-01-01")

    changed = panel.copy()
    changed.loc[changed["month"] >= cutoff, TARGET] = changed.loc[changed["month"] >= cutoff, TARGET] + 999
    shifted = build_features(changed)

    up_to_cutoff = base["month"] <= cutoff
    for column in FEATURE_COLUMNS:
        left = base.loc[up_to_cutoff, column].astype(float).to_numpy()
        right = shifted.loc[up_to_cutoff, column].astype(float).to_numpy()
        assert np.allclose(left, right, equal_nan=True), f"{column} leaks the current or future target"


def test_baselines_one_step_backtest_uses_only_past(clean_df: pd.DataFrame) -> None:
    monthly = monthly_vacation_summary(clean_df).set_index("month")["vacation_hours"]
    test_months = monthly.index[-6:]
    result = baseline_one_step_backtest(monthly, test_months)
    assert set(result["model"]) == set(BASELINE_MODELS)
    seasonal = result.loc[result["model"] == "seasonal_naive"].set_index("month")["forecast"]
    expected = monthly.loc[test_months[0] - pd.DateOffset(months=12)]
    assert seasonal.iloc[0] == pytest.approx(expected)


def test_baseline_recursive_forecast_shape(clean_df: pd.DataFrame) -> None:
    monthly = monthly_vacation_summary(clean_df).set_index("month")["vacation_hours"]
    future = pd.date_range(monthly.index.max() + pd.DateOffset(months=1), periods=6, freq="MS")
    result = baseline_recursive_forecast(monthly, future)
    assert len(result) == 6 * len(BASELINE_MODELS)
    assert (result["forecast"] >= 0).all()


def test_xgboost_backtest_and_recursive_forecast(clean_df: pd.DataFrame) -> None:
    panel = build_member_month_panel(clean_df)
    months = pd.date_range(panel["month"].min(), panel["month"].max(), freq="MS")
    holidays = holiday_counts_by_month(
        pd.DataFrame(
            {"date": [], "holiday_name": [], "holiday_type": [], "location": [], "source_file": [], "year": []}
        ).astype({"date": "datetime64[ns]", "year": "int64"}),
        months,
    )
    test_months = months[-6:]
    backtest = xgboost_one_step_backtest(panel, test_months, FAST_MODEL, holidays, fy_start_month=10)
    assert set(backtest["month"]) == set(test_months)
    assert (backtest["forecast"] >= 0).all()

    future_panel = extend_panel_for_future(panel, horizon=3, active_lookback_months=3)
    forecast = xgboost_recursive_forecast(future_panel, FAST_MODEL, holidays, fy_start_month=10)
    assert forecast["forecast"].notna().all()
    assert forecast["month"].nunique() == 3


def test_metrics_known_values() -> None:
    actual = pd.Series([10.0, 20.0])
    forecast = pd.Series([12.0, 17.0])
    metrics = compute_metrics(actual, forecast)
    assert metrics["MAE"] == pytest.approx(2.5)
    assert metrics["RMSE"] == pytest.approx(np.sqrt((4 + 9) / 2))
    assert metrics["WAPE"] == pytest.approx(5 / 30 * 100)
    assert metrics["Bias"] == pytest.approx(-0.5)


def test_best_model_and_insights_are_computed(clean_df: pd.DataFrame) -> None:
    monthly = monthly_vacation_summary(clean_df)
    series = monthly.set_index("month")["vacation_hours"]
    test_months = series.index[-6:]
    backtest = baseline_one_step_backtest(series, test_months)
    backtest["actual"] = backtest["month"].map(series)
    comparison = compare_models(backtest)
    winner = best_model(comparison)
    assert winner in BASELINE_MODELS

    lines = build_insights(
        monthly,
        calendar_month_profile(clean_df),
        fy_vacation_summary(clean_df),
        comparison,
        winner,
        pd.DataFrame(columns=["holiday_name", "date", "window_avg_hours", "rest_avg_hours", "difference_hours"]),
        forecast_total=123.0,
    )
    assert lines and all("cause" not in line.lower() or "does not show cause" in line.lower() for line in lines)
