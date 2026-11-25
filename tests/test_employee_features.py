"""Tests for employee-month features: gap zero-fill, lag correctness, leakage and missing-history handling."""

import numpy as np
import pandas as pd
import pytest

from vacation_forecasting.employee_features import (
    FEATURE_COLUMNS,
    add_employee_features,
    analysis_window,
    build_employee_month_frame,
    extend_for_future,
)


def _rows(entries):
    """entries: (member, 'YYYY-MM-DD', hours, is_vacation)."""
    data = []
    for member, date, hours, vac in entries:
        ts = pd.Timestamp(date)
        data.append(
            {
                "member_id": member,
                "pag": "Solvay",
                "location": "India",
                "transaction_date": ts,
                "month": ts.to_period("M").to_timestamp(),
                "hours": float(hours),
                "is_vacation": bool(vac),
            }
        )
    return pd.DataFrame(data)


HOLIDAYS = pd.DataFrame(columns=["date", "holiday_name", "holiday_type", "location", "source_file", "year"])


def _build(entries, start="2023-01-01", end="2023-06-01"):
    rows = _rows(entries)
    return add_employee_features(
        build_employee_month_frame(rows, HOLIDAYS, pd.Timestamp(start), pd.Timestamp(end))
    )


def _value(frame, member, month, column):
    row = frame.loc[(frame["member_id"] == member) & (frame["month"] == pd.Timestamp(month))]
    assert len(row) == 1
    return row[column].iat[0]


def test_gap_months_inside_span_are_zero_filled():
    frame = _build([("A", "2023-01-10", 8, True), ("A", "2023-04-10", 4, True)])
    months = frame.loc[frame["member_id"] == "A", "month"]
    assert list(months.dt.month) == [1, 2, 3, 4]
    assert _value(frame, "A", "2023-02-01", "vacation_hours") == 0.0


def test_months_outside_employee_span_are_absent():
    frame = _build([("A", "2023-02-10", 8, True), ("B", "2023-01-10", 8, True)])
    a_months = frame.loc[frame["member_id"] == "A", "month"]
    assert a_months.min() == pd.Timestamp("2023-02-01")
    assert a_months.max() == pd.Timestamp("2023-02-01")


def test_lag_1_is_previous_month_value():
    frame = _build([("A", "2023-01-10", 8, True), ("A", "2023-02-10", 16, True), ("A", "2023-03-10", 2, True)])
    assert _value(frame, "A", "2023-03-01", "vacation_lag_1") == 16.0
    assert _value(frame, "A", "2023-03-01", "vacation_lag_2") == 8.0


def test_features_never_use_the_current_month():
    frame = _build([("A", "2023-01-10", 8, True), ("A", "2023-02-10", 100, True)])
    # February's own 100 must not appear in February's own history features.
    assert _value(frame, "A", "2023-02-01", "vacation_lag_1") == 8.0
    assert _value(frame, "A", "2023-02-01", "hist_max_vacation") == 8.0
    assert _value(frame, "A", "2023-02-01", "hist_avg_vacation") == 8.0


def test_expanding_history_excludes_current_month():
    frame = _build([("A", "2023-01-10", 4, True), ("A", "2023-02-10", 8, True), ("A", "2023-03-10", 12, True)])
    assert _value(frame, "A", "2023-03-01", "hist_avg_vacation") == pytest.approx(6.0)
    assert _value(frame, "A", "2023-03-01", "hist_vacation_frequency") == pytest.approx(1.0)


def test_missing_history_stays_nan_not_zero():
    frame = _build([("A", "2023-01-10", 8, True), ("A", "2023-02-10", 8, True)])
    # First month of an employee has no history at all.
    assert np.isnan(_value(frame, "A", "2023-01-01", "vacation_lag_1"))
    assert np.isnan(_value(frame, "A", "2023-01-01", "hist_avg_vacation"))
    # Lag 12 points before the analysis window, so it is unknown rather than zero.
    assert np.isnan(_value(frame, "A", "2023-02-01", "vacation_same_month_last_year"))


def test_same_month_last_year_uses_value_twelve_months_back():
    entries = [("A", "2022-06-10", 24, True), ("A", "2022-10-10", 1, True), ("A", "2023-06-10", 2, True)]
    frame = _build(entries, start="2022-06-01", end="2023-06-01")
    assert _value(frame, "A", "2023-06-01", "vacation_same_month_last_year") == 24.0


def test_analysis_window_drops_partial_edge_months():
    dates = pd.Series(pd.to_datetime(["2022-10-03", "2025-08-20"]))
    start, end = analysis_window(dates)
    assert start == pd.Timestamp("2022-11-01")
    assert end == pd.Timestamp("2025-07-01")


def test_feature_columns_exclude_identifiers_and_calendar_year():
    for forbidden in ("member_id", "pag", "calendar_year"):
        assert forbidden not in FEATURE_COLUMNS
    assert all(column in _build([("A", "2023-01-10", 8, True)]).columns for column in FEATURE_COLUMNS)


def test_future_rows_have_no_actual_event_flag():
    frame = _build([("A", "2023-01-10", 8, True), ("A", "2023-02-10", 8, True)], start="2023-01-01", end="2023-02-01")
    future = extend_for_future(frame, HOLIDAYS, horizon=2, active_lookback_months=3)
    assert future.loc[future["is_future"], "actual_event"].isna().all()


def test_predicted_months_do_not_change_event_features():
    entries = [("A", "2023-01-10", 8, True), ("A", "2023-02-10", 8, True), ("A", "2023-03-10", 8, True)]
    future = extend_for_future(_build(entries, start="2023-01-01", end="2023-03-01"), HOLIDAYS, 3, 3)
    april = pd.Timestamp("2023-04-01")

    def later_event_features(predicted_hours: float) -> pd.DataFrame:
        work = future.copy()
        work.loc[work["is_future"] & (work["month"] == april), "vacation_hours"] = predicted_hours
        feats = add_employee_features(work)
        return feats.loc[feats["is_future"] & (feats["month"] > april), ["vacation_months_prev12", "hist_vacation_frequency"]]

    pd.testing.assert_frame_equal(later_event_features(0.0), later_event_features(1000.0))


def test_extend_for_future_adds_only_recently_active_employees():
    frame = _build(
        [
            ("A", "2023-01-10", 8, True),
            ("A", "2023-06-10", 8, True),
            ("B", "2023-01-10", 8, True),
        ],
        start="2023-01-01",
        end="2023-06-01",
    )
    # B's last activity is January, outside the 3-month lookback from June.
    future = extend_for_future(frame, HOLIDAYS, horizon=2, active_lookback_months=3)
    future_rows = future.loc[future["is_future"]]
    assert set(future_rows["member_id"]) == {"A"}
    assert sorted(future_rows["month"].dt.month.unique()) == [7, 8]
    assert future_rows[FEATURE_COLUMNS[0]].notna().all()


def test_forecast_population_is_latest_month_headcount_not_three_month_union():
    # B was present only in June, so a 3-month union would wrongly carry them into the forecast.
    frame = _build(
        [
            ("A", "2023-06-10", 8, True),
            ("A", "2023-07-10", 8, True),
            ("A", "2023-08-10", 8, True),
            ("B", "2023-06-10", 8, True),
            ("C", "2023-07-10", 8, True),
        ],
        start="2023-06-01",
        end="2023-08-01",
    )
    point_in_time = extend_for_future(frame, HOLIDAYS, horizon=2, active_lookback_months=1)
    assert set(point_in_time.loc[point_in_time["is_future"], "member_id"]) == {"A"}
    union = extend_for_future(frame, HOLIDAYS, horizon=2, active_lookback_months=3)
    assert set(union.loc[union["is_future"], "member_id"]) == {"A", "B", "C"}
