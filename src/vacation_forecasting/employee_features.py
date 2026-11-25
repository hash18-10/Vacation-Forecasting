"""Employee-month dataset and leakage-safe features for employee-level vacation forecasting.

Row: one employee in one calendar month inside the analysis window. Months between an employee's first and
last active month are included, and months without vacation are zero-filled (same convention as the member panel).
Target: vacation hours (TRC 105) in that month.

Leakage rule: every history feature for month m reads only rows dated before m (shift >= 1).
Calendar and holiday features are known in advance and are allowed.

Missing history: a lag that points before the employee's first observed month, or before the analysis
window, is NaN. It is never filled with 0, so "no data" and "no vacation" stay distinct.

Event flags (`actual_event`) come only from observed data. Predicted months are NaN and never count as events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vacation_forecasting.config import FY_START_MONTH, LOCATION_CODE_MAP, UNKNOWN_LOCATION
from vacation_forecasting.holiday_analysis import holiday_counts_by_month
from vacation_forecasting.normalization import financial_year_end, financial_year_label, fy_month_index

TARGET = "vacation_hours"
KEY = ["member_id", "month"]
LOCATION_NAMES: tuple[str, ...] = tuple(sorted(set(LOCATION_CODE_MAP.values()))) + (UNKNOWN_LOCATION,)

# Columns passed to XGBoost. Employee id, PAG (constant in this scope) and calendar year are kept in the
# dataset for reporting but are not model inputs: an id would memorise people, and a year trend cannot
# extrapolate beyond the three observed years.
FEATURE_COLUMNS: tuple[str, ...] = (
    "location_code",
    "month_of_year",
    "fy_quarter",
    "fy_month_index",
    "days_in_month",
    "holiday_count",
    "vacation_lag_1",
    "vacation_lag_2",
    "vacation_same_month_last_year",
    "vacation_prev3_sum",
    "vacation_prev12_sum",
    "vacation_occurrences_prev12",
    "vacation_months_prev12",
    "prior_months_observed",
    "hist_avg_vacation",
    "hist_max_vacation",
    "hist_vacation_frequency",
)

# Plain-language names for the model inputs, used in the report and the demo.
FEATURE_LABELS: dict[str, str] = {
    "location_code": "Employee location",
    "month_of_year": "Calendar month",
    "fy_quarter": "Financial-year quarter",
    "fy_month_index": "Month within the financial year",
    "days_in_month": "Days in the month",
    "holiday_count": "Public holidays in the month (employee's location)",
    "vacation_lag_1": "Vacation hours last month",
    "vacation_lag_2": "Vacation hours two months ago",
    "vacation_same_month_last_year": "Vacation hours in the same month last year",
    "vacation_prev3_sum": "Vacation hours in the last 3 months",
    "vacation_prev12_sum": "Vacation hours in the last 12 months",
    "vacation_occurrences_prev12": "Vacation occurrences in the last 12 months",
    "vacation_months_prev12": "Months with vacation in the last 12 months",
    "prior_months_observed": "Months of history available",
    "hist_avg_vacation": "Historical average monthly vacation hours",
    "hist_max_vacation": "Historical maximum monthly vacation hours",
    "hist_vacation_frequency": "Historical share of months with vacation",
}


def analysis_window(dates: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last complete calendar months covered by the data. Partial edge months are excluded."""
    first = pd.Timestamp(dates.min())
    last = pd.Timestamp(dates.max())
    start = first.to_period("M").to_timestamp()
    if first.normalize() != start:
        start = start + pd.DateOffset(months=1)
    end = last.to_period("M").to_timestamp()
    if last.normalize() < end + pd.offsets.MonthEnd(0):
        end = end - pd.DateOffset(months=1)
    if start > end:
        raise ValueError("No complete calendar month in the data")
    return start, end


def _most_common(values: pd.Series) -> str:
    mode = values.dropna().mode()
    return str(mode.iat[0]) if len(mode) else UNKNOWN_LOCATION


def build_employee_month_frame(
    rows: pd.DataFrame,
    holidays: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fy_start_month: int = FY_START_MONTH,
) -> pd.DataFrame:
    """One row per employee-month between the employee's first and last active month inside [start, end].

    `rows` needs columns: member_id, pag, location, month, hours, is_vacation.
    """
    if rows.empty:
        raise ValueError("No rows to build the employee-month dataset from")
    if rows["pag"].nunique() != 1:
        raise ValueError("Employee-month dataset needs exactly one PAG")

    window = rows.loc[(rows["month"] >= start) & (rows["month"] <= end)]
    if window.empty:
        raise ValueError(f"No rows between {start.date()} and {end.date()}")
    months_all = pd.date_range(start, end, freq="MS")

    window = window.assign(
        _vac_hours=window["hours"].where(window["is_vacation"], 0.0),
        _vac_rows=window["is_vacation"].astype(int),
    )
    observed = (
        window.groupby(["member_id", "month"])
        .agg(
            vacation_hours=("_vac_hours", "sum"),
            vacation_occurrences=("_vac_rows", "sum"),
            location=("location", _most_common),
        )
        .reset_index()
    )

    span = observed.groupby("member_id")["month"].agg(first_month="min", last_month="max").reset_index()
    grid = pd.MultiIndex.from_product([span["member_id"], months_all], names=KEY).to_frame(index=False)
    grid = grid.merge(span, on="member_id")
    grid = grid.loc[(grid["month"] >= grid["first_month"]) & (grid["month"] <= grid["last_month"])]
    frame = grid[KEY].merge(observed, on=KEY, how="left")
    frame[["vacation_hours", "vacation_occurrences"]] = frame[["vacation_hours", "vacation_occurrences"]].fillna(0.0)
    frame["actual_event"] = (frame["vacation_hours"] > 0).astype(float)
    frame["location"] = frame.groupby("member_id")["location"].transform(lambda s: s.ffill().bfill())
    frame["pag"] = str(rows["pag"].iat[0])
    frame["is_future"] = False
    frame = _add_calendar(frame, holidays, fy_start_month)
    return frame.sort_values(KEY).reset_index(drop=True)


def _add_calendar(frame: pd.DataFrame, holidays: pd.DataFrame, fy_start_month: int) -> pd.DataFrame:
    """Calendar, financial-year and holiday columns. All of them are known before the month starts."""
    out = frame.copy()
    fy_end = financial_year_end(out["month"], fy_start_month)
    out["financial_year_label"] = fy_end.map(financial_year_label)
    out["calendar_year"] = out["month"].dt.year
    out["month_of_year"] = out["month"].dt.month.astype(float)
    out["fy_month_index"] = fy_month_index(out["month"], fy_start_month).astype(float)
    out["fy_quarter"] = (out["fy_month_index"] - 1) // 3 + 1
    out["calendar_quarter"] = out["month"].dt.quarter
    out["days_in_month"] = out["month"].dt.days_in_month.astype(float)
    out["holiday_count"] = _holiday_counts_for_rows(out, holidays)
    code_lookup = {name: index for index, name in enumerate(LOCATION_NAMES)}
    out["location_code"] = out["location"].map(code_lookup).fillna(-1).astype(float)
    return out


def _holiday_counts_for_rows(frame: pd.DataFrame, holidays: pd.DataFrame) -> pd.Series:
    """Applicable holidays for each row's own location and month (India-wide holidays always count)."""
    months = pd.DatetimeIndex(sorted(frame["month"].unique()))
    lookup: dict[tuple[str, pd.Timestamp], float] = {}
    for location in frame["location"].unique():
        counts = holiday_counts_by_month(holidays, months, [location])
        for month, value in counts.items():
            lookup[(location, month)] = float(value)
    values = [lookup[(loc, month)] for loc, month in zip(frame["location"], frame["month"])]
    return pd.Series(values, index=frame.index, dtype=float)


def extend_for_future(
    frame: pd.DataFrame,
    holidays: pd.DataFrame,
    horizon: int,
    active_lookback_months: int,
    fy_start_month: int = FY_START_MONTH,
) -> pd.DataFrame:
    """Append `horizon` future months for employees active in the last `active_lookback_months` of history.

    Future rows have a missing target; the recursive forecast fills it in one month at a time.
    """
    last_month = frame["month"].max()
    cutoff = last_month - pd.DateOffset(months=active_lookback_months - 1)
    recent = frame.loc[frame["month"] >= cutoff].sort_values("month")
    active = recent.groupby("member_id").tail(1)[["member_id", "location"]].reset_index(drop=True)

    future_months = pd.date_range(last_month + pd.DateOffset(months=1), periods=horizon, freq="MS")
    future = active.merge(pd.DataFrame({"month": future_months}), how="cross")
    future["pag"] = frame["pag"].iat[0]
    future[TARGET] = np.nan
    future["vacation_occurrences"] = np.nan
    future["actual_event"] = np.nan
    future["is_future"] = True
    future = _add_calendar(future, holidays, fy_start_month)
    combined = pd.concat([frame, future], ignore_index=True)
    return combined.sort_values(KEY).reset_index(drop=True)


def _lookup(series: pd.Series, frame: pd.DataFrame, months_back: int) -> np.ndarray:
    """Value of `series` for the same employee `months_back` months earlier; NaN where no row exists."""
    index = pd.MultiIndex.from_arrays(
        [frame["member_id"], frame["month"] - pd.DateOffset(months=months_back)], names=KEY
    )
    return series.reindex(index).to_numpy(dtype=float)


def _lookback_stack(series: pd.Series, frame: pd.DataFrame, lags: range) -> np.ndarray:
    return np.column_stack([_lookup(series, frame, lag) for lag in lags])


def _nan_sum(stack: np.ndarray) -> np.ndarray:
    """Row sums that skip missing values; a row with no observed value stays NaN."""
    return np.where(np.isnan(stack).all(axis=1), np.nan, np.nansum(stack, axis=1))


def _event_window(events: pd.Series, f: pd.DataFrame) -> np.ndarray:
    """Twelve actual-event lags per row. Future rows read the 12 months ending at the last actual month."""
    origin = f.loc[~f["is_future"], "month"].max()
    stack = np.full((len(f), 12), np.nan)
    for month in f["month"].unique():
        month = pd.Timestamp(month)
        rows = (f["month"] == month).to_numpy()
        anchor = min(month, origin + pd.DateOffset(months=1))
        gap = (month.year - anchor.year) * 12 + (month.month - anchor.month)
        sub = f.loc[rows]
        for j in range(12):
            stack[rows, j] = _lookup(events, sub, gap + j + 1)
    return stack


def add_employee_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add history features. Each feature for month m uses only rows dated before m."""
    f = frame.sort_values(KEY).reset_index(drop=True)
    hours = f.set_index(KEY)[TARGET]
    occurrences = f.set_index(KEY)["vacation_occurrences"]

    f["vacation_lag_1"] = _lookup(hours, f, 1)
    f["vacation_lag_2"] = _lookup(hours, f, 2)
    f["vacation_same_month_last_year"] = _lookup(hours, f, 12)

    hours_12 = _lookback_stack(hours, f, range(1, 13))
    f["vacation_prev3_sum"] = _nan_sum(hours_12[:, :3])
    f["vacation_prev12_sum"] = _nan_sum(hours_12)
    event_12 = _event_window(f.set_index(KEY)["actual_event"], f)
    f["vacation_months_prev12"] = np.where(
        np.isnan(event_12).all(axis=1), np.nan, (np.nan_to_num(event_12) > 0).sum(axis=1)
    )
    f["vacation_occurrences_prev12"] = _nan_sum(_lookback_stack(occurrences, f, range(1, 13)))
    f["prior_months_observed"] = f.groupby("member_id").cumcount().astype(float)

    grouped = f.groupby("member_id")
    f["hist_avg_vacation"] = grouped[TARGET].transform(lambda s: s.shift(1).expanding().mean())
    f["hist_max_vacation"] = grouped[TARGET].transform(lambda s: s.shift(1).expanding().max())
    f["hist_vacation_frequency"] = grouped["actual_event"].transform(lambda s: s.shift(1).expanding().mean())
    return f
