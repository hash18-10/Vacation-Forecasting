"""Member-month panel and leakage-safe features.

Every history feature is built from values strictly before the target month (shift >= 1),
so a row never sees its own target. Calendar and holiday features are known in advance and are allowed.
"""

from __future__ import annotations

import pandas as pd

from vacation_forecasting.normalization import fy_month_index

FEATURE_COLUMNS: tuple[str, ...] = (
    "month_of_year",
    "fy_month_index",
    "holiday_count",
    "months_since_start",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "roll_mean_3",
    "roll_mean_12",
    "member_expanding_mean",
    "pag_total_lag_1",
    "pag_total_lag_12",
    "active_members_lag_1",
)
TARGET = "vacation_hours"


def build_member_month_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per member per calendar month between the member's first and last activity.

    `vacation_hours` is the sum of hours on TRC-classified vacation rows; months without vacation are 0.
    """
    if df.empty:
        raise ValueError("Cannot build a panel from an empty dataset")
    months_all = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    span = df.groupby("member_id")["month"].agg(first_month="min", last_month="max").reset_index()
    grid = pd.MultiIndex.from_product([span["member_id"], months_all], names=["member_id", "month"])
    panel = grid.to_frame(index=False).merge(span, on="member_id")
    panel = panel.loc[(panel["month"] >= panel["first_month"]) & (panel["month"] <= panel["last_month"])]
    panel = panel.drop(columns=["first_month", "last_month"])

    vacation = (
        df.loc[df["is_vacation"]]
        .groupby(["member_id", "month"])["hours"]
        .sum()
        .rename(TARGET)
        .reset_index()
    )
    panel = panel.merge(vacation, on=["member_id", "month"], how="left")
    panel[TARGET] = panel[TARGET].fillna(0.0).astype(float)

    last_location = df.sort_values("transaction_date").groupby("member_id")["location"].last()
    panel["location"] = panel["member_id"].map(last_location)
    panel["is_future"] = False
    return panel.sort_values(["member_id", "month"]).reset_index(drop=True)


def build_features(panel: pd.DataFrame, holiday_counts: pd.Series | None = None,
                   fy_start_month: int = 10) -> pd.DataFrame:
    """Add calendar, history and PAG-level features. All history features use only earlier months."""
    p = panel.sort_values(["member_id", "month"]).copy()
    grouped = p.groupby("member_id")[TARGET]

    p["month_of_year"] = p["month"].dt.month
    p["fy_month_index"] = fy_month_index(p["month"], fy_start_month)
    start = p["month"].min()
    p["months_since_start"] = (p["month"].dt.year - start.year) * 12 + (p["month"].dt.month - start.month)

    if holiday_counts is not None:
        p["holiday_count"] = p["month"].map(holiday_counts).fillna(0).astype(float)
    else:
        p["holiday_count"] = 0.0

    for lag in (1, 2, 3, 12):
        p[f"lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    p["roll_mean_3"] = shifted.groupby(p["member_id"]).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    p["roll_mean_12"] = shifted.groupby(p["member_id"]).transform(
        lambda s: s.rolling(12, min_periods=1).mean()
    )
    p["member_expanding_mean"] = shifted.groupby(p["member_id"]).transform(
        lambda s: s.expanding().mean()
    )

    # PAG-level totals and active headcount, shifted so the current month is never used.
    monthly = p.groupby("month").agg(
        total=(TARGET, "sum"), active=("member_id", "nunique")
    )
    p["pag_total_lag_1"] = p["month"].map(monthly["total"].shift(1))
    p["pag_total_lag_12"] = p["month"].map(monthly["total"].shift(12))
    p["active_members_lag_1"] = p["month"].map(monthly["active"].shift(1))
    return p.reset_index(drop=True)


def extend_panel_for_future(panel: pd.DataFrame, horizon: int, active_lookback_months: int = 3) -> pd.DataFrame:
    """Append `horizon` future months for members active in the last `active_lookback_months` of history."""
    last_month = panel["month"].max()
    cutoff = last_month - pd.DateOffset(months=active_lookback_months - 1)
    active = panel.loc[panel["month"] >= cutoff, "member_id"].unique()
    latest = (
        panel.loc[panel["member_id"].isin(active)]
        .sort_values("month")
        .groupby("member_id")
        .last()
        .reset_index()
    )
    future_months = pd.date_range(last_month + pd.DateOffset(months=1), periods=horizon, freq="MS")
    future = latest[["member_id", "location"]].merge(
        pd.DataFrame({"month": future_months}), how="cross"
    )
    future[TARGET] = float("nan")
    future["is_future"] = True
    combined = pd.concat([panel, future], ignore_index=True)
    return combined.sort_values(["member_id", "month"]).reset_index(drop=True)
