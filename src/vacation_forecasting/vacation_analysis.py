"""Vacation classification (TRC code only) and descriptive summaries used by the report."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from vacation_forecasting.config import VACATION_TRC_CODES


def classify_vacation(df: pd.DataFrame, trc_codes: Iterable[str] = VACATION_TRC_CODES) -> pd.Series:
    """True where the normalized TRC code is a vacation code. Descriptions are never used."""
    codes = set(trc_codes)
    return df["trc_code"].isin(codes).fillna(False).astype(bool)


def add_vacation_flag(df: pd.DataFrame, trc_codes: Iterable[str] = VACATION_TRC_CODES) -> pd.DataFrame:
    """Return a copy with an is_vacation boolean column."""
    out = df.copy()
    out["is_vacation"] = classify_vacation(out, trc_codes)
    return out


def available_pags(df: pd.DataFrame) -> list[str]:
    """Sorted PAG names present in the clean data."""
    return sorted(df["pag"].dropna().unique().tolist())


def select_pag(df: pd.DataFrame, pag: str) -> pd.DataFrame:
    """Rows for one canonical PAG. Raises if the PAG is not present."""
    subset = df.loc[df["pag"] == pag].copy()
    if subset.empty:
        raise ValueError(f"PAG '{pag}' not found. Available: {available_pags(df)}")
    return subset


def headline_metrics(df: pd.DataFrame) -> dict[str, float | int | str | None]:
    """Headline counts for the report, computed from the data."""
    vacation = df.loc[df["is_vacation"]]
    return {
        "total_rows": int(len(df)),
        "vacation_rows": int(len(vacation)),
        "vacation_share_of_rows": float(len(vacation) / len(df)) if len(df) else 0.0,
        "vacation_hours": float(vacation["hours"].sum()),
        "total_hours": float(df["hours"].sum()),
        "distinct_members": int(df["member_id"].nunique()),
        "distinct_members_on_vacation": int(vacation["member_id"].nunique()),
        "first_date": str(df["transaction_date"].min().date()) if len(df) else None,
        "last_date": str(df["transaction_date"].max().date()) if len(df) else None,
    }


def monthly_vacation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Vacation hours per calendar month with month-on-month and year-on-year change."""
    vacation = df.loc[df["is_vacation"]]
    monthly = (
        vacation.groupby("month")["hours"].sum().rename("vacation_hours").to_frame()
    )
    full_index = pd.date_range(
        df["month"].min(), df["month"].max(), freq="MS"
    )
    monthly = monthly.reindex(full_index, fill_value=0.0)
    monthly.index.name = "month"
    monthly["mom_change_pct"] = monthly["vacation_hours"].pct_change(fill_method=None) * 100
    monthly["yoy_change_pct"] = monthly["vacation_hours"].pct_change(12, fill_method=None) * 100
    return monthly.reset_index()


def fy_vacation_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Vacation hours and share of total hours per financial year (Oct to Sep)."""
    grouped = df.groupby("financial_year_label")
    summary = pd.DataFrame(
        {
            "vacation_hours": grouped.apply(
                lambda g: g.loc[g["is_vacation"], "hours"].sum(), include_groups=False
            ),
            "total_hours": grouped["hours"].sum(),
        }
    )
    summary["vacation_share_pct"] = summary["vacation_hours"] / summary["total_hours"] * 100
    return summary.reset_index()


def calendar_month_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average vacation hours by calendar month of year, across all years in the data."""
    monthly = monthly_vacation_summary(df)
    monthly["calendar_month"] = monthly["month"].dt.month
    profile = monthly.groupby("calendar_month")["vacation_hours"].mean().rename("avg_vacation_hours")
    return profile.reset_index()


def vacation_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Share of vacation hours by location, in descending order."""
    vacation = df.loc[df["is_vacation"]]
    dist = vacation.groupby("location")["hours"].sum().sort_values(ascending=False).rename("vacation_hours")
    total = dist.sum()
    out = dist.to_frame()
    out["share_pct"] = out["vacation_hours"] / total * 100 if total else 0.0
    return out.reset_index()


def member_vacation_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-member vacation hours and number of months with any vacation."""
    vacation = df.loc[df["is_vacation"]]
    return (
        vacation.groupby("member_id")
        .agg(vacation_hours=("hours", "sum"), vacation_months=("month", "nunique"))
        .sort_values("vacation_hours", ascending=False)
        .reset_index()
    )


def aggregate_predictions(predictions: pd.DataFrame, value_col: str = "forecast") -> pd.DataFrame:
    """Sum member-level predictions to a PAG-level monthly forecast."""
    return predictions.groupby("month", as_index=False)[value_col].sum()
