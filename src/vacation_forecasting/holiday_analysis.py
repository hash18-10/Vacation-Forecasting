"""Holiday calendar loading and location-aware holiday analysis.

CSV schema: date (YYYY-MM-DD), holiday_name, holiday_type, location, source_file, year.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from vacation_forecasting.config import INDIA_LOCATION

logger = logging.getLogger(__name__)

HOLIDAY_COLUMNS = ("date", "holiday_name", "holiday_type", "location", "source_file", "year")
HOLIDAY_TYPES = ("Mandatory", "Restricted/Optional")
DEDUP_COLUMNS = ["date", "holiday_name", "holiday_type", "location"]


def load_holidays(path: Path) -> pd.DataFrame:
    """Read the holiday calendar. Missing file returns an empty calendar with a warning.

    Dates must be ISO YYYY-MM-DD, `year` must match the date, and `holiday_type` must be a
    known type. Invalid rows raise instead of being silently dropped. Exact duplicates are removed.
    """
    if not path.exists():
        logger.warning("Holiday file not found at %s; holiday features will be zero", path)
        return _empty_holidays()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in HOLIDAY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} must contain columns {HOLIDAY_COLUMNS}; missing {missing}")
    frame = frame[list(HOLIDAY_COLUMNS)].apply(lambda column: column.str.strip())

    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if (frame["location"] == "").any():
        raise ValueError(f"{path.name} has rows with an empty location")
    bad_types = sorted(set(frame["holiday_type"]) - set(HOLIDAY_TYPES))
    if bad_types:
        raise ValueError(f"{path.name} has unknown holiday_type values: {bad_types}")
    years = frame["year"].astype(int)
    mismatched = frame.loc[years != frame["date"].dt.year, ["date", "year"]]
    if not mismatched.empty:
        raise ValueError(f"{path.name} has year values that do not match the date:\n{mismatched}")

    frame["year"] = years
    frame = frame.drop_duplicates(subset=DEDUP_COLUMNS).reset_index(drop=True)
    return frame


def _empty_holidays() -> pd.DataFrame:
    return pd.DataFrame(columns=list(HOLIDAY_COLUMNS)).astype({"date": "datetime64[ns]", "year": "int64"})


def applicable_holidays(holidays: pd.DataFrame, locations: list[str] | None = None) -> pd.DataFrame:
    """Holidays that apply to the given member locations.

    India-wide rows always apply. Location-specific rows apply only if their location is in `locations`.
    A holiday listed both India-wide and for a location is kept once.
    """
    if holidays.empty:
        return holidays
    wanted = {INDIA_LOCATION, *(locations or [])}
    relevant = holidays.loc[holidays["location"].isin(wanted)]
    # India-wide rows sort first so they win when the same holiday also appears for a location.
    relevant = relevant.assign(_national=relevant["location"].eq(INDIA_LOCATION))
    relevant = relevant.sort_values("_national", ascending=False, kind="stable")
    return relevant.drop_duplicates(subset=["date", "holiday_name"]).drop(columns="_national").reset_index(drop=True)


def holiday_counts_by_month(
    holidays: pd.DataFrame, months: pd.DatetimeIndex, locations: list[str] | None = None
) -> pd.Series:
    """Applicable holidays per month for the given member locations (India-wide holidays always count)."""
    relevant = applicable_holidays(holidays, locations)
    if relevant.empty:
        return pd.Series(0, index=months, dtype="int64")
    month_start = relevant["date"].dt.to_period("M").dt.to_timestamp()
    return month_start.value_counts().reindex(months, fill_value=0).astype("int64")


def holiday_window_effect(
    daily_vacation_hours: pd.Series, holidays: pd.DataFrame, window_days: int
) -> pd.DataFrame:
    """Compare average daily vacation hours within +/- `window_days` of each holiday with all other days.

    Descriptive only: it does not establish that holidays cause changes in vacation.
    """
    columns = ["holiday_name", "date", "window_avg_hours", "rest_avg_hours", "difference_hours"]
    if holidays.empty or daily_vacation_hours.empty:
        return pd.DataFrame(columns=columns)
    series = daily_vacation_hours.copy()
    series.index = pd.DatetimeIndex(series.index).normalize()
    rows = []
    for _, holiday in holidays.iterrows():
        day = holiday["date"].normalize()
        mask = (series.index >= day - pd.Timedelta(days=window_days)) & (
            series.index <= day + pd.Timedelta(days=window_days)
        )
        window_avg = float(series[mask].mean()) if mask.any() else float("nan")
        rest_avg = float(series[~mask].mean()) if (~mask).any() else float("nan")
        rows.append(
            {
                "holiday_name": holiday["holiday_name"],
                "date": day.date(),
                "window_avg_hours": window_avg,
                "rest_avg_hours": rest_avg,
                "difference_hours": window_avg - rest_avg,
            }
        )
    return pd.DataFrame(rows, columns=columns)
