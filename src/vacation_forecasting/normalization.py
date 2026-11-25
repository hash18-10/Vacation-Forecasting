"""Value-level normalization: PAG names, identifiers, transaction dates and financial years."""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Real

import pandas as pd

from vacation_forecasting.config import EXCEL_DATE_ORIGIN, FY_START_MONTH, PAG_ALIASES

# Excel serial 1 = 1900-01-01; serial 2958465 = 9999-12-31.
MIN_EXCEL_SERIAL = 1.0
MAX_EXCEL_SERIAL = 2958465.0


def clean_text(value: object) -> str | None:
    """Trim and collapse whitespace; return None for empty or missing values."""
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _match_key(text: str) -> str:
    """Lower-case alphanumeric key used to compare names regardless of punctuation."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


def normalize_pag_series(
    values: pd.Series, aliases: Mapping[str, tuple[str, ...]] = PAG_ALIASES
) -> pd.Series:
    """Map PAG/account spelling variants to their canonical name; unknown names are kept as cleaned text."""
    lookup = {
        _match_key(variant): canonical
        for canonical, variants in aliases.items()
        for variant in (canonical, *variants)
    }

    def _normalize(value: object) -> str | None:
        text = clean_text(value)
        if text is None:
            return None
        return lookup.get(_match_key(text), text)

    return values.map(_normalize).astype("object")


def derive_location_series(values: pd.Series, code_map: Mapping[str, str]) -> pd.Series:
    """Location name from a department description, e.g. "MIG GHO Solvay Strm5 HYD SEZ" -> "Hyderabad".

    Codes are matched as upper-case tokens that start on a word boundary; longer codes win, so PUNE beats PUN.
    Returns None where no configured code is present.
    """
    if not code_map:
        return pd.Series(None, index=values.index, dtype="object")
    codes = sorted(code_map, key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(code) for code in codes) + ")")

    def _derive(value: object) -> str | None:
        text = clean_text(value)
        match = pattern.search(text) if text else None
        return code_map[match.group(1)] if match else None

    return values.map(_derive).astype("object")


def normalize_identifier_series(values: pd.Series) -> pd.Series:
    """Normalize member IDs and TRC codes so that 105, 105.0 and ' 105 ' compare equal."""

    def _normalize(value: object) -> str | None:
        if isinstance(value, Real) and not isinstance(value, bool) and pd.notna(value):
            if float(value).is_integer():
                return str(int(value))
        text = clean_text(value)
        if text is None:
            return None
        return re.sub(r"\.0+$", "", text.upper())

    return values.map(_normalize).astype("object")


def parse_transaction_dates(values: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Convert Excel serial numbers and date text into datetime64 values; unparseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")

    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    numeric = pd.to_numeric(values, errors="coerce")
    serial_mask = numeric.between(MIN_EXCEL_SERIAL, MAX_EXCEL_SERIAL)
    if serial_mask.any():
        result.loc[serial_mask] = pd.to_datetime(
            numeric[serial_mask], unit="D", origin=EXCEL_DATE_ORIGIN
        )
    text_mask = values.notna() & ~serial_mask
    if text_mask.any():
        result.loc[text_mask] = pd.to_datetime(
            values[text_mask].astype(str), errors="coerce", dayfirst=dayfirst
        )
    return result


def financial_year_end(dates: pd.Series, fy_start_month: int = FY_START_MONTH) -> pd.Series:
    """Return the calendar year in which the financial year ends (FY2024 = Oct 2023 to Sep 2024)."""
    years = dates.dt.year
    if fy_start_month == 1:
        fy_end = years
    else:
        fy_end = years + (dates.dt.month >= fy_start_month).astype(int)
    return fy_end.where(dates.notna()).astype("Int64")


def financial_year_label(fy_end: int | None) -> str | None:
    """Format a financial year end as a label such as FY2024."""
    return None if fy_end is None or pd.isna(fy_end) else f"FY{int(fy_end)}"


def fy_month_index(dates: pd.Series, fy_start_month: int = FY_START_MONTH) -> pd.Series:
    """Position of the month within its financial year (1 = first month of the FY)."""
    index = ((dates.dt.month - fy_start_month) % 12) + 1
    return index.where(dates.notna()).astype("Int64")


def to_month_start(dates: pd.Series) -> pd.Series:
    """Truncate dates to the first day of their calendar month."""
    return dates.dt.to_period("M").dt.to_timestamp()
