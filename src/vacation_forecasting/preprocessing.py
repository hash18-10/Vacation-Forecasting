"""Turn raw timesheet rows into a validated clean dataset plus an auditable rejected set."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from vacation_forecasting.config import PAG_ALIASES, UNKNOWN_LOCATION, Settings
from vacation_forecasting.normalization import (
    clean_text,
    derive_location_series,
    financial_year_end,
    financial_year_label,
    fy_month_index,
    normalize_identifier_series,
    normalize_pag_series,
    parse_transaction_dates,
    to_month_start,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessResult:
    """Clean rows used for analysis, and rejected rows kept with their rejection reasons."""

    clean: pd.DataFrame
    rejected: pd.DataFrame


def preprocess_timesheets(raw: pd.DataFrame, settings: Settings) -> PreprocessResult:
    """Normalize values, flag rows that fail validation and derive month and financial-year columns."""
    if raw.empty:
        raise ValueError("No timesheet rows to preprocess")

    df = raw.copy()
    df["pag_original"] = df["pag"]
    df["pag"] = normalize_pag_series(df["pag"], PAG_ALIASES)
    df["member_id"] = normalize_identifier_series(df["member_id"])
    df["trc_code"] = normalize_identifier_series(df["trc_code"])
    df["transaction_date"] = parse_transaction_dates(df["transaction_date"])
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
    for optional in ("member_name", "trc_description", "location", "department_description"):
        if optional not in df.columns:
            df[optional] = None
    # An explicit location column wins; otherwise the location code is read from the department description.
    source_location = df["location"].map(clean_text).astype("object")
    derived_location = derive_location_series(df["department_description"], settings.location_code_map)
    df["location"] = source_location.fillna(derived_location).fillna(UNKNOWN_LOCATION)
    df["location_source"] = "unknown"
    df.loc[derived_location.notna(), "location_source"] = "department description"
    df.loc[source_location.notna(), "location_source"] = "location column"

    checks = pd.DataFrame(
        {
            "missing_date": df["transaction_date"].isna(),
            "missing_member_id": df["member_id"].isna(),
            "missing_pag": df["pag"].isna(),
            "missing_trc_code": df["trc_code"].isna(),
            "invalid_hours": df["hours"].isna(),
        },
        index=df.index,
    )
    reasons = checks.apply(lambda row: ";".join(name for name, bad in row.items() if bad), axis=1)
    is_rejected = reasons.ne("")

    rejected = df.loc[is_rejected].assign(rejection_reason=reasons[is_rejected])
    clean = df.loc[~is_rejected].copy()
    clean["month"] = to_month_start(clean["transaction_date"])
    fy_end = financial_year_end(clean["transaction_date"], settings.fy_start_month)
    clean["financial_year"] = fy_end
    clean["financial_year_label"] = fy_end.map(financial_year_label)
    clean["fy_month_index"] = fy_month_index(clean["transaction_date"], settings.fy_start_month)

    _log_preprocess_summary(clean, rejected, settings)
    return PreprocessResult(clean=clean, rejected=rejected)


def _log_preprocess_summary(clean: pd.DataFrame, rejected: pd.DataFrame, settings: Settings) -> None:
    """Log row counts, rejection reasons and PAG names that were not in the alias list."""
    logger.info("Preprocessing: %d clean rows, %d rejected rows", len(clean), len(rejected))
    if not rejected.empty:
        for reason, count in rejected["rejection_reason"].value_counts().items():
            logger.warning("Rejected %d row(s): %s", count, reason)
    known = set(PAG_ALIASES)
    unknown_pags = sorted(set(clean["pag"].dropna()) - known)
    if unknown_pags:
        logger.info("PAG names without an alias entry (used as-is): %s", unknown_pags[:20])
    if not clean.empty:
        logger.info(
            "Transaction dates %s to %s; financial years %s",
            clean["transaction_date"].min().date(),
            clean["transaction_date"].max().date(),
            sorted(clean["financial_year_label"].dropna().unique()),
        )
    if len(clean) < settings.min_history_months:
        logger.warning("Only %d clean rows overall; forecasts may be unreliable", len(clean))
