"""Central configuration for the vacation forecasting POC.

Business rules, column mappings and model settings live here so that the rest
of the code contains no year-specific or dataset-specific constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Vacation is identified by TRC code only. Descriptions are never used for classification.
VACATION_TRC_CODES: frozenset[str] = frozenset({"105"})
# Leave types counted for an employee: Vacation, Contingency, Sick and Restricted/Mandatory Holiday.
LEAVE_TRC_CODES: frozenset[str] = frozenset({"105", "109", "115", "610"})

# One full vacation day = 9 h. Checked on the real data: 1,846 of 1,886 vacation rows are 9.0 h and 37 are 4.5 h (half days).
FULL_DAY_HOURS: float = 9.0
# Working days are Monday to Friday. Set True to also remove public holidays that apply to the employee's location.
EXCLUDE_PUBLIC_HOLIDAYS: bool = False

# Canonical column -> accepted raw headers. Matching ignores case, spaces and punctuation.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "member_id": ("Employee ID", "Emp ID", "Member ID", "Employee Number", "Emp No"),
    "member_name": ("Employee Name", "Member Name"),
    "transaction_date": ("Transaction Date", "Trans Date", "Txn Date", "Date"),
    "pag": ("PAG", "PAG Name", "Account", "Account Name", "PAG/Account"),
    "trc_code": ("TRC Code", "TRC", "Transaction Code"),
    "trc_description": ("TRC Description", "TRC Desc", "TRC Name"),
    "hours": ("Hours", "Hrs", "Hours Worked", "Hours(Daily)"),
    "location": ("Location", "Work Location", "Office Location"),
    "department_description": ("DEPT DESC", "Dept Desc", "Department Desc"),
    "manager": ("MANAGER", "Manager Name", "Member Supervisor"),
    "director": ("DIRECTOR", "Director Name"),
}
REQUIRED_COLUMNS: tuple[str, ...] = ("member_id", "transaction_date", "pag", "trc_code", "hours")

# Canonical PAG name -> known spelling variants. Matching ignores case and punctuation.
PAG_ALIASES: dict[str, tuple[str, ...]] = {
    "Solvay": ("Solvay SA", "Solvay S.A.", "Solvay Group"),
}

# Financial year starts in October, so FY2024 = Oct 2023 to Sep 2024.
FY_START_MONTH = 10
EXCEL_DATE_ORIGIN = "1899-12-30"
# Fiscal year -> the only file its rows are taken from. The YTD file also holds late Sep 2025 rows of FY2025.
FISCAL_YEAR_SOURCE_FILES: dict[int, str] = {
    2025: "FY25 Cumilative IRR Timesheet_ Oct to Sept 1.xlsb",
}
UNKNOWN_LOCATION = "UNKNOWN"
INDIA_LOCATION = "India"  # national holidays; apply to every member
# Location code in DEPT DESC (e.g. "MIG GHO Solvay Strm5 HYD SEZ") -> location name used in the holiday CSV.
LOCATION_CODE_MAP: dict[str, str] = {
    "HYD": "Hyderabad",
    "BAN": "Bangalore",
    "BLR": "Bangalore",
    "CHE": "Chennai",
    "CHN": "Chennai",
    "MUM": "Mumbai",
    "PUN": "Pune",
    "PUNE": "Pune",
}
# Sheet holding the member-level timesheet in each real workbook; the TS corrections sheets have no TRC codes.
REAL_SHEET_OVERRIDES: dict[str, str] = {
    "FY23 Cumilative IRR Timesheet _Oct to Sept 1": "Memberwise Timesheet Summary",
    "FY24 Cumilative IRR Timesheet Oct to Sept 1": "Memberwise Timesheet Summary",
    "FY25 Cumilative IRR Timesheet_ Oct to Sept 1": "Memberwise Timesheet",
}


@dataclass(frozen=True)
class ModelSettings:
    """Conservative XGBoost parameters to limit overfitting on a small panel."""

    max_depth: int = 3
    learning_rate: float = 0.05
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 5.0
    random_state: int = 42


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the pipeline. Paths are resolved relative to the project root."""

    pag: str = "Solvay"
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "mfg_la")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports" / "output")
    holidays_csv: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "holidays.csv")
    vacation_trc_codes: frozenset[str] = VACATION_TRC_CODES
    # Sheet to read from every workbook unless overridden per file stem, e.g. {"FY2024": "Timesheet"}.
    sheet_name: str | int = 0
    sheet_overrides: dict[str, str | int] = field(default_factory=lambda: dict(REAL_SHEET_OVERRIDES))
    location_code_map: dict[str, str] = field(default_factory=lambda: dict(LOCATION_CODE_MAP))
    header_row: int = 0
    # Header row per file stem where it differs from header_row, e.g. FY24/FY25 have a blank first row.
    header_overrides: dict[str, int] = field(
        default_factory=lambda: {
            "FY24 Cumilative IRR Timesheet Oct to Sept 1": 1,
            "FY25 Cumilative IRR Timesheet_ Oct to Sept 1": 1,
        }
    )
    fy_start_month: int = FY_START_MONTH
    hours_per_vacation_day: float = FULL_DAY_HOURS
    exclude_public_holidays: bool = EXCLUDE_PUBLIC_HOLIDAYS
    forecast_horizon: int = 6
    # Model behind the application forecast. "seasonal_naive" was selected by reports/model_validation.md:
    # no recursive XGBoost model beat it on both MAE and WAPE in the pooled holdout and in every origin.
    production_model: str = "seasonal_naive"
    test_months: int = 6
    holiday_window_days: int = 5
    # Forecast headcount = members with a timesheet row in the last `active_lookback_months` complete months.
    # 1 = present in the latest complete month (point-in-time headcount, same definition as history).
    active_lookback_months: int = 1
    min_history_months: int = 24
    model: ModelSettings = field(default_factory=ModelSettings)


def get_settings() -> Settings:
    """Return default settings with optional environment overrides (no secrets involved)."""
    overrides: dict[str, object] = {}
    if pag := os.getenv("VF_PAG"):
        overrides["pag"] = pag
    if horizon := os.getenv("VF_FORECAST_HORIZON"):
        overrides["forecast_horizon"] = int(horizon)
    if raw_dir := os.getenv("VF_RAW_DIR"):
        overrides["raw_dir"] = Path(raw_dir)
        overrides["holidays_csv"] = Path(raw_dir) / "holidays.csv"
    if output_dir := os.getenv("VF_OUTPUT_DIR"):
        overrides["output_dir"] = Path(output_dir)
    if exclude := os.getenv("VF_EXCLUDE_PUBLIC_HOLIDAYS"):
        overrides["exclude_public_holidays"] = exclude.strip().lower() in {"1", "true", "yes"}
    return replace(Settings(), **overrides)
