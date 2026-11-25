"""Real-data validation for the Solvay vacation dataset (FY23 to FY25).

Loads the raw workbooks with the same loader and preprocessing as the pipeline, then computes descriptive
totals, data-quality checks and holiday-window patterns. Writes reports/real_data_validation.md.
No forecasting model is fitted or changed.

Run: python -m vacation_forecasting.real_data_validation
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vacation_forecasting.config import UNKNOWN_LOCATION, VACATION_TRC_CODES, Settings, get_settings
from vacation_forecasting.data_loader import (
    SOURCE_FILE_COLUMN,
    discover_timesheet_files,
    load_timesheet_file,
    sheet_for,
)
from vacation_forecasting.holiday_analysis import (
    applicable_holidays,
    holiday_counts_by_month,
    holiday_window_effect,
    load_holidays,
)
from vacation_forecasting.insights import build_insights
from vacation_forecasting.normalization import financial_year_end, financial_year_label, fy_month_index
from vacation_forecasting.preprocessing import preprocess_timesheets
from vacation_forecasting.reporting import df_to_markdown, render_report
from vacation_forecasting.vacation_analysis import add_vacation_flag, monthly_vacation_summary, select_pag

logger = logging.getLogger(__name__)

SCOPE_FYS = (2023, 2024, 2025)
SCOPE_YEARS = (2023, 2024, 2025)
KEY_FIELDS = (
    "member_id",
    "transaction_date",
    "pag",
    "trc_code",
    "trc_description",
    "hours",
    "location",
    "department_description",
)
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "real_data_validation.md"


@dataclass
class ValidationResult:
    markdown: str
    checks: pd.DataFrame
    fy_summary: pd.DataFrame
    monthly: pd.DataFrame


def _short(name: str) -> str:
    match = re.search(r"FY\d{2}", name)
    return match.group(0) if match else name


def _file_fy(name: str) -> float:
    """Financial year end implied by the file name (FY24 file -> 2024), or NaN."""
    match = re.search(r"FY(\d{2})", name)
    return float(2000 + int(match.group(1))) if match else float("nan")


def _fy_bounds(fy: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last day of a financial year (Oct to Sep)."""
    return pd.Timestamp(year=fy - 1, month=10, day=1), pd.Timestamp(year=fy, month=9, day=30)


def _md(frame: pd.DataFrame, digits: int = 1) -> str:
    return df_to_markdown(frame, float_digits=digits) if not frame.empty else "_No rows._"


def _load_files(settings: Settings) -> tuple[pd.DataFrame, dict[str, dict[str, str]], pd.DataFrame]:
    """Load every workbook through the pipeline's per-file loader and keep each file's column mapping."""
    files = discover_timesheet_files(settings.raw_dir, exclude=settings.holidays_csv)
    frames: list[pd.DataFrame] = []
    mappings: dict[str, dict[str, str]] = {}
    rows: list[dict[str, object]] = []
    for path in files:
        frame, mapping = load_timesheet_file(path, settings)
        frames.append(frame)
        mappings[path.name] = mapping
        rows.append(
            {
                "file": path.name,
                "sheet": str(sheet_for(path, settings)),
                "header row": settings.header_overrides.get(path.stem, settings.header_row) + 1,
                "rows loaded": len(frame),
            }
        )
    return pd.concat(frames, ignore_index=True), mappings, pd.DataFrame(rows)


def _fy_table(solvay: pd.DataFrame) -> pd.DataFrame:
    """Per financial year: date coverage, employees, hours, vacation totals and year-on-year change."""
    vac = solvay.loc[solvay["is_vacation"]]
    base = solvay.groupby("financial_year_label").agg(
        first_date=("transaction_date", "min"),
        last_date=("transaction_date", "max"),
        records=("member_id", "size"),
        employees=("member_id", "nunique"),
        total_hours=("hours", "sum"),
    )
    vacation = vac.groupby("financial_year_label").agg(
        vacation_records=("member_id", "size"),
        vacation_employees=("member_id", "nunique"),
        vacation_hours=("hours", "sum"),
    )
    fy = base.join(vacation, how="left").fillna(0).reset_index().sort_values("financial_year_label")
    fy = fy.reset_index(drop=True)
    fy["first_date"] = fy["first_date"].dt.strftime("%Y-%m-%d")
    fy["last_date"] = fy["last_date"].dt.strftime("%Y-%m-%d")
    fy["vacation_share_pct"] = fy["vacation_hours"] / fy["total_hours"] * 100
    for column in ("employees", "vacation_hours", "vacation_employees"):
        fy[f"{column}_yoy_pct"] = fy[column].pct_change() * 100
    return fy


def _monthly_table(
    solvay: pd.DataFrame,
    holidays: pd.DataFrame,
    locations: list[str],
    settings: Settings,
    dataset_max: pd.Timestamp,
) -> pd.DataFrame:
    """Monthly vacation hours, vacation and active employee counts, applicable holidays and FY labels."""
    monthly = monthly_vacation_summary(solvay)
    vac = solvay.loc[solvay["is_vacation"]]
    counts = vac.groupby("month").agg(
        vacation_records=("member_id", "size"),
        vacation_employees=("member_id", "nunique"),
    ).reset_index()
    monthly = monthly.merge(counts, on="month", how="left")
    monthly[["vacation_records", "vacation_employees"]] = (
        monthly[["vacation_records", "vacation_employees"]].fillna(0).astype(int)
    )
    active = solvay.groupby("month")["member_id"].nunique()
    monthly["active_employees"] = active.reindex(monthly["month"]).fillna(0).astype(int).to_numpy()
    monthly["applicable_holidays"] = holiday_counts_by_month(
        holidays, pd.DatetimeIndex(monthly["month"]), locations
    ).to_numpy()
    monthly["financial_year_label"] = financial_year_end(monthly["month"], settings.fy_start_month).map(
        financial_year_label
    )
    monthly["fy_month"] = fy_month_index(monthly["month"], settings.fy_start_month).astype(int)
    month_end = monthly["month"] + pd.offsets.MonthEnd(0)
    monthly["partial"] = monthly["month"].eq(dataset_max.to_period("M").to_timestamp()) & (
        dataset_max < month_end
    )
    return monthly


def _same_period_yoy(monthly: pd.DataFrame) -> pd.DataFrame:
    """Oct to Aug totals per FY (11 months), so that the comparison excludes the partial September 2025."""
    part = monthly.loc[monthly["fy_month"] <= 11]
    table = part.groupby("financial_year_label").agg(
        vacation_hours=("vacation_hours", "sum"),
        months=("month", "size"),
    ).reset_index()
    table["yoy_pct"] = table["vacation_hours"].pct_change() * 100
    return table


def _extremes(complete: pd.DataFrame) -> pd.DataFrame:
    """Highest and lowest complete months per FY and across all complete months."""
    rows: list[dict[str, object]] = []
    groups = list(complete.groupby("financial_year_label")) + [("All complete months", complete)]
    for label, group in groups:
        for kind, idx in (
            ("Highest", group["vacation_hours"].idxmax()),
            ("Lowest", group["vacation_hours"].idxmin()),
        ):
            row = group.loc[idx]
            rows.append(
                {
                    "period": label,
                    "type": kind,
                    "month": f"{row['month']:%b %Y}",
                    "vacation hours": row["vacation_hours"],
                    "vacation employees": row["vacation_employees"],
                }
            )
    return pd.DataFrame(rows)


def _location_table(solvay: pd.DataFrame, fy_labels: list[str]) -> pd.DataFrame:
    """Per location: employees (primary location in each FY) and vacation hours per FY."""
    vac = solvay.loc[solvay["is_vacation"]]
    primary = (
        solvay.groupby(["financial_year_label", "member_id"])["location"]
        .agg(lambda values: values.mode().iloc[0])
        .rename("primary_location")
        .reset_index()
    )
    employees = (
        primary.groupby(["primary_location", "financial_year_label"]).size()
        .unstack(fill_value=0)
        .reindex(columns=fy_labels, fill_value=0)
    )
    hours = (
        vac.groupby(["location", "financial_year_label"])["hours"].sum()
        .unstack(fill_value=0)
        .reindex(columns=fy_labels, fill_value=0)
    )
    locations = sorted(set(employees.index) | set(hours.index))
    employees = employees.reindex(locations, fill_value=0)
    hours = hours.reindex(locations, fill_value=0)
    table = pd.DataFrame({"location": locations})
    for label in fy_labels:
        table[f"{label} employees"] = employees[label].to_numpy()
        table[f"{label} vacation hours"] = hours[label].to_numpy()
    return table


def _holiday_applicability(holidays: pd.DataFrame, locations: list[str], fy_labels: list[str]) -> pd.DataFrame:
    """Holidays applicable to each location (India-wide plus location rows) inside each FY window."""
    rows: list[dict[str, object]] = []
    for location in locations:
        applicable = applicable_holidays(holidays, [location])
        row: dict[str, object] = {
            "location": location,
            "CSV rows for location": int((holidays["location"] == location).sum()),
        }
        for label in fy_labels:
            start, end = _fy_bounds(int(label[2:]))
            row[f"{label} applicable holidays"] = int(applicable["date"].between(start, end).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_rows(holidays: pd.DataFrame, fy_labels: list[str]) -> pd.DataFrame:
    """Which calendar years of the holiday CSV each FY window needs, and which are missing."""
    years = set(holidays["year"].astype(int))
    rows: list[dict[str, str]] = []
    for label in fy_labels:
        fy = int(label[2:])
        start, end = _fy_bounds(fy)
        needed = sorted({fy - 1, fy})
        missing = [year for year in needed if year not in years]
        rows.append(
            {
                "financial year": label,
                "window": f"{start:%b %Y} to {end:%b %Y}",
                "CSV years needed": ", ".join(map(str, needed)),
                "missing CSV years": ", ".join(map(str, missing)) or "none",
            }
        )
    return pd.DataFrame(rows)


def _holiday_window(
    solvay: pd.DataFrame,
    holidays: pd.DataFrame,
    locations: list[str],
    window_days: int,
    dataset_max: pd.Timestamp,
) -> pd.DataFrame:
    """Average daily vacation hours around each applicable holiday, using weekdays only."""
    vac = solvay.loc[solvay["is_vacation"]]
    start = _fy_bounds(SCOPE_FYS[0])[0]
    last_day = dataset_max.normalize()
    per_day = vac.groupby(vac["transaction_date"].dt.normalize())["hours"].sum()
    weekdays = pd.bdate_range(start, last_day)
    daily = per_day.reindex(weekdays, fill_value=0.0)
    applicable = applicable_holidays(holidays, locations)
    inner = applicable.loc[
        applicable["date"].between(
            start + pd.Timedelta(days=window_days), last_day - pd.Timedelta(days=window_days)
        )
    ]
    return holiday_window_effect(daily, inner, window_days)


def run_validation(settings: Settings) -> ValidationResult:
    """Run every validation step and return the rendered report with its supporting tables."""
    raw, mappings, files = _load_files(settings)
    pre = preprocess_timesheets(raw, settings)
    clean = pre.clean.copy()
    rejected = pre.rejected.copy()

    clean["file_fy"] = clean[SOURCE_FILE_COLUMN].map(_file_fy).astype("float64")
    clean["outside_file_window"] = clean["file_fy"].notna() & clean["financial_year"].astype("float64").ne(
        clean["file_fy"]
    )
    source_year = clean["Year"] if "Year" in clean.columns else pd.Series(None, index=clean.index, dtype="object")
    # Normalise source labels ("FY23", "FY2023") to a 4-digit year before comparing.
    source_digits = pd.to_numeric(
        source_year.astype(str).str.extract(r"(\d{2,4})", expand=False), errors="coerce"
    )
    source_fy = source_digits.where(source_digits >= 100, source_digits + 2000)
    clean["label_mismatch"] = (
        source_year.notna()
        & source_fy.notna()
        & source_fy.ne(clean["financial_year"].astype("float64"))
    )
    dataset_max = clean["transaction_date"].max()

    scope = clean.loc[clean["financial_year"].isin(SCOPE_FYS)]
    fy22_rows = int(clean["financial_year"].isin([2022]).sum())
    variants = scope.loc[
        scope["pag"].fillna("").str.contains("solvay", case=False) & scope["pag"].ne(settings.pag)
    ]
    solvay = add_vacation_flag(select_pag(scope, settings.pag))
    vac = solvay.loc[solvay["is_vacation"]]
    locations = sorted(loc for loc in solvay["location"].unique() if loc != UNKNOWN_LOCATION)
    fy_labels = sorted(solvay["financial_year_label"].dropna().unique())

    holidays = load_holidays(settings.holidays_csv)
    fy = _fy_table(solvay)
    monthly = _monthly_table(solvay, holidays, locations, settings, dataset_max)
    complete = monthly.loc[~monthly["partial"]]
    calendar_profile = (
        complete.assign(calendar_month=complete["month"].dt.month)
        .groupby("calendar_month")["vacation_hours"]
        .mean()
        .rename("avg_vacation_hours")
        .reset_index()
    )
    same_period = _same_period_yoy(monthly)
    extremes = _extremes(complete)
    location_table = _location_table(solvay, fy_labels)
    holiday_table = _holiday_applicability(holidays, locations, fy_labels)
    coverage = _coverage_rows(holidays, fy_labels)
    effect = _holiday_window(solvay, holidays, locations, settings.holiday_window_days, dataset_max)

    # Per-file statistics.
    file_stats = clean.groupby(SOURCE_FILE_COLUMN).agg(
        first_date=("transaction_date", "min"),
        last_date=("transaction_date", "max"),
        rows_outside_file_fy=("outside_file_window", "sum"),
        year_label_mismatches=("label_mismatch", "sum"),
    )
    rejected_counts = rejected.groupby(SOURCE_FILE_COLUMN).size().rename("rejected rows") if not rejected.empty else None
    file_table = files.set_index("file").join(file_stats)
    if rejected_counts is not None:
        file_table = file_table.join(rejected_counts)
    file_table = file_table.fillna(0).reset_index()
    for column in ("first_date", "last_date"):
        file_table[column] = pd.to_datetime(file_table[column]).dt.strftime("%Y-%m-%d")
    file_table["file"] = file_table["file"].map(_short)

    # Column mapping per file.
    mapping_rows = []
    for field in KEY_FIELDS:
        row: dict[str, str] = {"canonical field": field}
        for file_name, mapping in mappings.items():
            row[_short(file_name)] = mapping.get(field, "not present")
        mapping_rows.append(row)
    mapping_table = pd.DataFrame(mapping_rows)

    # Data-quality checks.
    checks: list[dict[str, str]] = []

    def add(area: str, check: str, result: str, status: str) -> None:
        checks.append({"area": area, "check": check, "result": result, "status": status})

    add(
        "Sources",
        "Timesheet files loaded",
        ", ".join(files["file"].map(_short)),
        "PASS" if len(files) == 3 else "WARN",
    )
    add("Sources", "Required columns resolved in each file", "All required canonical columns found", "PASS")
    add(
        "Sources",
        "Row reconciliation",
        f"{len(raw):,} loaded = {len(clean):,} clean + {len(rejected):,} rejected",
        "PASS" if len(raw) == len(clean) + len(rejected) else "FAIL",
    )
    reasons = rejected["rejection_reason"].value_counts() if not rejected.empty else pd.Series(dtype="int64")
    add(
        "Sources",
        "Rejected rows (missing or invalid key fields)",
        "; ".join(f"{reason}: {count:,}" for reason, count in reasons.items()) or "none",
        "PASS" if rejected.empty else "WARN",
    )
    add(
        "Dates",
        "Transaction date range (all PAGs)",
        f"{clean['transaction_date'].min():%Y-%m-%d} to {dataset_max:%Y-%m-%d}",
        "INFO",
    )
    outside = clean.groupby(SOURCE_FILE_COLUMN)["outside_file_window"].sum()
    add(
        "Dates",
        "Rows outside the Oct-Sep FY implied by their file name",
        "; ".join(f"{_short(name)}: {count:,}" for name, count in outside.items())
        + " (assigned to the FY of the transaction date)",
        "WARN" if outside.sum() else "PASS",
    )
    mismatch = clean.groupby(SOURCE_FILE_COLUMN)["label_mismatch"].sum()
    has_year = clean.assign(has_year=source_year.notna()).groupby(SOURCE_FILE_COLUMN)["has_year"].any()
    add(
        "Dates",
        "Source Year label vs Oct-Sep date rule",
        "; ".join(
            f"{_short(name)}: {count:,} rows" if has_year[name] else f"{_short(name)}: no Year column"
            for name, count in mismatch.items()
        ),
        "WARN" if mismatch.sum() else "PASS",
    )
    add(
        "Scope",
        "FY2022 rows (Sep 2022) outside FY23-FY25 scope",
        f"{fy22_rows:,} rows excluded from metrics",
        "INFO",
    )
    variant_text = "; ".join(
        f"{name}: {len(group):,} rows, {group.loc[group['trc_code'].isin(VACATION_TRC_CODES), 'hours'].sum():,.1f} vacation h"
        for name, group in variants.groupby("pag")
    )
    add("Scope", "Solvay variant PAGs excluded (in FY23-FY25)", variant_text or "none", "INFO")
    descriptions = sorted(vac["trc_description"].dropna().unique())
    add(
        "Vacation",
        "TRC 105 descriptions in Solvay vacation rows",
        ", ".join(descriptions) or "none",
        "PASS" if descriptions == ["Vacation"] else "WARN",
    )
    non_positive = int((vac["hours"] <= 0).sum())
    add(
        "Vacation",
        "Non-positive vacation hours",
        f"{non_positive:,} rows",
        "PASS" if non_positive == 0 else "WARN",
    )
    day_hours = solvay.groupby(["member_id", solvay["transaction_date"].dt.normalize()])["hours"].sum()
    over_24 = int((day_hours > 24).sum())
    add(
        "Hours",
        "Member-days above 24 total hours (Solvay, all TRC)",
        f"{over_24:,} member-days",
        "PASS" if over_24 == 0 else "WARN",
    )
    dup_mask = vac.duplicated(subset=["member_id", "transaction_date", "trc_code", "hours"], keep="first")
    add(
        "Duplicates",
        "Repeated vacation records (same member, date, TRC and hours)",
        f"{int(dup_mask.sum()):,} extra rows, {vac.loc[dup_mask, 'hours'].sum():,.1f} h",
        "PASS" if not dup_mask.any() else "WARN",
    )
    add(
        "Location",
        "Location source for Solvay rows",
        "; ".join(f"{name}: {count:,}" for name, count in solvay["location_source"].value_counts().items()),
        "INFO",
    )
    unknown = int((vac["location"] == UNKNOWN_LOCATION).sum())
    add(
        "Location",
        "Vacation rows with unresolved location",
        f"{unknown:,} rows",
        "PASS" if unknown == 0 else "WARN",
    )
    add(
        "Holidays",
        "Holiday CSV parsed and validated",
        f"{len(holidays):,} rows; years {sorted(holidays['year'].unique().tolist())}",
        "PASS",
    )
    csv_years = set(holidays["year"].astype(int))
    missing_years = sorted(
        {year for label in fy_labels for year in (int(label[2:]) - 1, int(label[2:])) if year not in csv_years}
    )
    add(
        "Holidays",
        "Calendar years covering FY23-FY25 windows",
        f"missing: {', '.join(map(str, missing_years)) or 'none'}",
        "WARN" if missing_years else "PASS",
    )
    for location in locations:
        loc_rows = holidays.loc[holidays["location"] == location]
        loc_years = set(loc_rows["year"].astype(int))
        gaps = [year for year in SCOPE_YEARS if year not in loc_years]
        result = f"{len(loc_rows)} location-specific rows"
        if gaps:
            result += f"; none for {', '.join(map(str, gaps))}"
        add("Holidays", f"Location holidays for {location}", result, "WARN" if gaps else "PASS")
    total_vacation = float(vac["hours"].sum())
    fy_total = float(fy["vacation_hours"].sum())
    month_total = float(monthly["vacation_hours"].sum())
    location_total = float(location_table.filter(like="vacation hours").to_numpy().sum())
    reconciled = all(abs(value - total_vacation) < 1e-6 for value in (fy_total, month_total, location_total))
    add(
        "Reconciliation",
        "Vacation hours: FY vs monthly vs location totals",
        f"{total_vacation:,.1f} h in all three views",
        "PASS" if reconciled else "FAIL",
    )
    expected_end = _fy_bounds(SCOPE_FYS[-1])[1]
    add(
        "Coverage",
        "Latest FY in scope fully covered",
        f"data ends {dataset_max:%Y-%m-%d}; FY2025 ends {expected_end:%Y-%m-%d}",
        "PASS" if dataset_max >= expected_end else "WARN",
    )
    checks_table = pd.DataFrame(checks)

    # Insights are computed from complete months only; the partial September 2025 is excluded.
    insights = build_insights(
        complete,
        calendar_profile,
        fy,
        pd.DataFrame(),
        "",
        effect,
        None,
    )

    open_issues = checks_table.loc[checks_table["status"].isin(["WARN", "FAIL"])]
    monthly_display = pd.DataFrame(
        {
            "Month": monthly["month"].dt.strftime("%Y-%m") + monthly["partial"].map({True: " (partial)", False: ""}),
            "FY": monthly["financial_year_label"],
            "Vacation hours": monthly["vacation_hours"],
            "Vacation records": monthly["vacation_records"],
            "Vacation employees": monthly["vacation_employees"],
            "Active employees": monthly["active_employees"],
            "MoM %": monthly["mom_change_pct"],
            "YoY %": monthly["yoy_change_pct"],
            "Applicable holidays": monthly["applicable_holidays"],
        }
    )
    fy_display = fy.rename(
        columns={
            "financial_year_label": "FY",
            "first_date": "First date",
            "last_date": "Last date",
            "records": "Solvay records",
            "employees": "Employees",
            "total_hours": "Total hours",
            "vacation_records": "Vacation records",
            "vacation_employees": "Vacation employees",
            "vacation_hours": "Vacation hours",
            "vacation_share_pct": "Vacation share %",
            "employees_yoy_pct": "Employees YoY %",
            "vacation_employees_yoy_pct": "Vacation employees YoY %",
            "vacation_hours_yoy_pct": "Vacation hours YoY %",
        }
    )
    fy_display_cols = [
        "FY", "First date", "Last date", "Solvay records", "Employees", "Total hours",
        "Vacation records", "Vacation employees", "Vacation hours", "Vacation share %",
        "Employees YoY %", "Vacation employees YoY %", "Vacation hours YoY %",
    ]
    fy_display = fy_display[fy_display_cols]

    if effect.empty:
        effect_text = "_No holiday windows fall fully inside the data range._"
        effect_summary = ""
    else:
        ordered = effect.dropna(subset=["difference_hours"]).sort_values("difference_hours", ascending=False)
        top_bottom = pd.concat([ordered.head(5), ordered.tail(5)]).drop_duplicates()
        effect_text = _md(
            top_bottom.rename(
                columns={
                    "holiday_name": "Holiday",
                    "date": "Date",
                    "window_avg_hours": "Window avg total vacation hours per day",
                    "rest_avg_hours": "Other days avg total vacation hours per day",
                    "difference_hours": "Difference total vacation hours per day",
                }
            ),
            2,
        )
        effect_summary = (
            f"{len(ordered)} holidays evaluated; {int((ordered['difference_hours'] > 0).sum())} with a higher "
            f"window average. Median difference {ordered['difference_hours'].median():+.2f} total vacation hours per day. "
            "Weekdays only, +/- "
            f"{settings.holiday_window_days} days. Descriptive timing association only."
        )

    scope_text = "\n".join(
        [
            f"- Sources: {len(files)} member-level workbooks and `holidays.csv` ({len(holidays)} rows), "
            "read through the pipeline loader and preprocessing.",
            "- PAG: Solvay only; Solvay Pool and other variants are excluded.",
            "- Financial year: Oct to Sep, assigned from the transaction date. The source `Year` label is not used.",
            "- Vacation: TRC code 105.",
            "- Location: the workbooks have no location column. The location code in DEPT DESC "
            "(HYD, BAN/BLR, CHE/CHN, MUM, PUN/PUNE) is mapped to the holiday-CSV location name.",
            f"- Scope: FY23 to FY25. Data ends {dataset_max:%Y-%m-%d}, so September 2025 is partial and "
            "excluded from averages, extremes and insights.",
        ]
    )

    sections = [
        ("A", "Scope and method", scope_text),
        ("B", "Source files and column mapping", _md(file_table) + "\n\n" + _md(mapping_table)),
        ("C", "Data-quality checks", _md(checks_table)),
        ("D", "Financial-year totals and year-on-year change", _md(fy_display)),
        (
            "E",
            "Monthly vacation hours and employee counts",
            _md(monthly_display)
            + "\n\nSeptember 2025 is partial because the data ends on "
            f"{dataset_max:%Y-%m-%d}.",
        ),
        ("F", "Same-period year-on-year (Oct to Aug, 11 complete months)", _md(same_period)),
        ("G", "Highest and lowest complete months", _md(extremes)),
        (
            "H",
            "Locations and holiday applicability",
            "**Employees and vacation hours by location**\n\n"
            + _md(location_table)
            + "\n\n**Holidays applicable to each location**\n\n"
            + _md(holiday_table)
            + "\n\n**Holiday calendar coverage by FY window**\n\n"
            + _md(coverage),
        ),
        ("I", "Holiday-window patterns (weekdays, descriptive only)", effect_text + "\n\n" + effect_summary),
        ("J", "Insights", "\n".join(f"- {line}" for line in insights) or "_No insights._"),
        (
            "K",
            "Open issues and decisions",
            "\n".join(
                f"- **{row['area']}**: {row['check']} - {row['result']} ({row['status']})"
                for _, row in open_issues.iterrows()
            )
            or "_No open issues._",
        ),
    ]
    markdown = render_report("Real Data Validation: Solvay Vacation Dataset (FY23 to FY25)", sections)
    return ValidationResult(markdown=markdown, checks=checks_table, fy_summary=fy, monthly=monthly)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    result = run_validation(settings)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(result.markdown, encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    print(result.checks["status"].value_counts().to_string())
    print(
        result.fy_summary[["financial_year_label", "vacation_hours", "vacation_employees", "employees"]]
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
