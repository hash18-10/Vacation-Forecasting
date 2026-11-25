"""Employee-level forecast report: runs the employee-month pipeline and writes reports/employee_level_forecast_evaluation.md.

Three kinds of output are kept apart: HISTORICAL (what happened), FORECAST (what the model predicts) and
FORECAST EXPLANATION (which inputs the model used). Model attributions describe the fitted model, not causes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vacation_forecasting.business_view import BusinessView, build_business_view
from vacation_forecasting.config import Settings, get_settings
from vacation_forecasting.data_loader import load_raw_timesheets
from vacation_forecasting.employee_forecasting import (
    EmployeeForecastRun,
    feature_weights,
    run_employee_level_forecast,
    with_model_labels,
)
from vacation_forecasting.holiday_analysis import load_holidays
from vacation_forecasting.insights import model_comparison_statements
from vacation_forecasting.preprocessing import preprocess_timesheets
from vacation_forecasting.reporting import df_to_markdown, render_report
from vacation_forecasting.vacation_analysis import add_vacation_flag, select_pag

REPORT_NAME = "employee_level_forecast_evaluation.md"


def _summary_line(label: str, summary: dict, period: str) -> str:
    return (
        f"- {label} ({period}, {summary['months']} months): vacation {summary['vacation_pct']:.1f}%, "
        f"availability {summary['availability_pct']:.1f}%, {summary['vacation_days']:,.0f} vacation days, "
        f"{summary['avg_vacation_days_per_employee']:.1f} average vacation days per employee."
    )


def _business_table(monthly: pd.DataFrame) -> str:
    if monthly.empty:
        return "_No rows._"
    table = monthly.copy()
    table["month"] = table["month"].dt.strftime("%b %Y")
    table = table.rename(columns={
        "month": "Month",
        "weekdays": "Weekdays",
        "employees": "Employees",
        "available_person_days": "Available person-days",
        "vacation_days": "Vacation days",
        "vacation_pct": "Vacation %",
        "availability_pct": "Availability %",
        "avg_vacation_days_per_employee": "Avg vacation days per employee",
    })
    return df_to_markdown(table, float_digits=1)


def _methodology(business: BusinessView, settings: Settings) -> str:
    holiday_text = (
        "on. Weekday public holidays for each employee's location and India-wide holidays are removed."
        if business.exclude_public_holidays
        else "off. Every Monday to Friday is a working day."
    )
    partial = ", ".join(f"{m:%b %Y}" for m in business.excluded_partial_months) or "none"
    return "\n".join([
        f"- Vacation day = vacation hours / {business.hours_per_vacation_day:g} h. Full days are 9 h and half days "
        "4.5 h, so a half day counts as 0.5.",
        "- Working days are Monday to Friday. Weekends are never counted.",
        f"- Public holiday exclusion: {holiday_text}",
        "- Employees (history): members with at least one timesheet row in the month. Employees (forecast): "
        "members the model forecasts, taken from those with a timesheet row in the latest complete month "
        f"(lookback = {settings.active_lookback_months} month(s)). The same headcount is held for every forecast month.",
        "- Vacation % = vacation days / available person-days x 100. Availability % = 100 - vacation %. "
        "Period figures are sums of days over sums of person-days, not averages of monthly percentages.",
        "- Average vacation days per employee = vacation days / employees.",
        f"- Partial months excluded from history: {partial}.",
    ])


def _explanation_text(run: EmployeeForecastRun) -> str:
    weights = feature_weights(run.contributions_by_month)
    if weights.empty:
        return "_No forecast rows._"
    table = weights.rename(columns={"feature": "Input", "signal": "Signal", "share_pct": "Share of contribution (%)"})
    return (
        "Share of total absolute XGBoost contribution per input, averaged over forecast months. These describe what "
        "the fitted model uses, not what causes vacation.\n\n"
        + df_to_markdown(table[["Input", "Signal", "Share of contribution (%)"]], float_digits=1)
    )


def _metrics_table(frame: pd.DataFrame) -> str:
    cols = ["model", "n_rows", "MAE", "RMSE", "WAPE", "Bias"]
    return df_to_markdown(frame[cols], float_digits=2)


def build_markdown(run: EmployeeForecastRun, business: BusinessView, settings: Settings) -> str:
    pag = business.pag
    start, end = run.window
    test_start, test_end = run.test_months.min(), run.test_months.max()
    rec = run.recursive_monthly
    rec_total, actual_total = rec["forecast"].sum(), rec["actual"].sum()
    model_lines = model_comparison_statements(run.pag_table(), run.employee_table())
    hs, fs = business.history_summary, business.forecast_summary
    forecast_months = business.forecast_monthly["month"]
    summary_lines = []
    if hs["months"]:
        summary_lines.append(_summary_line("Historical", hs, f"{start:%b %Y} to {end:%b %Y}"))
    if fs["months"]:
        summary_lines.append(
            _summary_line("Forecast", fs, f"{forecast_months.min():%b %Y} to {forecast_months.max():%b %Y}")
        )
    issues = "\n".join(f"- {issue}" for issue in business.issues) or "- No consistency issues found."
    sections = [
        (
            "1",
            "Business Summary",
            "\n".join(summary_lines + [f"- {line}" for line in business.forecast_lines + business.history_lines]),
        ),
        (
            "2",
            "Monthly Forecast",
            "\n\n".join([
                "Forecast vacation days come from the employee-level model (hours / "
                f"{business.hours_per_vacation_day:g}). Percentages use working days only.",
                _business_table(business.forecast_monthly),
            ]),
        ),
        (
            "3",
            "Historical Trend",
            "\n\n".join([
                f"Complete months in the analysis window ({start:%b %Y} to {end:%b %Y}).",
                _business_table(business.history_monthly),
            ]),
        ),
        (
            "4",
            "Methodology",
            _methodology(business, settings),
        ),
        (
            "5",
            "Data Checks",
            issues,
        ),
        (
            "6",
            "Technical Details",
            "\n\n".join([
                "The recursive feedback issue has been fixed. Predicted vacation months were earlier counted "
                "as vacation events in the history features, so the forecast fed on its own output. The "
                "event-count features now use observed vacation only, and predicted months cannot add events. "
                "The pre-fix diagnosis is in recursive_forecast_diagnostic.md.",
                f"Employee-level model run for {pag}: {len(run.dataset):,} employee-month rows, "
                f"{run.train_rows:,} training rows before the holdout.",
                f"Holdout: {test_start:%Y-%m} to {test_end:%Y-%m} ({len(run.test_months)} months), "
                f"{run.test_rows:,} employee-month rows. The one-step models predict each month from data dated "
                "before it. The recursive row predicts every holdout month from its own earlier predictions.",
                "PAG level (monthly totals):",
                _metrics_table(with_model_labels(run.pag_table())),
                "Employee level (employee-months):",
                _metrics_table(with_model_labels(run.employee_table())),
                "\n".join(f"- {line}" for line in model_lines),
                f"- Recursive holdout total: {rec_total:,.1f} h vs actual {actual_total:,.1f} h "
                f"({(rec_total / actual_total - 1) * 100:+.1f}%).",
                "Forecast hours by month (model output before conversion to days):",
                _future_table(run),
                "Forecast explanation:",
                _explanation_text(run),
            ]),
        ),
        (
            "7",
            "Limitations",
            "\n".join(
                [
                    "- Vacation days assume 9 h per full day. A different day length changes every day count.",
                    "- Forecast employees are members active in the lookback window. This can differ from the current "
                    "headcount, which changes the percentages.",
                    "- The holdout is 6 months. Differences between models are indicative only.",
                    "- Seasonal naive uses the same month last year and treats missing history as 0. It is kept as "
                    "the baseline.",
                    "- Event-count inputs use observed data only. Other inputs still use predicted hours for forecast "
                    "months: lags, 3- and 12-month sums, vacation occurrences over 12 months, and historical "
                    "average and maximum.",
                    "- Explanations are XGBoost attributions for the fitted model. They are not causes.",
                    "- Holiday counts come from the holiday file. Coverage gaps there affect the holiday input.",
                    "- The forecast starts after the last complete month in the data.",
                ]
            ),
        ),
    ]
    return render_report(f"Vacation Forecast: {pag}", sections)


def _future_table(run: EmployeeForecastRun) -> str:
    if run.future_monthly.empty:
        return "_No forecast rows._"
    table = run.future_monthly.copy()
    table["month"] = table["month"].dt.strftime("%Y-%m")
    return df_to_markdown(table.rename(columns={
        "month": "Month", "forecast": "Forecast hours", "employees": "Employees",
        "same_month_last_year_actual": "Same month last year (actual)",
    }), float_digits=1)


def load_clean_all(settings: Settings) -> pd.DataFrame:
    """Clean, vacation-flagged rows for every PAG in the workbook."""
    raw = load_raw_timesheets(settings)
    pre = preprocess_timesheets(raw, settings)
    return add_vacation_flag(pre.clean, settings.vacation_trc_codes)


def load_clean_pag(settings: Settings) -> pd.DataFrame:
    return select_pag(load_clean_all(settings), settings.pag)


def main(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    pag_df = load_clean_pag(settings)
    holidays = load_holidays(settings.holidays_csv)
    run = run_employee_level_forecast(pag_df, holidays, settings)
    business = build_business_view(pag_df, holidays, run, settings)
    markdown = build_markdown(run, business, settings)
    out_dir = settings.output_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_NAME
    path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    main()
