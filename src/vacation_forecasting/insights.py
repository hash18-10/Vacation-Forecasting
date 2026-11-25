"""Data-driven insight statements. Every sentence is computed from the inputs; no causal claims are made."""

from __future__ import annotations

import pandas as pd

from vacation_forecasting.employee_forecasting import EMPLOYEE_XGBOOST, MODEL_LABELS, SEASONAL_NAIVE

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def build_insights(
    monthly: pd.DataFrame,
    calendar_profile: pd.DataFrame,
    fy_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    best: str,
    holiday_effect: pd.DataFrame,
    forecast_total: float | None,
) -> list[str]:
    """Return a list of observational statements derived from the computed tables."""
    lines: list[str] = []

    if not calendar_profile.empty:
        peak = calendar_profile.loc[calendar_profile["avg_vacation_hours"].idxmax()]
        low = calendar_profile.loc[calendar_profile["avg_vacation_hours"].idxmin()]
        lines.append(
            f"On average, vacation hours were highest in {MONTH_NAMES[int(peak['calendar_month']) - 1]} "
            f"({peak['avg_vacation_hours']:,.1f} h) and lowest in {MONTH_NAMES[int(low['calendar_month']) - 1]} "
            f"({low['avg_vacation_hours']:,.1f} h) across the observed years."
        )

    if not fy_summary.empty:
        last = fy_summary.iloc[-1]
        lines.append(
            f"{last['financial_year_label']} vacation hours were {last['vacation_hours']:,.1f} h, "
            f"or {last['vacation_share_pct']:.1f}% of recorded hours."
        )
        if len(fy_summary) > 1:
            prev = fy_summary.iloc[-2]
            if prev["vacation_hours"]:
                change = (last["vacation_hours"] / prev["vacation_hours"] - 1) * 100
                direction = "up" if change >= 0 else "down"
                lines.append(
                    f"Compared with {prev['financial_year_label']}, vacation hours are {direction} "
                    f"{abs(change):.1f}% in {last['financial_year_label']}."
                )

    latest_yoy = monthly["yoy_change_pct"].dropna() if "yoy_change_pct" in monthly else pd.Series(dtype=float)
    if not latest_yoy.empty:
        lines.append(f"The most recent year-on-year change in a month was {latest_yoy.iloc[-1]:+.1f}%.")

    if not comparison.empty:
        best_row = comparison.loc[comparison["model"] == best].iloc[0]
        lines.append(
            f"On the holdout, {best} had the lowest MAE ({best_row['MAE']:,.1f} h per month) "
            f"and a WAPE of {best_row['WAPE']:.1f}%."
        )
        bias_dir = "over-forecast" if best_row["Bias"] > 0 else "under-forecast"
        lines.append(
            f"The selected model {bias_dir}s on average by {abs(best_row['Bias']):,.1f} h per month on the holdout."
        )

    if not holiday_effect.empty and holiday_effect["difference_hours"].notna().any():
        top = holiday_effect.loc[holiday_effect["difference_hours"].idxmax()]
        lines.append(
            f"The holiday window with the highest average daily vacation was {top['holiday_name']} "
            f"({top['date']}), with {top['difference_hours']:+,.2f} h per day against other days. "
            "This is a timing association only and does not show cause."
        )

    if forecast_total is not None:
        lines.append(f"The selected model projects {forecast_total:,.1f} vacation hours over the forecast horizon.")

    return lines


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def history_statements(history: pd.DataFrame, calendar_profile: pd.DataFrame) -> list[str]:
    """Highest and lowest months in the window, and calendar months above or below the average level."""
    if history.empty or history["vacation_hours"].sum() == 0:
        return []
    peak = history.loc[history["vacation_hours"].idxmax()]
    low = history.loc[history["vacation_hours"].idxmin()]
    lines = [
        f"Highest vacation month in the window: {peak['month']:%b %Y} ({peak['vacation_hours']:,.1f} h).",
        f"Lowest vacation month in the window: {low['month']:%b %Y} ({low['vacation_hours']:,.1f} h).",
    ]
    if not calendar_profile.empty:
        avg = calendar_profile.set_index("calendar_month")["avg_vacation_hours"]
        above = [MONTH_NAMES[int(m) - 1][:3] for m in avg.index[avg > avg.mean()]]
        below = [MONTH_NAMES[int(m) - 1][:3] for m in avg.index[avg < avg.mean()]]
        lines.append(f"Calendar months above the average monthly level: {', '.join(above) or 'none'}.")
        lines.append(f"Calendar months below the average monthly level: {', '.join(below) or 'none'}.")
    return lines


def year_over_year_statement(history: pd.DataFrame) -> str | None:
    """Last 12 months against the 12 months before them. Needs 24 months of history."""
    ordered = history.sort_values("month").reset_index(drop=True)
    if len(ordered) < 24:
        return None
    last = ordered.iloc[-12:]
    prior = ordered.iloc[-24:-12]
    prior_total = prior["vacation_hours"].sum()
    if prior_total == 0:
        return None
    last_total = last["vacation_hours"].sum()
    change = (last_total / prior_total - 1) * 100
    return (
        f"Last 12 months ({last['month'].iloc[0]:%b %Y} to {last['month'].iloc[-1]:%b %Y}): {last_total:,.1f} h, "
        f"{change:+.1f}% vs the previous 12 months ({prior['month'].iloc[0]:%b %Y} to {prior['month'].iloc[-1]:%b %Y})."
    )


def fiscal_year_statement(fy_summary: pd.DataFrame) -> str | None:
    """Change between the last two complete financial years."""
    complete = fy_summary.loc[fy_summary["months_observed"] == 12]
    if len(complete) < 2:
        return None
    prev, last = complete.iloc[-2], complete.iloc[-1]
    if prev["vacation_hours"] == 0:
        return None
    change = (last["vacation_hours"] / prev["vacation_hours"] - 1) * 100
    return (
        f"{last['financial_year_label']} vacation hours are {change:+.1f}% vs "
        f"{prev['financial_year_label']} (both complete years)."
    )


def _gap_sentence(diff: float, baseline: str, unit: str) -> str:
    """States whether the forecast is higher, lower or the same as the baseline, with the size of the gap."""
    if round(diff, 1) == 0:
        return f"The forecast is the same as {baseline}."
    word = "higher" if diff > 0 else "lower"
    unit_text = " percentage points" if unit == "pp" else unit
    return f"The forecast is {abs(diff):.1f}{unit_text} {word} than {baseline}."


def forecast_statements(forecast: pd.DataFrame) -> list[str]:
    """Forecast total, peak and low months, half-horizon trend, and comparison with the same months last year."""
    if forecast.empty:
        return []
    peak = forecast.loc[forecast["forecast"].idxmax()]
    low = forecast.loc[forecast["forecast"].idxmin()]
    total = forecast["forecast"].sum()
    lines = [
        f"Forecast total {forecast['month'].min():%b %Y} to {forecast['month'].max():%b %Y}: {total:,.1f} h.",
        f"Highest forecast month: {peak['month']:%b %Y} ({peak['forecast']:,.1f} h). "
        f"Lowest: {low['month']:%b %Y} ({low['forecast']:,.1f} h).",
    ]
    half = len(forecast) // 2
    if half >= 1:
        first = forecast["forecast"].iloc[:half].mean()
        second = forecast["forecast"].iloc[-half:].mean()
        direction = "rises" if second > first else "falls" if second < first else "is flat"
        lines.append(
            f"Trend across the horizon: the monthly average {direction} from {first:,.1f} h "
            f"to {second:,.1f} h in the second half."
        )
    last_year = forecast["same_month_last_year_actual"]
    if last_year.notna().all() and last_year.sum() > 0:
        change = (total / last_year.sum() - 1) * 100
        lines.append(
            f"Same months last year (actual): {last_year.sum():,.1f} h. {_gap_sentence(change, 'that period', '%')}"
        )
    return lines


def model_comparison_statements(pag: pd.DataFrame, employee: pd.DataFrame) -> list[str]:
    """Lowest holdout MAE at each level, and how seasonal naive compares with employee-month XGBoost."""
    best_pag = pag.loc[pag["MAE"].idxmin()]
    best_emp = employee.loc[employee["MAE"].idxmin()]
    lines = [
        f"Lowest PAG-level holdout MAE: {_label(best_pag['model'])} ({best_pag['MAE']:,.1f} h per month).",
        f"Lowest employee-level holdout MAE: {_label(best_emp['model'])} "
        f"({best_emp['MAE']:,.2f} h per employee-month).",
    ]
    sn = employee.loc[employee["model"] == SEASONAL_NAIVE, "MAE"]
    xgb = employee.loc[employee["model"] == EMPLOYEE_XGBOOST, "MAE"]
    if not sn.empty and not xgb.empty:
        s, x = float(sn.iloc[0]), float(xgb.iloc[0])
        if s < x:
            lines.append(f"At employee level, seasonal naive has lower MAE than employee-month XGBoost ({s:.2f} vs {x:.2f} h).")
        else:
            lines.append(f"At employee level, employee-month XGBoost has lower MAE than seasonal naive ({x:.2f} vs {s:.2f} h).")
    months = int(pag["n_rows"].iloc[0])
    lines.append(f"The holdout covers {months} months, so these differences are indicative only.")
    return lines


def _valid_pct(*values: float | None) -> bool:
    return all(v is not None and not pd.isna(v) for v in values)


def historical_trend_lines(
    history: pd.DataFrame,
    history_summary: dict,
    previous_period: dict | None,
    prior_period: dict | None,
) -> list[str]:
    """Historical facts: overall level, calendar pattern, and change between the last two 12-month periods."""
    lines: list[str] = []
    if history_summary["months"] > 0 and _valid_pct(history_summary["vacation_pct"]):
        lines.append(
            f"Historical vacation % over {history_summary['months']} complete months: "
            f"{history_summary['vacation_pct']:.1f}% (availability {history_summary['availability_pct']:.1f}%)."
        )
    if previous_period and _valid_pct(previous_period["vacation_pct"]):
        lines.append(
            f"Previous 12 months: vacation {previous_period['vacation_pct']:.1f}%, "
            f"average {previous_period['avg_vacation_days_per_employee']:.1f} vacation days per employee."
        )
    if previous_period and prior_period and _valid_pct(previous_period["vacation_pct"], prior_period["vacation_pct"]):
        diff = previous_period["vacation_pct"] - prior_period["vacation_pct"]
        direction = "increased" if diff > 0 else "decreased" if diff < 0 else "did not change"
        lines.append(f"Vacation % {direction} compared with the 12 months before that ({diff:+.1f} percentage points).")
    valid = history.dropna(subset=["vacation_pct"])
    if not valid.empty:
        by_calendar_month = valid.groupby(valid["month"].dt.month)["vacation_pct"].mean()
        peak_month, low_month = by_calendar_month.idxmax(), by_calendar_month.idxmin()
        lines.append(
            f"{MONTH_NAMES[peak_month - 1]} has the highest average vacation % across the history "
            f"({by_calendar_month.loc[peak_month]:.1f}%). {MONTH_NAMES[low_month - 1]} has the lowest "
            f"({by_calendar_month.loc[low_month]:.1f}%)."
        )
    return lines


def forecast_insight_lines(
    forecast: pd.DataFrame,
    forecast_summary: dict,
    previous_period: dict | None,
    last_year_summary: dict | None,
    latest_employees: int | None,
) -> list[str]:
    """Forecast statements in business language. Every number is calculated from the forecast table."""
    if forecast_summary["months"] == 0 or not _valid_pct(forecast_summary["vacation_pct"]):
        return []
    lines = [
        f"The forecasted vacation level is {forecast_summary['vacation_pct']:.1f}%.",
        f"Expected availability is {forecast_summary['availability_pct']:.1f}%.",
        f"Employees are expected to take an average of {forecast_summary['avg_vacation_days_per_employee']:.1f} "
        f"vacation days each over the forecast period ({forecast_summary['vacation_days']:,.0f} days in total).",
    ]
    valid = forecast.dropna(subset=["vacation_pct"])
    if len(valid) > 1:
        peak = valid.loc[valid["vacation_pct"].idxmax()]
        lines.append(
            f"{peak['month']:%B %Y} is expected to have the highest vacation activity "
            f"({peak['vacation_pct']:.1f}%)."
        )
    if last_year_summary and _valid_pct(last_year_summary["vacation_pct"]):
        diff = forecast_summary["vacation_pct"] - last_year_summary["vacation_pct"]
        lines.append(
            f"The same months last year were {last_year_summary['vacation_pct']:.1f}%. "
            + _gap_sentence(diff, "the same months last year", "pp")
        )
    if previous_period and _valid_pct(previous_period["vacation_pct"]):
        diff = forecast_summary["vacation_pct"] - previous_period["vacation_pct"]
        lines.append(
            f"The previous 12 months were {previous_period['vacation_pct']:.1f}%. "
            + _gap_sentence(diff, "the previous 12 months", "pp")
        )
    if latest_employees and forecast_summary["employees"] > 0:
        if abs(forecast_summary["employees"] / latest_employees - 1) > 0.10:
            lines.append(
                f"Forecast employees ({forecast_summary['employees']:,}) differ from the latest complete month's "
                f"headcount ({latest_employees:,}). Check this before relying on the percentages."
            )
    lines.append("The forecast is based on historical vacation patterns for this PAG.")
    return lines
