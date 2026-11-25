"""Origin-based validation of the forecast models on one shared population and holdout.

At each origin month, every model sees data up to that month only. The population is the members with a
timesheet row in the origin month, the same rule the production forecast uses. Recursive models feed their
own predictions back in. One-step models use actual values inside the holdout, so they are reference only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vacation_forecasting.config import Settings, get_settings
from vacation_forecasting.demo import build_demo
from vacation_forecasting.employee_features import (
    KEY,
    add_employee_features,
    analysis_window,
    build_employee_month_frame,
    extend_for_future,
)
from vacation_forecasting.employee_forecasting import (
    EMPLOYEE_XGBOOST,
    employee_backtest,
    fit_employee_model,
    recursive_employee_forecast,
)
from vacation_forecasting.employee_report import load_clean_pag
from vacation_forecasting.evaluation import compute_metrics
from vacation_forecasting.features import build_member_month_panel, extend_panel_for_future
from vacation_forecasting.forecasting import xgboost_one_step_backtest, xgboost_recursive_forecast
from vacation_forecasting.holiday_analysis import holiday_counts_by_month, load_holidays
from vacation_forecasting.reporting import df_to_markdown, render_report

SEASONAL = "seasonal_naive"
AGG_ONE = "aggregate_one_step"
EMP_ONE = "employee_one_step"
AGG_REC = "aggregate_recursive"
EMP_REC = "employee_recursive"
MODEL_ORDER = (SEASONAL, AGG_ONE, EMP_ONE, AGG_REC, EMP_REC)
MODEL_LABELS: dict[str, str] = {
    SEASONAL: "Seasonal naive (same month last year)",
    AGG_ONE: "Aggregate XGBoost, one-step (reference only)",
    EMP_ONE: "Employee XGBoost, one-step (reference only)",
    AGG_REC: "Aggregate XGBoost, recursive",
    EMP_REC: "Employee XGBoost, recursive",
}
REPORT_NAME = "model_validation.md"


@dataclass
class OriginResult:
    """Holdout predictions for one origin. Vacation values are hours; `pag` is summed over the population."""

    origin: pd.Timestamp
    holdout: pd.DatetimeIndex
    population: int
    employee_months: pd.DataFrame
    pag: pd.DataFrame
    history_monthly_avg_hours: float


def origin_blocks(months: pd.DatetimeIndex, horizon: int, n_blocks: int) -> list[pd.DatetimeIndex]:
    """Holdout blocks of `horizon` months, latest first, without overlap."""
    ordered = pd.DatetimeIndex(sorted(months))
    blocks = []
    for k in range(n_blocks):
        stop = len(ordered) - horizon * k
        start = stop - horizon
        if start < 1:
            raise ValueError("Not enough months for the requested holdout blocks")
        blocks.append(ordered[start:stop])
    return blocks


def _vacation_hours(rows: pd.DataFrame) -> pd.Series:
    return rows.loc[rows["is_vacation"]].groupby(["member_id", "month"])["hours"].sum()


def _hours_per_occurrence(rows: pd.DataFrame) -> float:
    vac = rows.loc[rows["is_vacation"], "hours"]
    return float(vac.sum() / max(len(vac), 1))


def _align(frame: pd.DataFrame, grid: pd.MultiIndex) -> np.ndarray:
    return frame.set_index(KEY)["forecast"].reindex(grid).fillna(0.0).to_numpy()


def backtest_origin(
    rows: pd.DataFrame,
    holidays: pd.DataFrame,
    settings: Settings,
    holdout: pd.DatetimeIndex,
    start: pd.Timestamp,
) -> OriginResult:
    """Score every model on the same population and holdout. `rows` are clean vacation-flagged rows in the window."""
    fy, cfg = settings.fy_start_month, settings.model
    origin = holdout.min() - pd.DateOffset(months=1)
    hist = rows.loc[rows["month"] <= origin]
    upto = rows.loc[rows["month"] <= holdout.max()]
    population = sorted(hist.loc[hist["month"] == origin, "member_id"].unique())
    if not population:
        raise ValueError(f"No members with a timesheet row in origin month {origin:%Y-%m}")

    grid = pd.MultiIndex.from_product([population, holdout], names=KEY)
    vac_all = _vacation_hours(rows)
    out = pd.DataFrame(index=grid)
    out["actual"] = vac_all.reindex(grid, fill_value=0.0).to_numpy()
    last_year = pd.MultiIndex.from_arrays(
        [grid.get_level_values(0), grid.get_level_values(1) - pd.DateOffset(months=12)], names=KEY
    )
    out[SEASONAL] = vac_all.reindex(last_year, fill_value=0.0).to_numpy()

    hours_per_occurrence = _hours_per_occurrence(hist)
    origin_frame = build_employee_month_frame(hist, holidays, start, origin, fy)
    employee_model = fit_employee_model(add_employee_features(origin_frame), cfg)
    future = extend_for_future(origin_frame, holidays, len(holdout), 1, fy)
    emp_rec, _ = recursive_employee_forecast(future, employee_model, hours_per_occurrence)
    out[EMP_REC] = _align(emp_rec, grid)

    full_features = add_employee_features(build_employee_month_frame(upto, holidays, start, holdout.max(), fy))
    emp_one = employee_backtest(full_features, holdout, cfg)
    out[EMP_ONE] = _align(emp_one.rename(columns={EMPLOYEE_XGBOOST: "forecast"}), grid)

    months_hol = pd.date_range(start, holdout.max(), freq="MS")
    hist_locations = sorted(hist["location"].dropna().unique().tolist())
    hol_counts = holiday_counts_by_month(holidays, months_hol, hist_locations)
    panel_hist = build_member_month_panel(hist)
    agg_rec = xgboost_recursive_forecast(extend_panel_for_future(panel_hist, len(holdout), 1), cfg, hol_counts, fy)
    out[AGG_REC] = _align(agg_rec, grid)

    upto_locations = sorted(upto["location"].dropna().unique().tolist())
    hol_full = holiday_counts_by_month(holidays, months_hol, upto_locations)
    agg_one = xgboost_one_step_backtest(build_member_month_panel(upto), holdout, cfg, hol_full, fy)
    out[AGG_ONE] = _align(agg_one, grid)

    employee_months = out.reset_index()
    pag = employee_months.groupby("month")[list(MODEL_ORDER) + ["actual"]].sum().reindex(holdout).fillna(0.0)
    present = (
        rows.loc[rows["month"].isin(holdout) & rows["member_id"].isin(population)]
        .groupby("month")["member_id"].nunique()
    )
    pag["population_present"] = present.reindex(holdout, fill_value=0).to_numpy()
    pag.index.name = "month"

    n_hist_months = len(pd.date_range(start, origin, freq="MS"))
    pop_hist_hours = hist.loc[hist["is_vacation"] & hist["member_id"].isin(population), "hours"].sum()
    return OriginResult(
        origin=origin,
        holdout=holdout,
        population=len(population),
        employee_months=employee_months,
        pag=pag,
        history_monthly_avg_hours=float(pop_hist_hours / n_hist_months),
    )


def run_validation(rows: pd.DataFrame, holidays: pd.DataFrame, settings: Settings, n_blocks: int = 2) -> list[OriginResult]:
    start, end = analysis_window(rows["transaction_date"])
    window = rows.loc[(rows["month"] >= start) & (rows["month"] <= end)].copy()
    blocks = origin_blocks(pd.date_range(start, end, freq="MS"), settings.test_months, n_blocks)
    return [backtest_origin(window, holidays, settings, block, start) for block in blocks]


def score(employee_months: pd.DataFrame, pag: pd.DataFrame, level: str, settings: Settings) -> pd.DataFrame:
    """MAE, RMSE, WAPE and Bias per model, in vacation days. `level` is 'pag' (monthly totals) or 'employee'."""
    hpd = settings.hours_per_vacation_day
    frame = pag.reset_index() if level == "pag" else employee_months
    table = []
    for model in MODEL_ORDER:
        metrics = compute_metrics(frame["actual"] / hpd, frame[model] / hpd)
        table.append({"key": model, "model": MODEL_LABELS[model], "n_rows": len(frame), **metrics})
    table = pd.DataFrame(table)
    base = table.loc[table["key"] == SEASONAL].iloc[0]
    table["MAE vs SN %"] = (table["MAE"] / base["MAE"] - 1) * 100
    table["WAPE vs SN pts"] = table["WAPE"] - base["WAPE"]
    return table


def beats_seasonal(table: pd.DataFrame, model: str) -> bool:
    base = table.loc[table["key"] == SEASONAL].iloc[0]
    row = table.loc[table["key"] == model].iloc[0]
    return bool(row["MAE"] < base["MAE"] and row["WAPE"] < base["WAPE"])


def recommend(pooled: pd.DataFrame, per_origin: list[pd.DataFrame]) -> tuple[str, str]:
    """Pick a model only if it beats seasonal naive on MAE and WAPE, pooled and in every origin."""
    qualifying = [
        m for m in (AGG_REC, EMP_REC)
        if beats_seasonal(pooled, m) and all(beats_seasonal(t, m) for t in per_origin)
    ]
    if not qualifying:
        return SEASONAL, (
            "No recursive XGBoost model beats seasonal naive on both MAE and WAPE in the pooled holdout and in "
            "every origin. Use seasonal naive for the application forecast."
        )
    best = min(qualifying, key=lambda m: float(pooled.loc[pooled["key"] == m, "MAE"].iloc[0]))
    return best, (
        f"{MODEL_LABELS[best]} beats seasonal naive on MAE and WAPE in the pooled holdout and in every origin, "
        "and has the lowest pooled MAE of the qualifying models."
    )


def inflation_check(result: OriginResult, settings: Settings) -> pd.DataFrame:
    """Holdout totals in vacation days, compared with the actuals, seasonal naive and the historical monthly average."""
    hpd = settings.hours_per_vacation_day
    months = len(result.holdout)
    frame = result.pag.sum()
    totals = {
        "Actual": frame["actual"],
        "Seasonal naive": frame[SEASONAL],
        "Historical monthly average": result.history_monthly_avg_hours * months,
        "Aggregate XGBoost, recursive": frame[AGG_REC],
        "Employee XGBoost, recursive": frame[EMP_REC],
    }
    actual_days = frame["actual"] / hpd
    sn_days = frame[SEASONAL] / hpd
    avg_days = result.history_monthly_avg_hours * months / hpd
    rows = []
    for name, hours in totals.items():
        days = hours / hpd
        rows.append({
            "Series": name,
            "Holdout total (days)": days,
            "vs actual": days / actual_days if actual_days else np.nan,
            "vs seasonal naive": days / sn_days if sn_days else np.nan,
            "vs historical average": days / avg_days if avg_days else np.nan,
        })
    return pd.DataFrame(rows)


def end_to_end_check(future_employees: pd.DataFrame, forecast_monthly: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, float]:
    """Recompute forecast hours -> days -> available person-days -> percentages and compare with the app numbers."""
    hpd = settings.hours_per_vacation_day
    hours = future_employees.groupby("month")["forecast"].sum()
    employees = future_employees.groupby("month")["member_id"].nunique()
    app = forecast_monthly.set_index("month")
    rows = []
    for month in hours.index:
        weekdays = int(np.busday_count(month.strftime("%Y-%m-%d"), (month + pd.DateOffset(months=1)).strftime("%Y-%m-%d")))
        days = float(hours[month]) / hpd
        available = int(employees[month]) * weekdays
        vac_pct = days / available * 100 if available else np.nan
        avail_pct = 100 - vac_pct
        rows.append({
            "Month": month.strftime("%b %Y"),
            "Employees": int(employees[month]),
            "Working days": weekdays,
            "Available person-days": available,
            "Forecast vacation days": days,
            "Vacation %": vac_pct,
            "Availability %": avail_pct,
            "Difference vs app (pts)": max(
                abs(vac_pct - app.loc[month, "vacation_pct"]),
                abs(avail_pct - app.loc[month, "availability_pct"]),
                abs(days - app.loc[month, "vacation_days"]),
            ),
        })
    table = pd.DataFrame(rows)
    return table, float(table["Difference vs app (pts)"].max())


def build_markdown(
    results: list[OriginResult],
    settings: Settings,
    e2e: pd.DataFrame,
    e2e_max_diff: float,
    pooled_pag: pd.DataFrame,
    pooled_emp: pd.DataFrame,
    per_origin_pag: list[pd.DataFrame],
    choice: str,
    choice_text: str,
) -> str:
    fmt = lambda t: df_to_markdown(t, float_digits=2)  # noqa: E731
    cols = ["model", "n_rows", "MAE", "RMSE", "WAPE", "Bias", "MAE vs SN %", "WAPE vs SN pts"]
    origin_sections = []
    for result, table in zip(results, per_origin_pag):
        origin_sections.append(
            f"Origin {result.origin:%b %Y}, holdout {result.holdout.min():%b %Y} to {result.holdout.max():%b %Y}, "
            f"population {result.population} members.\n\n" + fmt(table[cols])
        )
    drift = "\n".join(
        f"- Origin {r.origin:%b %Y}: members in the population with a row in each holdout month: "
        + ", ".join(f"{m:%b %Y} {int(c)}" for m, c in r.pag["population_present"].items())
        + f" (of {r.population})."
        for r in results
    )
    inflation = "\n\n".join(
        f"Origin {r.origin:%b %Y}:\n\n" + df_to_markdown(inflation_check(r, settings), float_digits=2)
        for r in results
    )
    sections = [
        ("1", "Protocol", "\n".join([
            f"- Units: vacation days (hours / {settings.hours_per_vacation_day:g}) per month, PAG totals or employee-months.",
            f"- Holdout: {settings.test_months} months per origin, {len(results)} origins, each on the same rule.",
            "- Population: members with a timesheet row in the origin month. The forecast population is held "
            "fixed over the holdout.",
            "- Recursive models use only data up to the origin and feed their own predictions forward.",
            "- One-step models use actual values inside the holdout. They are reference only.",
            "- WAPE = sum of absolute errors / sum of actuals x 100. Bias = mean of (forecast - actual).",
        ])),
        ("2", "PAG Level, Pooled Over Origins", fmt(pooled_pag[cols])),
        ("3", "PAG Level, By Origin", "\n\n".join(origin_sections)),
        ("4", "Employee Level, Pooled", fmt(pooled_emp[cols])),
        ("5", "Recursive Inflation Check", inflation),
        ("6", "Headcount Drift", drift + "\n\nActual values for members who left are 0, but the forecast keeps them in the population."),
        ("7", "End-to-End Forecast Check (Production Forecast)", "\n".join([
            "Forecast hours / 9 = vacation days. Available person-days = employees x working days (Mon to Fri). "
            "Vacation % = vacation days / available person-days x 100. Availability % = 100 - vacation %.",
            "",
            df_to_markdown(e2e, float_digits=2),
            f"Maximum difference vs app numbers: {e2e_max_diff:.6f}.",
        ])),
        ("8", "Recommendation", f"Use **{MODEL_LABELS[choice]}** for the application.\n\n{choice_text}"),
        ("9", "Limitations", "\n".join([
            f"- {len(results)} origins only; the holdout is short, so differences are indicative.",
            "- One-step rows use actual values inside the holdout and are not deployable as a 6-month forecast.",
            "- The population is fixed over the horizon. Leavers stay in the forecast; new joiners are not added.",
            "- Holiday counts are not part of the backtest decision because public-holiday exclusion is off.",
        ])),
    ]
    return render_report(f"Model Validation: {settings.pag}", sections)


def main(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    rows = load_clean_pag(settings)
    holidays = load_holidays(settings.holidays_csv)
    results = run_validation(rows, holidays, settings, n_blocks=2)

    pooled_emp = score(pd.concat([r.employee_months for r in results], ignore_index=True), None, "employee", settings)
    pooled_pag_frame = pd.concat([r.pag.reset_index() for r in results], ignore_index=True)
    pooled_pag = score(None, pooled_pag_frame.set_index("month"), "pag", settings)
    per_origin_pag = [score(None, r.pag, "pag", settings) for r in results]
    choice, choice_text = recommend(pooled_pag, per_origin_pag)

    demo = build_demo(rows, holidays, settings)
    e2e, e2e_max = end_to_end_check(demo.run.future_employees, demo.business.forecast_monthly, settings)

    markdown = build_markdown(
        results, settings, e2e, e2e_max, pooled_pag, pooled_emp, per_origin_pag, choice, choice_text
    )
    out_dir = settings.output_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_NAME
    path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    main()
