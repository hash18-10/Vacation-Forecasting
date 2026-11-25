"""End-to-end pipeline: load, clean, analyse, backtest, forecast, insights and report."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vacation_forecasting.config import Settings
from vacation_forecasting.data_loader import load_raw_timesheets
from vacation_forecasting.evaluation import (
    best_model,
    compare_models,
    compute_metrics,
    largest_errors,
    overfitting_check,
    residual_intervals,
)
from vacation_forecasting.features import build_features, build_member_month_panel, extend_panel_for_future
from vacation_forecasting.forecasting import (
    BASELINE_MODELS,
    XGBOOST_MODEL,
    aggregate_to_pag,
    baseline_one_step_backtest,
    baseline_recursive_forecast,
    fit_xgboost,
    predict_xgboost,
    xgboost_one_step_backtest,
    xgboost_recursive_forecast,
)
from vacation_forecasting.holiday_analysis import (
    applicable_holidays,
    holiday_counts_by_month,
    holiday_window_effect,
    load_holidays,
)
from vacation_forecasting.insights import build_insights
from vacation_forecasting.preprocessing import preprocess_timesheets
from vacation_forecasting.reporting import df_to_markdown, render_report, write_csv_outputs, write_report
from vacation_forecasting.vacation_analysis import (
    add_vacation_flag,
    calendar_month_profile,
    fy_vacation_summary,
    headline_metrics,
    member_vacation_profile,
    monthly_vacation_summary,
    select_pag,
    vacation_distribution,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Outputs of one pipeline run, kept in memory for tests and the Streamlit app."""

    pag: str
    backtest: pd.DataFrame
    comparison: pd.DataFrame
    best: str
    forecast: pd.DataFrame
    member_forecast: pd.DataFrame
    insights: list[str]
    report_path: Path | None = None
    csv_paths: list[Path] = field(default_factory=list)


def run_pipeline(settings: Settings, write_outputs: bool = True) -> PipelineResult:
    """Run the full analysis for the configured PAG."""
    raw = load_raw_timesheets(settings)
    pre = preprocess_timesheets(raw, settings)
    flagged = add_vacation_flag(pre.clean, settings.vacation_trc_codes)
    pag_df = select_pag(flagged, settings.pag)

    monthly = monthly_vacation_summary(pag_df)
    series = monthly.set_index("month")["vacation_hours"]
    needed = settings.min_history_months + settings.test_months
    if len(series) < needed:
        raise ValueError(
            f"Need at least {needed} months of history for {settings.pag}; found {len(series)}"
        )

    horizon = settings.forecast_horizon
    test_months = series.index[-settings.test_months:]
    future_months = pd.date_range(series.index.max() + pd.DateOffset(months=1), periods=horizon, freq="MS")
    holidays = load_holidays(settings.holidays_csv)
    locations = sorted(pag_df["location"].dropna().unique().tolist())
    holiday_counts = holiday_counts_by_month(holidays, series.index.append(future_months), locations)

    # One-step backtests on the holdout.
    panel = build_member_month_panel(pag_df)
    baseline_bt = baseline_one_step_backtest(series, test_months)
    xgb_member_bt = xgboost_one_step_backtest(panel, test_months, settings.model, holiday_counts,
                                              settings.fy_start_month)
    xgb_bt = (
        xgb_member_bt.groupby("month")["forecast"].sum().reset_index().assign(model=XGBOOST_MODEL)
    )
    backtest = pd.concat([baseline_bt, xgb_bt[["month", "model", "forecast"]]], ignore_index=True)
    backtest["actual"] = backtest["month"].map(series)
    comparison = compare_models(backtest)
    winner = best_model(comparison)

    # In-sample check for XGBoost (training months only) against its holdout error.
    features = build_features(panel, holiday_counts, settings.fy_start_month)
    train_rows = features.loc[features["month"] < test_months.min()]
    train_model = fit_xgboost(train_rows, settings.model)
    train_pred = pd.DataFrame({"month": train_rows["month"].to_numpy(),
                               "forecast": predict_xgboost(train_model, train_rows)})
    train_agg = train_pred.groupby("month")["forecast"].sum()
    train_metrics = compute_metrics(series.loc[train_agg.index], train_agg)
    test_metrics = comparison.loc[comparison["model"] == XGBOOST_MODEL, ["MAE", "RMSE", "WAPE", "Bias"]]
    overfit = overfitting_check(train_metrics, test_metrics.iloc[0].to_dict())

    # Recursive forecasts beyond the last observed month.
    future_panel = extend_panel_for_future(panel, horizon, settings.active_lookback_months)
    baseline_fc = baseline_recursive_forecast(series, future_months)
    xgb_member_fc = xgboost_recursive_forecast(future_panel, settings.model, holiday_counts,
                                               settings.fy_start_month)
    xgb_fc = aggregate_to_pag(xgb_member_fc).reset_index().assign(model=XGBOOST_MODEL)
    all_forecasts = pd.concat([baseline_fc, xgb_fc[["month", "model", "forecast"]]], ignore_index=True)

    best_fc = all_forecasts.loc[all_forecasts["model"] == winner, ["month", "forecast"]].reset_index(drop=True)
    best_bt = backtest.loc[backtest["model"] == winner]
    forecast = residual_intervals(best_bt, best_fc, level=0.8)

    daily = (
        pag_df.loc[pag_df["is_vacation"]].groupby(pag_df["transaction_date"].dt.normalize())["hours"].sum()
    )
    daily = daily.reindex(pd.date_range(pag_df["transaction_date"].min().normalize(),
                                        pag_df["transaction_date"].max().normalize(), freq="D"),
                          fill_value=0.0)
    holiday_effect = holiday_window_effect(
        daily, applicable_holidays(holidays, locations), settings.holiday_window_days
    )

    insights = build_insights(
        monthly, calendar_month_profile(pag_df), fy_vacation_summary(pag_df), comparison, winner,
        holiday_effect, float(best_fc["forecast"].sum()),
    )

    result = PipelineResult(
        pag=settings.pag, backtest=backtest, comparison=comparison, best=winner,
        forecast=forecast, member_forecast=xgb_member_fc, insights=insights,
    )
    if write_outputs:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        pre.clean.to_csv(settings.processed_dir / "clean_timesheets.csv", index=False)
        pre.rejected.to_csv(settings.processed_dir / "rejected_rows.csv", index=False)
        result.csv_paths = write_csv_outputs(settings.output_dir, settings.pag, backtest, forecast, xgb_member_fc)
        markdown = _build_markdown(settings, pre, pag_df, monthly, comparison, winner, forecast,
                                   backtest, holiday_effect, overfit, insights, horizon)
        result.report_path = write_report(settings.output_dir, settings.pag, markdown)
        logger.info("Wrote outputs to %s", settings.output_dir)
    return result


def _build_markdown(settings, pre, pag_df, monthly, comparison, winner, forecast, backtest,
                    holiday_effect, overfit, insights, horizon) -> str:
    """Build sections A to N of the report from computed tables."""
    metrics = headline_metrics(pag_df)
    rejected_counts = (
        pre.rejected["rejection_reason"].value_counts().reset_index()
        if not pre.rejected.empty else pd.DataFrame({"rejection_reason": [], "count": []})
    )
    recent = monthly.tail(12)[["month", "vacation_hours", "mom_change_pct", "yoy_change_pct"]]
    fy = fy_vacation_summary(pag_df)
    calendar = calendar_month_profile(pag_df)
    location = vacation_distribution(pag_df)
    members = member_vacation_profile(pag_df).head(10)
    worst = largest_errors(backtest, 5)
    sections = [
        ("A", "Scope and data sources",
         f"PAG analysed: **{settings.pag}**. Source folder: `{settings.raw_dir.name}`. "
         f"Forecast horizon: {horizon} months. Revenue forecasting is out of scope."),
        ("B", "Data quality",
         f"Rows loaded: {len(pre.clean) + len(pre.rejected):,}. Clean: {len(pre.clean):,}. "
         f"Rejected: {len(pre.rejected):,}.\n\n" + df_to_markdown(rejected_counts, 0)),
        ("C", "Vacation definition",
         f"Vacation is defined by TRC code only: {', '.join(sorted(settings.vacation_trc_codes))}. "
         "Descriptions are not used. The monthly target is vacation hours per member-month."),
        ("D", "Headline metrics", df_to_markdown(pd.Series(metrics).rename("value").reset_index()
                                                 .rename(columns={"index": "metric"}), 2)),
        ("E", "Monthly vacation trend (last 12 months)", df_to_markdown(recent)),
        ("F", "Financial year summary (Oct to Sep)", df_to_markdown(fy)),
        ("G", "Seasonality by calendar month", df_to_markdown(calendar)),
        ("H", "Vacation by location", df_to_markdown(location)),
        ("I", "Top members by vacation hours", df_to_markdown(members)),
        ("J", "Holiday window analysis (descriptive)",
         df_to_markdown(holiday_effect) if not holiday_effect.empty else "No holiday calendar rows found."),
        ("K", "Model comparison on holdout (one-step ahead)", df_to_markdown(comparison, 2)),
        ("L", f"Forecast ({winner}, selected by holdout MAE)", df_to_markdown(forecast, 1)),
        ("M", "Largest holdout errors and overfitting check",
         df_to_markdown(worst, 1) + "\n\n" + "\n".join(f"- {k}: {v}" for k, v in overfit.items())),
        ("N", "Insights and limitations",
         "\n".join(f"- {line}" for line in insights)
         + "\n\n- Results are associations in the historical data, not causal effects.\n"
           "- Baselines and XGBoost are compared on the same one-step holdout.\n"
           "- Intervals come from holdout residuals and are indicative only."),
    ]
    return render_report(f"Vacation Forecasting Report: {settings.pag}", sections)
