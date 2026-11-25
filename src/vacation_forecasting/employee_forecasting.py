"""Employee-month forecasting: chronological holdout, comparison with baselines and a recursive forecast.

All comparisons are one-step-ahead on the same holdout months. Each test month is predicted only from
data dated before it, and the holdout is never used for training. The future forecast is recursive:
each predicted month is fed back in as history for the next one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from vacation_forecasting.config import ModelSettings, Settings
from vacation_forecasting.employee_features import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    KEY,
    TARGET,
    add_employee_features,
    analysis_window,
    build_employee_month_frame,
    extend_for_future,
)
from vacation_forecasting.evaluation import best_model, compute_metrics
from vacation_forecasting.forecasting import baseline_one_step_backtest, xgboost_one_step_backtest
from vacation_forecasting.features import build_member_month_panel
from vacation_forecasting.holiday_analysis import holiday_counts_by_month

logger = logging.getLogger(__name__)

SEASONAL_NAIVE = "seasonal_naive"
EXISTING_XGBOOST = "existing_aggregate_xgboost"
EMPLOYEE_XGBOOST = "employee_month_xgboost"
MODEL_ORDER = (SEASONAL_NAIVE, EXISTING_XGBOOST, EMPLOYEE_XGBOOST)
# Recursive forecast on the holdout, after the feedback fix. Reported next to the one-step models.
RECURSIVE = "recursive_forecast"
MODEL_LABELS: dict[str, str] = {
    SEASONAL_NAIVE: "Seasonal naive (same month last year)",
    EXISTING_XGBOOST: "Existing aggregate XGBoost (one-step)",
    EMPLOYEE_XGBOOST: "Employee-month XGBoost (one-step)",
    RECURSIVE: "Employee-month XGBoost (recursive, post-fix)",
}


@dataclass
class EmployeeForecastRun:
    """Outputs of one employee-level run, kept in memory for the report and tests."""

    window: tuple[pd.Timestamp, pd.Timestamp]
    train_months: pd.DatetimeIndex
    test_months: pd.DatetimeIndex
    dataset: pd.DataFrame
    train_rows: int
    test_rows: int
    pag_series: pd.Series
    monthly_test: pd.DataFrame
    pag_comparison: pd.DataFrame
    employee_comparison: pd.DataFrame
    predictions: pd.DataFrame
    best: str
    future_employees: pd.DataFrame = field(default_factory=pd.DataFrame)
    future_monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    xgboost_future_employees: pd.DataFrame = field(default_factory=pd.DataFrame)
    production_model: str = SEASONAL_NAIVE
    contributions_by_month: pd.DataFrame = field(default_factory=pd.DataFrame)
    recursive_pag_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    recursive_employee_comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    recursive_monthly: pd.DataFrame = field(default_factory=pd.DataFrame)

    def pag_table(self) -> pd.DataFrame:
        """PAG-level holdout metrics for every model, including the recursive forecast."""
        return pd.concat([self.pag_comparison, self.recursive_pag_comparison], ignore_index=True)

    def employee_table(self) -> pd.DataFrame:
        """Employee-level holdout metrics for every model, including the recursive forecast."""
        return pd.concat([self.employee_comparison, self.recursive_employee_comparison], ignore_index=True)


def with_model_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace internal model names with the labels used in the report and the demo."""
    return frame.assign(model=frame["model"].map(lambda name: MODEL_LABELS.get(name, name)))


def split_months(months: pd.DatetimeIndex, test_months: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Chronological split: the last `test_months` months are the holdout; everything earlier is training."""
    ordered = pd.DatetimeIndex(sorted(months))
    if len(ordered) <= test_months:
        raise ValueError(f"Need more than {test_months} months; found {len(ordered)}")
    return ordered[:-test_months], ordered[-test_months:]


def _make_regressor(settings: ModelSettings) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        max_depth=settings.max_depth,
        learning_rate=settings.learning_rate,
        n_estimators=settings.n_estimators,
        subsample=settings.subsample,
        colsample_bytree=settings.colsample_bytree,
        min_child_weight=settings.min_child_weight,
        reg_lambda=settings.reg_lambda,
        random_state=settings.random_state,
        n_jobs=1,
    )


def fit_employee_model(train: pd.DataFrame, settings: ModelSettings) -> XGBRegressor:
    """Fit on employee-months with a known target. Missing history features are handled natively by XGBoost."""
    known = train.loc[train[TARGET].notna() & ~train["is_future"]]
    if known.empty:
        raise ValueError("No training rows with known targets")
    model = _make_regressor(settings)
    model.fit(known[list(FEATURE_COLUMNS)].astype(float), known[TARGET].astype(float))
    return model


def predict_employee_model(model: XGBRegressor, frame: pd.DataFrame) -> np.ndarray:
    """Non-negative predicted vacation hours per employee-month."""
    return np.clip(model.predict(frame[list(FEATURE_COLUMNS)].astype(float)), 0.0, None)


def employee_backtest(features: pd.DataFrame, test_months: pd.DatetimeIndex, settings: ModelSettings) -> pd.DataFrame:
    """Train on months before the holdout; predict each holdout employee-month from prior data only."""
    train = features.loc[features["month"] < test_months.min()]
    test = features.loc[features["month"].isin(test_months) & ~features["is_future"]].copy()
    model = fit_employee_model(train, settings)
    test[EMPLOYEE_XGBOOST] = predict_employee_model(model, test)
    test[SEASONAL_NAIVE] = test["vacation_same_month_last_year"].fillna(0.0).clip(lower=0.0)
    return test[[*KEY, TARGET, EMPLOYEE_XGBOOST, SEASONAL_NAIVE]].rename(columns={TARGET: "actual"})


def existing_xgboost_backtest(
    window_rows: pd.DataFrame,
    holidays: pd.DataFrame,
    months_all: pd.DatetimeIndex,
    test_months: pd.DatetimeIndex,
    settings: ModelSettings,
    fy_start_month: int,
) -> pd.DataFrame:
    """The existing member-panel XGBoost from forecasting.py, run on the same window and holdout."""
    panel = build_member_month_panel(window_rows)
    locations = sorted(window_rows["location"].dropna().unique().tolist())
    holiday_counts = holiday_counts_by_month(holidays, months_all, locations)
    backtest = xgboost_one_step_backtest(panel, test_months, settings, holiday_counts, fy_start_month)
    return backtest[KEY + ["forecast"]].rename(columns={"forecast": EXISTING_XGBOOST})


def _metric_row(model: str, actual: pd.Series, forecast: pd.Series) -> dict[str, object]:
    return {"model": model, "n_rows": len(actual), **compute_metrics(actual, forecast)}


def _check_totals(predictions: pd.DataFrame, series: pd.Series, test_months: pd.DatetimeIndex) -> None:
    """Employee-level actuals must add up to the PAG monthly actuals, or the comparison is not like-for-like."""
    summed = predictions.groupby("month")["actual"].sum().reindex(test_months, fill_value=0.0)
    expected = series.reindex(test_months, fill_value=0.0)
    if not np.allclose(summed.to_numpy(), expected.to_numpy(), atol=1e-6):
        raise ValueError("Employee-level actuals do not reconcile with PAG monthly actuals")


def _contributions(model: XGBRegressor, step: pd.DataFrame) -> pd.DataFrame:
    """Per-feature XGBoost contributions (SHAP-style attribution from the booster). Model attribution, not causation."""
    dmatrix = xgb.DMatrix(step[list(FEATURE_COLUMNS)].astype(float))
    raw = model.get_booster().predict(dmatrix, pred_contribs=True)
    out = pd.DataFrame(raw[:, : len(FEATURE_COLUMNS)], columns=list(FEATURE_COLUMNS))
    out["month"] = step["month"].to_numpy()
    return out


def recursive_employee_forecast(
    future_frame: pd.DataFrame,
    model: XGBRegressor,
    hours_per_occurrence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast future months one at a time; each prediction becomes history for the next month.

    Returns (employee-level future rows, mean contributions per month).
    """
    work = future_frame.copy()
    future_months = sorted(work.loc[work["is_future"], "month"].unique())
    contribution_parts: list[pd.DataFrame] = []
    for month in future_months:
        feats = add_employee_features(work)
        step = feats.loc[feats["is_future"] & feats["month"].eq(month)]
        preds = predict_employee_model(model, step)
        contribution_parts.append(_contributions(model, step))

        pred_series = pd.Series(preds, index=pd.MultiIndex.from_frame(step[KEY]))
        mask = work["is_future"] & work["month"].eq(month)
        values = pred_series.reindex(pd.MultiIndex.from_frame(work.loc[mask, KEY])).to_numpy()
        work.loc[mask, TARGET] = values
        work.loc[mask, "vacation_occurrences"] = values / hours_per_occurrence

    final = add_employee_features(work)
    employees = final.loc[
        final["is_future"],
        KEY + ["location", TARGET, "vacation_same_month_last_year", "vacation_prev12_sum"],
    ]
    employees = employees.rename(columns={TARGET: "forecast"}).reset_index(drop=True)
    contributions = pd.concat(contribution_parts, ignore_index=True)
    contributions_by_month = contributions.groupby("month")[list(FEATURE_COLUMNS)].sum()
    return employees, contributions_by_month


def recursive_holdout_forecast(
    features: pd.DataFrame,
    test_months: pd.DatetimeIndex,
    settings: ModelSettings,
    hours_per_occurrence: float,
) -> pd.DataFrame:
    """Forecast the holdout months recursively, so history after the split comes only from predictions."""
    model = fit_employee_model(features.loc[features["month"] < test_months.min()], settings)
    hold = features.copy()
    in_test = hold["month"].isin(test_months)
    hold.loc[in_test, [TARGET, "vacation_occurrences", "actual_event"]] = np.nan
    hold.loc[in_test, "is_future"] = True
    employees, _ = recursive_employee_forecast(hold, model, hours_per_occurrence)
    return employees


MAX_SEASONAL_HORIZON = 12


def seasonal_naive_future(future_frame: pd.DataFrame) -> pd.DataFrame:
    """Production forecast: each member's vacation hours in the same month last year (0 if no row).

    Same rule as the validation harness. Only actual history is read, because every forecast month is at most
    12 months after the last actual month. Returns the same columns as `recursive_employee_forecast`.
    """
    horizon_months = int(future_frame.loc[future_frame["is_future"], "month"].nunique())
    if horizon_months > MAX_SEASONAL_HORIZON:
        raise ValueError(f"Seasonal naive supports at most {MAX_SEASONAL_HORIZON} future months, got {horizon_months}")
    feats = add_employee_features(future_frame)
    employees = feats.loc[
        feats["is_future"],
        KEY + ["location", "vacation_same_month_last_year", "vacation_prev12_sum"],
    ].copy()
    employees["forecast"] = employees["vacation_same_month_last_year"].fillna(0.0).clip(lower=0.0)
    return employees[KEY + ["location", "forecast", "vacation_same_month_last_year", "vacation_prev12_sum"]].reset_index(
        drop=True
    )


def _hours_per_occurrence(rows: pd.DataFrame) -> float:
    """Average hours per vacation row, used to convert predicted hours into occurrences."""
    return float(rows.loc[rows["is_vacation"], "hours"].sum() / max(int(rows["is_vacation"].sum()), 1))


def feature_weights(contributions_by_month: pd.DataFrame) -> pd.DataFrame:
    """Share of total absolute XGBoost contribution per input, averaged over forecast months. Attribution only."""
    columns = ["feature", "signal", "share_pct"]
    if contributions_by_month.empty:
        return pd.DataFrame(columns=columns)
    mean_abs = contributions_by_month.abs().mean()
    share = mean_abs / mean_abs.sum() * 100
    out = pd.DataFrame(
        {
            "feature": share.index,
            "signal": [FEATURE_LABELS[name] for name in share.index],
            "share_pct": share.to_numpy(),
        }
    )
    return out.sort_values("share_pct", ascending=False).reset_index(drop=True)


def run_employee_level_forecast(rows: pd.DataFrame, holidays: pd.DataFrame, settings: Settings) -> EmployeeForecastRun:
    """Full employee-level run: dataset, holdout comparison, then a future forecast trained on all complete months.

    `rows` is the clean, vacation-flagged data for one PAG (needs member_id, pag, location, month,
    transaction_date, hours, is_vacation).
    """
    start, end = analysis_window(rows["transaction_date"])
    months_all = pd.date_range(start, end, freq="MS")
    window_rows = rows.loc[(rows["month"] >= start) & (rows["month"] <= end)].copy()

    dataset = build_employee_month_frame(window_rows, holidays, start, end, settings.fy_start_month)
    features = add_employee_features(dataset)
    train_months, test_months = split_months(months_all, settings.test_months)

    series = (
        window_rows.loc[window_rows["is_vacation"]].groupby("month")["hours"].sum()
        .reindex(months_all, fill_value=0.0)
    )
    pag_baseline = baseline_one_step_backtest(series, test_months)
    pag_seasonal = pag_baseline.loc[pag_baseline["model"] == SEASONAL_NAIVE].set_index("month")["forecast"]

    employee_bt = employee_backtest(features, test_months, settings.model)
    existing_bt = existing_xgboost_backtest(
        window_rows, holidays, months_all, test_months, settings.model, settings.fy_start_month
    )
    predictions = employee_bt.merge(existing_bt, on=KEY, how="inner")
    _check_totals(employee_bt, series, test_months)

    hours_per_occurrence = _hours_per_occurrence(window_rows)
    recursive_hold = recursive_holdout_forecast(features, test_months, settings.model, hours_per_occurrence)
    recursive_pred = predictions[KEY + ["actual"]].merge(
        recursive_hold[KEY + ["forecast"]], on=KEY, how="inner"
    )
    recursive_monthly = pd.DataFrame(
        {
            "actual": series.reindex(test_months),
            "forecast": recursive_hold.groupby("month")["forecast"].sum().reindex(test_months, fill_value=0.0),
        }
    )
    recursive_monthly.index.name = "month"
    recursive_pag_comparison = pd.DataFrame(
        [_metric_row(RECURSIVE, recursive_monthly["actual"], recursive_monthly["forecast"])]
    )
    recursive_employee_comparison = pd.DataFrame(
        [_metric_row(RECURSIVE, recursive_pred["actual"], recursive_pred["forecast"])]
    )

    # PAG totals: each model is summed from its own full output. Employee rows are only inner-joined
    # for the employee-level comparison.
    monthly_test = pd.DataFrame(
        {
            "actual": series.reindex(test_months),
            SEASONAL_NAIVE: pag_seasonal.reindex(test_months),
            EXISTING_XGBOOST: existing_bt.groupby("month")[EXISTING_XGBOOST].sum().reindex(test_months),
            EMPLOYEE_XGBOOST: employee_bt.groupby("month")[EMPLOYEE_XGBOOST].sum().reindex(test_months),
        }
    )
    monthly_test.index.name = "month"

    pag_rows = [_metric_row(model, monthly_test["actual"], monthly_test[model]) for model in MODEL_ORDER]
    pag_comparison = pd.DataFrame(pag_rows)
    employee_rows = [
        _metric_row(model, predictions["actual"], predictions[model]) for model in MODEL_ORDER
    ]
    employee_comparison = pd.DataFrame(employee_rows)
    winner = best_model(pag_comparison.rename(columns={"n_rows": "n_months"}))

    # Final model for the future forecast: trained on every complete month.
    final_model = fit_employee_model(features, settings.model)
    future_frame = extend_for_future(
        dataset, holidays, settings.forecast_horizon, settings.active_lookback_months, settings.fy_start_month
    )
    xgboost_future, contributions_by_month = recursive_employee_forecast(
        future_frame, final_model, hours_per_occurrence
    )
    if settings.production_model == SEASONAL_NAIVE:
        future_employees = seasonal_naive_future(future_frame)
    elif settings.production_model == EMPLOYEE_XGBOOST:
        future_employees = xgboost_future
    else:
        raise ValueError(f"Unknown production model: {settings.production_model}")
    future_months = pd.DatetimeIndex(sorted(future_employees["month"].unique()))
    same_month_last_year = series.reindex(future_months - pd.DateOffset(months=12)).to_numpy()
    future_monthly = pd.DataFrame(
        {
            "month": future_months,
            "forecast": future_employees.groupby("month")["forecast"].sum().reindex(future_months).to_numpy(),
            "employees": future_employees.groupby("month")["member_id"].nunique().reindex(future_months).to_numpy(),
            "same_month_last_year_actual": same_month_last_year,
        }
    )

    train_rows = int((features["month"].isin(train_months) & ~features["is_future"]).sum())
    logger.info(
        "Employee-level run: %d employee-month rows, %d train rows, %d holdout rows; best PAG model %s",
        len(dataset), train_rows, len(predictions), winner,
    )
    return EmployeeForecastRun(
        window=(start, end),
        train_months=train_months,
        test_months=test_months,
        dataset=dataset,
        train_rows=train_rows,
        test_rows=len(predictions),
        pag_series=series,
        monthly_test=monthly_test.reset_index(),
        xgboost_future_employees=xgboost_future,
        production_model=settings.production_model,
        pag_comparison=pag_comparison,
        employee_comparison=employee_comparison,
        predictions=predictions,
        best=winner,
        future_employees=future_employees,
        future_monthly=future_monthly,
        contributions_by_month=contributions_by_month,
        recursive_pag_comparison=recursive_pag_comparison,
        recursive_employee_comparison=recursive_employee_comparison,
        recursive_monthly=recursive_monthly.reset_index(),
    )
