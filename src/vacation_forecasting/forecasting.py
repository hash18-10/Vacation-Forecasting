"""Baseline and XGBoost forecasting.

Backtests are one-step-ahead on a chronological holdout: each test month is predicted only from
data before that month. Future forecasts are recursive, feeding each predicted month back in as history.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from vacation_forecasting.config import ModelSettings
from vacation_forecasting.features import FEATURE_COLUMNS, TARGET, build_features

logger = logging.getLogger(__name__)

BASELINE_MODELS: tuple[str, ...] = ("seasonal_naive", "trailing_3m_avg", "trailing_12m_avg")
XGBOOST_MODEL = "xgboost"
KEY = ["member_id", "month"]


def _baseline_value(history: pd.Series, month: pd.Timestamp, model: str) -> float:
    """Single-step baseline value for `month` from a history indexed by month (strictly earlier months)."""
    if model == "seasonal_naive":
        same_month_last_year = month - pd.DateOffset(months=12)
        if same_month_last_year in history.index:
            return float(history.loc[same_month_last_year])
        return float(history.iloc[-12:].mean()) if len(history) else 0.0
    window = 3 if model == "trailing_3m_avg" else 12
    if model not in BASELINE_MODELS:
        raise ValueError(f"Unknown baseline: {model}")
    return float(history.iloc[-window:].mean()) if len(history) else 0.0


def baseline_one_step_backtest(series: pd.Series, test_months: pd.DatetimeIndex) -> pd.DataFrame:
    """For each test month, forecast from the actuals before it (no test-period leakage)."""
    rows = []
    for model in BASELINE_MODELS:
        for month in test_months:
            history = series.loc[series.index < month]
            rows.append(
                {"month": month, "model": model, "forecast": max(_baseline_value(history, month, model), 0.0)}
            )
    return pd.DataFrame(rows)


def baseline_recursive_forecast(series: pd.Series, future_months: pd.DatetimeIndex) -> pd.DataFrame:
    """Recursive baseline forecast: each predicted month is appended to the history used for the next one."""
    rows = []
    for model in BASELINE_MODELS:
        working = series.copy()
        for month in future_months:
            value = max(_baseline_value(working, month, model), 0.0)
            working.loc[month] = value
            rows.append({"month": month, "model": model, "forecast": value})
    return pd.DataFrame(rows)


def fit_xgboost(train: pd.DataFrame, settings: ModelSettings) -> XGBRegressor:
    """Fit XGBoost on known-target rows. Missing lag values are handled natively by XGBoost."""
    known = train.loc[train[TARGET].notna() & ~train["is_future"]]
    if known.empty:
        raise ValueError("No training rows with known targets")
    model = XGBRegressor(
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
    model.fit(known[list(FEATURE_COLUMNS)].astype(float), known[TARGET].astype(float))
    return model


def predict_xgboost(model: XGBRegressor, frame: pd.DataFrame) -> np.ndarray:
    """Predict non-negative vacation hours."""
    preds = model.predict(frame[list(FEATURE_COLUMNS)].astype(float))
    return np.clip(preds, 0.0, None)


def xgboost_one_step_backtest(
    panel: pd.DataFrame, test_months: pd.DatetimeIndex, settings: ModelSettings,
    holiday_counts: pd.Series | None, fy_start_month: int,
) -> pd.DataFrame:
    """Train on months before the test window; predict test months using actual history (one-step)."""
    features = build_features(panel, holiday_counts, fy_start_month)
    cutoff = test_months.min()
    model = fit_xgboost(features.loc[features["month"] < cutoff], settings)
    test = features.loc[features["month"].isin(test_months) & ~features["is_future"]].copy()
    test["forecast"] = predict_xgboost(model, test)
    test["model"] = XGBOOST_MODEL
    return test[KEY + ["forecast", "model", TARGET]]


def xgboost_recursive_forecast(
    panel_with_future: pd.DataFrame, settings: ModelSettings,
    holiday_counts: pd.Series | None, fy_start_month: int,
) -> pd.DataFrame:
    """Train on all history, then forecast month by month, feeding predictions back in as history."""
    features = build_features(panel_with_future, holiday_counts, fy_start_month)
    model = fit_xgboost(features, settings)

    work = panel_with_future.copy()
    future_months = sorted(work.loc[work["is_future"], "month"].unique())
    for month in future_months:
        feats = build_features(work, holiday_counts, fy_start_month)
        step = feats.loc[feats["is_future"] & (feats["month"] == month)]
        preds = predict_xgboost(model, step)
        lookup = pd.Series(preds, index=pd.MultiIndex.from_frame(step[KEY]))
        mask = work["is_future"] & (work["month"] == month)
        work.loc[mask, TARGET] = lookup.reindex(
            pd.MultiIndex.from_frame(work.loc[mask, KEY])
        ).to_numpy()

    result = work.loc[work["is_future"], KEY + [TARGET]].rename(columns={TARGET: "forecast"})
    result["model"] = XGBOOST_MODEL
    logger.info("XGBoost recursive forecast produced %d member-month rows", len(result))
    return result


def aggregate_to_pag(member_frame: pd.DataFrame, value_col: str = "forecast") -> pd.Series:
    """Sum member-month values to a monthly PAG-level series."""
    return member_frame.groupby("month")[value_col].sum().sort_index()
