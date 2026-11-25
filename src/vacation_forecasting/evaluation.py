"""Evaluation metrics, model comparison and residual-based forecast intervals (PAG-level monthly)."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_COLUMNS = ("MAE", "RMSE", "WAPE", "Bias")


def compute_metrics(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """MAE, RMSE, WAPE (%) and Bias (mean forecast minus actual; positive means over-forecast)."""
    a = actual.astype(float).to_numpy()
    f = forecast.astype(float).to_numpy()
    if len(a) == 0:
        raise ValueError("Cannot compute metrics on zero observations")
    err = f - a
    total_actual = np.abs(a).sum()
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "WAPE": float(np.abs(err).sum() / total_actual * 100) if total_actual else float("nan"),
        "Bias": float(np.mean(err)),
    }


def compare_models(backtest: pd.DataFrame) -> pd.DataFrame:
    """Metrics per model. `backtest` needs columns: month, model, actual, forecast."""
    rows = []
    for model, group in backtest.groupby("model"):
        rows.append({"model": model, "n_months": len(group), **compute_metrics(group["actual"], group["forecast"])})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def best_model(comparison: pd.DataFrame) -> str:
    """Model with the lowest MAE on the holdout (ties broken by WAPE)."""
    ranked = comparison.sort_values(["MAE", "WAPE"]).reset_index(drop=True)
    return str(ranked.loc[0, "model"])


def largest_errors(backtest: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Holdout rows with the largest absolute error."""
    out = backtest.assign(abs_error=(backtest["forecast"] - backtest["actual"]).abs())
    return out.sort_values("abs_error", ascending=False).head(n).reset_index(drop=True)


def overfitting_check(train_metrics: dict[str, float], test_metrics: dict[str, float]) -> dict[str, float | str]:
    """Compare in-sample and holdout MAE. A large ratio suggests overfitting."""
    train_mae = train_metrics["MAE"]
    test_mae = test_metrics["MAE"]
    ratio = test_mae / train_mae if train_mae > 0 else float("inf")
    if ratio > 2:
        verdict = "possible overfitting: holdout MAE is more than twice the training MAE"
    elif ratio > 1.2:
        verdict = "mild gap between training and holdout error"
    else:
        verdict = "no material gap between training and holdout error"
    return {"train_MAE": train_mae, "test_MAE": test_mae, "ratio": float(ratio), "verdict": verdict}


def residual_intervals(
    backtest: pd.DataFrame, forecast: pd.DataFrame, level: float = 0.8
) -> pd.DataFrame:
    """Approximate interval from empirical holdout residuals (actual minus forecast).

    `forecast` needs columns month, forecast. Intervals are not guaranteed coverage; with few
    holdout points they are wide and indicative only.
    """
    residuals = (backtest["actual"] - backtest["forecast"]).to_numpy()
    alpha = (1 - level) / 2
    lo, hi = np.quantile(residuals, [alpha, 1 - alpha]) if len(residuals) else (0.0, 0.0)
    out = forecast[["month", "forecast"]].copy()
    out["lower"] = (out["forecast"] + lo).clip(lower=0.0)
    out["upper"] = out["forecast"] + hi
    return out
