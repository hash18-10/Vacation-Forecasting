"""Diagnostic for the recursive employee forecast. Read-only: uses production functions, changes no model settings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vacation_forecasting.config import get_settings
from vacation_forecasting.employee_features import (
    FEATURE_COLUMNS,
    KEY,
    TARGET,
    add_employee_features,
    analysis_window,
    build_employee_month_frame,
    extend_for_future,
)
from vacation_forecasting.employee_forecasting import (
    EMPLOYEE_XGBOOST,
    EXISTING_XGBOOST,
    SEASONAL_NAIVE,
    fit_employee_model,
    predict_employee_model,
    recursive_employee_forecast,
    run_employee_level_forecast,
    split_months,
)
from vacation_forecasting.employee_report import load_clean_pag
from vacation_forecasting.evaluation import compute_metrics
from vacation_forecasting.holiday_analysis import load_holidays

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def recursive_run(hold: pd.DataFrame, model, hours_per_occurrence: float, floor: float = 0.0, trace: bool = False):
    """Same loop as production recursive_employee_forecast, plus optional feedback floor and per-step trace."""
    work = hold.copy()
    parts = []
    for month in sorted(work.loc[work["is_future"], "month"].unique()):
        feats = add_employee_features(work)
        step = feats.loc[feats["is_future"] & feats["month"].eq(month)]
        preds = predict_employee_model(model, step)
        fed = np.where(preds < floor, 0.0, preds)
        if trace:
            parts.append(step[KEY + list(FEATURE_COLUMNS)].assign(pred=preds, fed=fed))
        keyed = pd.Series(fed, index=pd.MultiIndex.from_frame(step[KEY]))
        mask = work["is_future"] & work["month"].eq(month)
        values = keyed.reindex(pd.MultiIndex.from_frame(work.loc[mask, KEY])).to_numpy()
        work.loc[mask, TARGET] = values
        work.loc[mask, "vacation_occurrences"] = values / hours_per_occurrence
    final = add_employee_features(work)
    future = final.loc[final["is_future"]].copy().rename(columns={TARGET: "forecast"})
    trace_df = pd.concat(parts, ignore_index=True) if trace else None
    return future, trace_df


def leakage_check(dataset: pd.DataFrame) -> list[tuple[pd.Timestamp, list[str]]]:
    """Perturb the target of one month; any feature of that same month that changes is leakage."""
    cols = list(FEATURE_COLUMNS)
    base = add_employee_features(dataset)
    violations = []
    for month in sorted(dataset["month"].unique()):
        pert = dataset.copy()
        sel = pert["month"].eq(month)
        pert.loc[sel, TARGET] += 1000.0
        pert.loc[sel, "vacation_occurrences"] += 1000.0
        alt = add_employee_features(pert)
        a = base.loc[base["month"].eq(month), cols].fillna(-999.0).to_numpy()
        b = alt.loc[alt["month"].eq(month), cols].fillna(-999.0).to_numpy()
        changed = np.where(~np.isclose(a, b).all(axis=0))[0]
        if changed.size:
            violations.append((month, [cols[i] for i in changed]))
    return violations


def main() -> None:
    settings = get_settings()
    pag_df = load_clean_pag(settings)
    holidays = load_holidays(settings.holidays_csv)

    start, end = analysis_window(pag_df["transaction_date"])
    months_all = pd.date_range(start, end, freq="MS")
    window_rows = pag_df.loc[(pag_df["month"] >= start) & (pag_df["month"] <= end)].copy()
    hpo = float(window_rows.loc[window_rows["is_vacation"], "hours"].sum()
                / max(int(window_rows["is_vacation"].sum()), 1))
    dataset = build_employee_month_frame(window_rows, holidays, start, end, settings.fy_start_month)
    features = add_employee_features(dataset)
    train_months, test_months = split_months(months_all, settings.test_months)
    model = fit_employee_model(features.loc[features["month"] < test_months.min()], settings.model)

    # 1. One-step predictions on the holdout (actual history at every step).
    holdout = features.loc[features["month"].isin(test_months) & ~features["is_future"]].copy()
    holdout["onestep"] = predict_employee_model(model, holdout)

    # 2. Recursive predictions on the same holdout months (history after the split comes only from predictions).
    hold = features.copy()
    hm = hold["month"].isin(test_months)
    hold.loc[hm, [TARGET, "vacation_occurrences", "actual_event"]] = np.nan
    hold.loc[hm, "is_future"] = True
    rec, trace = recursive_run(hold, model, hpo, trace=True)
    prod_emp, _ = recursive_employee_forecast(hold, model, hpo)
    replica_diff = (
        rec[KEY + ["forecast"]].merge(prod_emp[KEY + ["forecast"]], on=KEY, suffixes=("_mine", "_prod"))
    )
    print("== Replica check: max |replica - production recursive| =",
          float((replica_diff["forecast_mine"] - replica_diff["forecast_prod"]).abs().max()))

    # 3. Monthly one-step vs recursive vs actual vs employee seasonal-naive (holdout).
    monthly = pd.DataFrame({
        "actual": holdout.groupby("month")[TARGET].sum(),
        "one_step": holdout.groupby("month")["onestep"].sum(),
        "recursive": rec.groupby("month")["forecast"].sum(),
        "emp_seasonal_naive": holdout.assign(sn=holdout["vacation_same_month_last_year"].fillna(0.0))
        .groupby("month")["sn"].sum(),
    })
    print("\n== Holdout monthly totals (hours)")
    print(monthly.round(1).to_string())

    print("\n== Per employee-month: mean prediction and share of positive values")
    print(pd.DataFrame({
        "actual_mean": [holdout[TARGET].mean(), holdout[TARGET].mean()],
        "one_step_mean": [holdout["onestep"].mean(), np.nan],
        "recursive_mean": [np.nan, rec["forecast"].mean()],
        "actual_share_pos": [(holdout[TARGET] > 0).mean(), (holdout[TARGET] > 0).mean()],
        "one_step_share_pos": [(holdout["onestep"] > 0).mean(), np.nan],
        "recursive_share_pos": [np.nan, (rec["forecast"] > 0).mean()],
    }, index=["actual vs one_step", "actual vs recursive"]).round(4).to_string())

    # 4. Feedback features: one-step (actual history) vs recursive (predicted history), monthly means.
    fb_cols = ["vacation_months_prev12", "hist_vacation_frequency", "vacation_occurrences_prev12", "vacation_prev12_sum"]
    fb_one = holdout.groupby("month")[fb_cols].mean()
    fb_rec = trace.groupby("month")[fb_cols].mean()
    print("\n== Feedback features, one-step (actual history) vs recursive (predicted history)")
    for col in fb_cols:
        print(f"{col}: one_step={fb_one[col].round(3).tolist()} recursive={fb_rec[col].round(3).tolist()}")

    # 5. Counterfactual: feed back only predictions at or above a floor. Diagnostic only, not a fix.
    print("\n== Counterfactual feedback floor (hours). Floor 0 = production behaviour")
    cf = {}
    for floor in (0.0, 0.5, 1.0):
        alt, _ = recursive_run(hold, model, hpo, floor=floor)
        cf[f"floor_{floor}"] = alt.groupby("month")["forecast"].sum()
    cf_df = pd.DataFrame(cf)
    cf_df["actual"] = monthly["actual"]
    print(cf_df.round(1).to_string())

    # 6. One employee trace (holdout, has >= 12 months of history before the holdout).
    eligible = holdout.loc[holdout["prior_months_observed"] >= 12, "member_id"].unique()
    holdout_totals = holdout.groupby("member_id")[TARGET].sum().loc[eligible]
    pick = holdout_totals.idxmax()
    trace_one = trace.loc[trace["member_id"].eq(pick)].merge(
        holdout.loc[holdout["member_id"].eq(pick), KEY + [TARGET, "onestep"]], on=KEY
    )
    trace_cols = [
        "month", TARGET, "onestep", "pred", "fed", "vacation_lag_1", "vacation_same_month_last_year",
        "vacation_prev3_sum", "vacation_prev12_sum", "vacation_occurrences_prev12",
        "vacation_months_prev12", "prior_months_observed", "hist_vacation_frequency", "hist_avg_vacation",
    ]
    print(f"\n== Trace employee {pick}")
    print(trace_one[trace_cols].round(2).to_string(index=False))

    # 7. Leakage: changing month m's target must not change month m's features.
    violations = leakage_check(dataset)
    print("\n== Leakage check: months whose own features change when their target changes:",
          violations if violations else "none")

    # 8. Production future: same recursion as production, aggregated by month.
    train_max_pmo = float(features["prior_months_observed"].max())
    model_all = fit_employee_model(features, settings.model)
    future_frame = extend_for_future(
        dataset, holidays, settings.forecast_horizon, settings.active_lookback_months, settings.fy_start_month
    )
    prod_fut, _ = recursive_run(future_frame, model_all, hpo)
    pag_by_month = window_rows.loc[window_rows["is_vacation"]].groupby("month")["hours"].sum()
    fut_months = pd.DatetimeIndex(sorted(prod_fut["month"].unique()))
    grid_counts = dataset.groupby("month").size()
    fut_table = pd.DataFrame({
        "forecast_total": prod_fut.groupby("month")["forecast"].sum(),
        "employees_forecast": prod_fut.groupby("month")["member_id"].nunique(),
        "pag_same_month_last_year": pag_by_month.reindex(fut_months - pd.DateOffset(months=12)).to_numpy(),
        "employee_seasonal_naive": prod_fut.assign(sn=prod_fut["vacation_same_month_last_year"].fillna(0.0))
        .groupby("month")["sn"].sum(),
        "emp_rows_same_month_last_year": grid_counts.reindex(fut_months - pd.DateOffset(months=12)).to_numpy(),
    })
    print("\n== Production future (Sep 2025 onward)")
    print(fut_table.round(1).to_string())
    print("\n== prior_months_observed: training max", train_max_pmo,
          "| future min/max", float(prod_fut["prior_months_observed"].min()),
          float(prod_fut["prior_months_observed"].max()),
          "| share of future rows above training max", round(float((prod_fut["prior_months_observed"] > train_max_pmo).mean()), 3))
    print("\n== Future feedback features (mean by month)")
    print(prod_fut.groupby("month")[fb_cols].mean().round(3).to_string())


def compare_holdout() -> None:
    """Post-fix 6-month recursive holdout next to the one-step XGBoost and seasonal-naive baselines."""
    settings = get_settings()
    pag_df = load_clean_pag(settings)
    holidays = load_holidays(settings.holidays_csv)
    run = run_employee_level_forecast(pag_df, holidays, settings)
    start, end = run.window
    window_rows = pag_df.loc[(pag_df["month"] >= start) & (pag_df["month"] <= end)]
    hpo = float(window_rows.loc[window_rows["is_vacation"], "hours"].sum()
                / max(int(window_rows["is_vacation"].sum()), 1))
    features = add_employee_features(run.dataset)
    model = fit_employee_model(features.loc[features["month"] < run.test_months.min()], settings.model)

    hold = features.copy()
    hm = hold["month"].isin(run.test_months)
    hold.loc[hm, [TARGET, "vacation_occurrences", "actual_event"]] = np.nan
    hold.loc[hm, "is_future"] = True
    rec, trace = recursive_run(hold, model, hpo, trace=True)

    emp = run.predictions.merge(rec[KEY + ["forecast"]], on=KEY, how="inner")
    model_cols = {
        "recursive (post-fix)": "forecast",
        "employee XGBoost one-step": EMPLOYEE_XGBOOST,
        "existing aggregate XGBoost one-step": EXISTING_XGBOOST,
        "seasonal naive": SEASONAL_NAIVE,
    }

    def metric_row(name: str, actual: pd.Series, forecast: pd.Series) -> dict:
        return {"model": name, "n_rows": len(actual), **compute_metrics(actual, forecast)}

    monthly = run.monthly_test.set_index("month")
    monthly["forecast"] = rec.groupby("month")["forecast"].sum().reindex(run.test_months).to_numpy()

    print(f"== Employee-month metrics, holdout ({len(emp)} rows)")
    print(pd.DataFrame([metric_row(n, emp["actual"], emp[c]) for n, c in model_cols.items()]).round(3).to_string(index=False))
    print("\n== PAG monthly metrics, holdout (6 months)")
    print(pd.DataFrame([metric_row(n, monthly["actual"], monthly[c]) for n, c in model_cols.items()]).round(3).to_string(index=False))
    cols = ["actual"] + list(dict.fromkeys(model_cols.values()))
    print("\n== Monthly totals (hours)")
    print(monthly[cols].round(1).to_string())

    fb_cols = ["vacation_months_prev12", "hist_vacation_frequency", "vacation_occurrences_prev12", "vacation_prev12_sum"]
    step_means = trace.groupby("month")[fb_cols].mean()
    step_means["recursive_total"] = rec.groupby("month")["forecast"].sum()
    print("\n== Inflation check: feedback features and monthly total by holdout step")
    print(step_means.round(3).to_string())


if __name__ == "__main__":
    compare_holdout()
