# Model Validation: Solvay

## 1. Protocol

- Units: vacation days (hours / 9) per month, PAG totals or employee-months.
- Holdout: 6 months per origin, 2 origins, each on the same rule.
- Population: members with a timesheet row in the origin month. The forecast population is held fixed over the holdout.
- Recursive models use only data up to the origin and feed their own predictions forward.
- One-step models use actual values inside the holdout. They are reference only.
- WAPE = sum of absolute errors / sum of actuals x 100. Bias = mean of (forecast - actual).

## 2. PAG Level, Pooled Over Origins

| model | n_rows | MAE | RMSE | WAPE | Bias | MAE vs SN % | WAPE vs SN pts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Seasonal naive (same month last year) | 12 | 13.30 | 16.91 | 30.19 | -0.38 | 0.00 | 0.00 |
| Aggregate XGBoost, one-step (reference only) | 12 | 11.71 | 17.82 | 26.59 | 2.67 | -11.91 | -3.60 |
| Employee XGBoost, one-step (reference only) | 12 | 13.89 | 21.34 | 31.55 | 5.75 | 4.50 | 1.36 |
| Aggregate XGBoost, recursive | 12 | 15.06 | 21.37 | 34.19 | 9.00 | 13.25 | 4.00 |
| Employee XGBoost, recursive | 12 | 19.68 | 26.15 | 44.69 | 15.55 | 48.02 | 14.50 |

## 3. PAG Level, By Origin

Origin Feb 2025, holdout Mar 2025 to Aug 2025, population 135 members.

| model | n_rows | MAE | RMSE | WAPE | Bias | MAE vs SN % | WAPE vs SN pts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Seasonal naive (same month last year) | 6 | 9.00 | 9.76 | 33.03 | 2.33 | 0.00 | 0.00 |
| Aggregate XGBoost, one-step (reference only) | 6 | 4.88 | 6.83 | 17.91 | -2.74 | -45.76 | -15.11 |
| Employee XGBoost, one-step (reference only) | 6 | 6.30 | 6.87 | 23.12 | 1.61 | -30.01 | -9.91 |
| Aggregate XGBoost, recursive | 6 | 7.58 | 8.65 | 27.82 | 5.52 | -15.77 | -5.21 |
| Employee XGBoost, recursive | 6 | 12.76 | 15.14 | 46.82 | 12.76 | 41.75 | 13.79 |

Origin Aug 2024, holdout Sep 2024 to Feb 2025, population 132 members.

| model | n_rows | MAE | RMSE | WAPE | Bias | MAE vs SN % | WAPE vs SN pts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Seasonal naive (same month last year) | 6 | 17.59 | 21.83 | 28.92 | -3.09 | 0.00 | 0.00 |
| Aggregate XGBoost, one-step (reference only) | 6 | 18.54 | 24.26 | 30.48 | 8.09 | 5.41 | 1.56 |
| Employee XGBoost, one-step (reference only) | 6 | 21.49 | 29.38 | 35.33 | 9.89 | 22.16 | 6.41 |
| Aggregate XGBoost, recursive | 6 | 22.54 | 28.96 | 37.05 | 12.48 | 28.10 | 8.13 |
| Employee XGBoost, recursive | 6 | 26.61 | 33.73 | 43.74 | 18.34 | 51.23 | 14.82 |

## 4. Employee Level, Pooled

| model | n_rows | MAE | RMSE | WAPE | Bias | MAE vs SN % | WAPE vs SN pts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Seasonal naive (same month last year) | 1602 | 0.49 | 1.29 | 147.29 | -0.00 | 0.00 | 0.00 |
| Aggregate XGBoost, one-step (reference only) | 1602 | 0.44 | 0.95 | 134.57 | 0.02 | -8.64 | -12.72 |
| Employee XGBoost, one-step (reference only) | 1602 | 0.46 | 0.96 | 139.71 | 0.04 | -5.15 | -7.58 |
| Aggregate XGBoost, recursive | 1602 | 0.52 | 0.99 | 156.23 | 0.07 | 6.07 | 8.94 |
| Employee XGBoost, recursive | 1602 | 0.53 | 1.00 | 160.69 | 0.12 | 9.09 | 13.39 |

## 5. Recursive Inflation Check

Origin Feb 2025:

| Series | Holdout total (days) | vs actual | vs seasonal naive | vs historical average |
| --- | --- | --- | --- | --- |
| Actual | 163.50 | 1.00 | 0.92 | 0.61 |
| Seasonal naive | 177.50 | 1.09 | 1.00 | 0.67 |
| Historical monthly average | 266.82 | 1.63 | 1.50 | 1.00 |
| Aggregate XGBoost, recursive | 196.59 | 1.20 | 1.11 | 0.74 |
| Employee XGBoost, recursive | 240.04 | 1.47 | 1.35 | 0.90 |

Origin Aug 2024:

| Series | Holdout total (days) | vs actual | vs seasonal naive | vs historical average |
| --- | --- | --- | --- | --- |
| Actual | 365.00 | 1.00 | 1.05 | 1.38 |
| Seasonal naive | 346.44 | 0.95 | 1.00 | 1.31 |
| Historical monthly average | 265.22 | 0.73 | 0.77 | 1.00 |
| Aggregate XGBoost, recursive | 439.87 | 1.21 | 1.27 | 1.66 |
| Employee XGBoost, recursive | 475.04 | 1.30 | 1.37 | 1.79 |

## 6. Headcount Drift

- Origin Feb 2025: members in the population with a row in each holdout month: Mar 2025 134, Apr 2025 132, May 2025 129, Jun 2025 128, Jul 2025 81, Aug 2025 72 (of 135).
- Origin Aug 2024: members in the population with a row in each holdout month: Sep 2024 132, Oct 2024 130, Nov 2024 127, Dec 2024 124, Jan 2025 123, Feb 2025 119 (of 132).

Actual values for members who left are 0, but the forecast keeps them in the population.

## 7. End-to-End Forecast Check (Production Forecast)

Forecast hours / 9 = vacation days. Available person-days = employees x working days (Mon to Fri). Vacation % = vacation days / available person-days x 100. Availability % = 100 - vacation %.

| Month | Employees | Working days | Available person-days | Forecast vacation days | Vacation % | Availability % | Difference vs app (pts) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sep 2025 | 88 | 22 | 1936 | 28.00 | 1.45 | 98.55 | 0.00 |
| Oct 2025 | 88 | 23 | 2024 | 26.00 | 1.28 | 98.72 | 0.00 |
| Nov 2025 | 88 | 20 | 1760 | 50.50 | 2.87 | 97.13 | 0.00 |
| Dec 2025 | 88 | 23 | 2024 | 69.00 | 3.41 | 96.59 | 0.00 |
| Jan 2026 | 88 | 22 | 1936 | 42.50 | 2.20 | 97.80 | 0.00 |
| Feb 2026 | 88 | 20 | 1760 | 7.50 | 0.43 | 99.57 | 0.00 |
Maximum difference vs app numbers: 0.000000.

## 8. Recommendation

Use **Seasonal naive (same month last year)** for the application.

No recursive XGBoost model beats seasonal naive on both MAE and WAPE in the pooled holdout and in every origin. Use seasonal naive for the application forecast.

## 9. Limitations

- 2 origins only; the holdout is short, so differences are indicative.
- One-step rows use actual values inside the holdout and are not deployable as a 6-month forecast.
- The population is fixed over the horizon. Leavers stay in the forecast; new joiners are not added.
- Holiday counts are not part of the backtest decision because public-holiday exclusion is off.
