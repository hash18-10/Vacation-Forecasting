# Recursive Forecast Diagnostic (Sep 2025 to Feb 2026)

Diagnostic only. No model tuning, no feature changes, no split changes, no vacation-definition changes, and no change to historical validation results. Production code is unchanged. The diagnostic module is `src/vacation_forecasting/recursive_diagnostic.py`.

## 1. Summary

The recursive forecast is about 2x last year's same months (Oct 1,119 vs 441; Dec 1,828 vs 932). The evidence points to a feedback loop as the main cause:

- The model predicts a small positive value for 84% of holdout employee-months, while only 11% have actual vacation.
- Each predicted value is fed back into the history. Any value above 0 is counted as a vacation month or occurrence by the count and frequency features.
- Those counts therefore grow with each recursive step, and the growth carries into the future months.

This is a ranked diagnosis, not a tested fix. The floor counterfactual (Section 4) reduces the holdout overshoot but does not fully remove it.

## 2. Root cause ranking

| Rank | Cause | Status | Evidence |
|---|---|---|---|
| 1 | Feedback of tiny positive predictions into count and frequency features (`vacation_months_prev12`, `vacation_occurrences_prev12`, `hist_vacation_frequency`) | Strongly supported | Section 3 and Section 7 |
| 2 | Feedback drift compounds over six steps, so later months carry more inflated history | Supported (same mechanism) | Section 3 and Section 7 |
| 3 | Constant set of 159 employees held for all six months | Minor contributor | Headcount ratio is about 1.16x against last year's 137 rows in Oct, so it cannot explain 2x |
| 4 | `prior_months_observed` moves outside the training range | Unverified contributor | 43% of future rows exceed the training max of 34. Tree models clamp to the training range, so the direct effect should be bounded |
| 5 | Leakage | Ruled out | Section 6 |
| 6 | Aggregation error | Ruled out | Section 8 |

## 3. One-step vs recursive comparison (holdout, Mar to Aug 2025)

| Month | Actual | One-step | Recursive | Employee seasonal naive |
|---|---|---|---|---|
| 2025-03 | 288.0 | 336.6 | 336.6 | 261.0 |
| 2025-04 | 288.0 | 321.3 | 355.8 | 234.0 |
| 2025-05 | 360.0 | 324.0 | 409.6 | 234.0 |
| 2025-06 | 247.5 | 332.7 | 440.2 | 270.0 |
| 2025-07 | 333.0 | 212.3 | 308.8 | 117.0 |
| 2025-08 | 342.0 | 221.4 | 345.7 | 207.0 |
| **Total** | **1,858.5** | **1,748.3 (-6%)** | **2,196.7 (+18%)** | **1,323.0 (-29%)** |

Prediction behaviour (employee-month level):

| Series | Mean prediction | Share of predictions above 0 |
|---|---|---|
| Actual | 2.35 h | 11% |
| One-step | 2.21 h | 84% |
| Recursive | 2.78 h | 83% |

## 4. Feedback floor counterfactual

The floor sets any prediction below the floor to 0 before it is fed back. This is a counterfactual only, not a proposed change.

| Month | Floor 0 (production) | Floor 0.5 | Floor 1.0 | Actual |
|---|---|---|---|---|
| 2025-03 | 336.6 | 328.2 | 314.9 | 288.0 |
| 2025-04 | 355.8 | 344.7 | 336.6 | 288.0 |
| 2025-05 | 409.6 | 392.5 | 373.0 | 360.0 |
| 2025-06 | 440.2 | 411.3 | 393.9 | 247.5 |
| 2025-07 | 308.8 | 283.7 | 256.7 | 333.0 |
| 2025-08 | 345.7 | 307.4 | 283.1 | 342.0 |
| **Total** | **2,196.7** | **2,067.8** | **1,958.2** | **1,858.5** |

A floor of 1.0 removes about 70% of the holdout overshoot (+338 h to +100 h), but the result is still above actual and above the one-step total. Thresholding is not sufficient on its own.

## 5. Example employee trace (employee 481132)

This employee had 0 vacation for most of the early holdout, then 189 h in Jul and 9 h in Aug.

| Month | Actual | One-step | Pred (fed back) | Lag 1 | Same month last year | Months prev 12 (count) | Occurrences prev 12 | Hist frequency | Prior months observed |
|---|---|---|---|---|---|---|---|---|---|
| 2025-03 | 0.0 | 1.55 | 1.55 | 0.00 | 18.0 | 6.0 | 20.00 | 0.38 | 29 |
| 2025-04 | 0.0 | 1.17 | 1.44 | 1.55 | 0.0 | 6.0 | 18.17 | 0.40 | 30 |
| 2025-05 | 0.0 | 0.00 | 2.26 | 1.44 | 9.0 | 7.0 | 18.34 | 0.42 | 31 |
| 2025-06 | 0.0 | 0.00 | 3.70 | 2.26 | 0.0 | 7.0 | 17.59 | 0.44 | 32 |
| 2025-07 | 189.0 | 0.00 | 3.50 | 3.70 | 9.0 | 8.0 | 18.01 | 0.45 | 33 |
| 2025-08 | 9.0 | 9.49 | 3.61 | 3.50 | 18.0 | 8.0 | 17.40 | 0.47 | 34 |

The trace shows the mechanism. The model's predicted values (1.4 to 3.7 h) are above 0, so each one increments the vacation-month count. The count rises from 6 to 8 and the frequency rises from 0.38 to 0.47 even though the employee had almost no real vacation in those months. The one-step model, which uses actual history, predicts 0 in May and June.

## 6. Feature values during recursion

Mean by month, one-step (actual history) vs recursive (predicted history), holdout:

| Feature | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 |
|---|---|---|---|---|---|---|
| `vacation_months_prev12`, one-step | 1.94 | 1.85 | 1.72 | 1.74 | 1.73 | 1.51 |
| `vacation_months_prev12`, recursive | 1.94 | 2.55 | 3.08 | 3.73 | 4.47 | 5.00 |
| `hist_vacation_frequency`, one-step | 0.182 | 0.174 | 0.157 | 0.157 | 0.148 | 0.137 |
| `hist_vacation_frequency`, recursive | 0.182 | 0.275 | 0.350 | 0.375 | 0.395 | 0.415 |
| `vacation_occurrences_prev12`, one-step | 4.29 | 4.13 | 3.75 | 3.85 | 3.92 | 3.91 |
| `vacation_occurrences_prev12`, recursive | 4.29 | 4.16 | 3.88 | 4.00 | 4.25 | 4.20 |
| `vacation_prev12_sum`, one-step | 38.0 | 36.5 | 33.1 | 34.1 | 34.9 | 34.9 |
| `vacation_prev12_sum`, recursive | 38.0 | 36.9 | 34.3 | 35.5 | 37.8 | 37.5 |

The count and frequency features diverge sharply. The sum feature (`vacation_prev12_sum`) diverges only slightly, which is consistent with the predicted values being small. The inflation comes from counting positive predictions as events.

Future months (production run, mean by month):

| Month | `vacation_months_prev12` | `hist_vacation_frequency` | `vacation_occurrences_prev12` | `vacation_prev12_sum` |
|---|---|---|---|---|
| 2025-09 | 1.49 | 0.148 | 3.58 | 31.8 |
| 2025-10 | 2.35 | 0.224 | 3.75 | 33.3 |
| 2025-11 | 3.20 | 0.276 | 4.26 | 37.9 |
| 2025-12 | 3.99 | 0.316 | 4.55 | 40.5 |
| 2026-01 | 4.77 | 0.349 | 5.22 | 46.4 |
| 2026-02 | 5.62 | 0.377 | 5.83 | 51.8 |

The future values follow the same upward drift as the holdout. Frequency has nearly tripled by February.

## 7. Seasonal-naive comparison (production future)

| Month | Production forecast | Employee seasonal naive | PAG same month last year | Employee rows same month last year |
|---|---|---|---|---|
| 2025-09 | 669.3 | 432.0 | 441.0 | 139 |
| 2025-10 | 1,119.1 | 387.0 | 441.0 | 137 |
| 2025-11 | 1,158.6 | 756.0 | 805.5 | 136 |
| 2025-12 | 1,827.8 | 877.5 | 931.5 | 136 |
| 2026-01 | 1,467.4 | 612.0 | 657.0 | 140 |
| 2026-02 | 1,373.3 | 247.0 | 283.0 | 137 |

The seasonal-naive baseline is the reference. The production forecast is 1.5x to 2.5x above it in every future month. This comparison does not validate either method. The holdout accuracy is not claimed here, and XGBoost is not claimed to be better than the baseline.

## 8. Leakage verdict

No leakage. The check perturbed each month's target and occurrences by +1000 and confirmed that the same month's own features did not change. Lags, rolling sums and expanding history all use shifted or prior-month data.

## 9. Aggregation correctness

Correct. The replica of the production recursion matches with a maximum difference of 0.0. Production sums employee forecasts by month and counts unique members (159 in every future month). The employee-level actuals reconcile with the PAG totals in the earlier validation.

## 10. Recommended fix (not implemented)

Awaiting approval before any change, since the request was diagnostic only.

1. Stop treating predicted values as vacation events. Compute `vacation_months_prev12`, `vacation_occurrences_prev12` and `hist_vacation_frequency` from actual history only. During recursion, freeze these features at their last observed values, or use a separate count that only increments on actual data.
2. Re-run the same recursion on the holdout after the change and compare against actual and the seasonal-naive baseline before trusting any future output.
3. Decide whether the 159-employee set should stay constant over six months, or whether employees should drop out based on the historical pattern for each month.
4. Review `prior_months_observed` (43% of future rows exceed the training max). Cap or re-scale it, or evaluate whether it should be a feature at all.

Note: the six-month holdout (791 employee-month rows) is too small to support an accuracy claim, so any change should be judged on the holdout and then on a longer backtest.
