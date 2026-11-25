# Real Data Validation: Solvay Vacation Dataset (FY23 to FY25)

## A. Scope and method

- Sources: 3 member-level workbooks and `holidays.csv` (118 rows), read through the pipeline loader and preprocessing.
- PAG: Solvay only; Solvay Pool and other variants are excluded.
- Financial year: Oct to Sep, assigned from the transaction date. The source `Year` label is not used.
- Vacation: TRC code 105.
- Location: the workbooks have no location column. The location code in DEPT DESC (HYD, BAN/BLR, CHE/CHN, MUM, PUN/PUNE) is mapped to the holiday-CSV location name.
- Scope: FY23 to FY25. Data ends 2025-09-20, so September 2025 is partial and excluded from averages, extremes and insights.

## B. Source files and column mapping

| file | sheet | header row | rows loaded | first_date | last_date | rows_outside_file_fy | year_label_mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FY23 | Memberwise Timesheet Summary | 1 | 778187 | 2022-09-25 | 2023-09-23 | 15554 | 167797 |
| FY24 | Memberwise Timesheet Summary | 2 | 785145 | 2023-09-24 | 2024-09-21 | 15199 | 15199 |
| FY25 | Memberwise Timesheet | 2 | 837326 | 2024-09-22 | 2025-09-20 | 19050 | 0 |

| canonical field | FY23 | FY24 | FY25 |
| --- | --- | --- | --- |
| member_id | MEMBER ID | MEMBER ID | MEMBER ID |
| transaction_date | TRANSACTION DATE | TRANSACTION DATE | TRANSACTION DATE |
| pag | PAG | PAG | PAG |
| trc_code | TRC CODE | TRC CODE | TRC CODE |
| trc_description | TRC DESC | TRC DESC | TRC DESC |
| hours | HOURS(DAILY) | HOURS(DAILY) | HOURS(DAILY) |
| location | not present | not present | not present |
| department_description | DEPT DESC | DEPT DESC | DEPT DESC |

## C. Data-quality checks

| area | check | result | status |
| --- | --- | --- | --- |
| Sources | Timesheet files loaded | FY23, FY24, FY25 | PASS |
| Sources | Required columns resolved in each file | All required canonical columns found | PASS |
| Sources | Row reconciliation | 2,400,658 loaded = 2,400,658 clean + 0 rejected | PASS |
| Sources | Rejected rows (missing or invalid key fields) | none | PASS |
| Dates | Transaction date range (all PAGs) | 2022-09-25 to 2025-09-20 | INFO |
| Dates | Rows outside the Oct-Sep FY implied by their file name | FY23: 15,554; FY24: 15,199; FY25: 19,050 (assigned to the FY of the transaction date) | WARN |
| Dates | Source Year label vs Oct-Sep date rule | FY23: 167,797 rows; FY24: 15,199 rows; FY25: no Year column | WARN |
| Scope | FY2022 rows (Sep 2022) outside FY23-FY25 scope | 15,554 rows excluded from metrics | INFO |
| Scope | Solvay variant PAGs excluded (in FY23-FY25) | MFG EU Solvay Pool: 7 rows, 0.0 vacation h; Solvay Pool: 130 rows, 216.0 vacation h | INFO |
| Vacation | TRC 105 descriptions in Solvay vacation rows | Vacation | PASS |
| Vacation | Non-positive vacation hours | 0 rows | PASS |
| Hours | Member-days above 24 total hours (Solvay, all TRC) | 0 member-days | PASS |
| Duplicates | Repeated vacation records (same member, date, TRC and hours) | 0 extra rows, 0.0 h | PASS |
| Location | Location source for Solvay rows | department description: 140,850 | INFO |
| Location | Vacation rows with unresolved location | 0 rows | PASS |
| Holidays | Holiday CSV parsed and validated | 118 rows; years [2023, 2024, 2025, 2026] | PASS |
| Holidays | Calendar years covering FY23-FY25 windows | missing: 2022 | WARN |
| Holidays | Location holidays for Bangalore | 2 location-specific rows; none for 2025 | WARN |
| Holidays | Location holidays for Chennai | 0 location-specific rows; none for 2023, 2024, 2025 | WARN |
| Holidays | Location holidays for Hyderabad | 3 location-specific rows; none for 2024 | WARN |
| Holidays | Location holidays for Mumbai | 3 location-specific rows; none for 2025 | WARN |
| Holidays | Location holidays for Pune | 3 location-specific rows; none for 2025 | WARN |
| Reconciliation | Vacation hours: FY vs monthly vs location totals | 16,727.5 h in all three views | PASS |
| Coverage | Latest FY in scope fully covered | data ends 2025-09-20; FY2025 ends 2025-09-30 | WARN |

## D. Financial-year totals and year-on-year change

| FY | First date | Last date | Solvay records | Employees | Total hours | Vacation records | Vacation employees | Vacation hours | Vacation share % | Employees YoY % | Vacation employees YoY % | Vacation hours YoY % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FY2023 | 2022-10-02 | 2023-09-30 | 44470 | 176 | 300,799.0 | 646 | 92 | 5,761.5 | 1.9 |  |  |  |
| FY2024 | 2023-10-01 | 2024-09-30 | 47529 | 163 | 303,080.9 | 657 | 85 | 5,845.5 | 1.9 | -7.4 | -7.6 | 1.5 |
| FY2025 | 2024-10-01 | 2025-09-19 | 48851 | 181 | 284,333.4 | 576 | 77 | 5,120.5 | 1.8 | 11.0 | -9.4 | -12.4 |

## E. Monthly vacation hours and employee counts

| Month | FY | Vacation hours | Vacation records | Vacation employees | Active employees | MoM % | YoY % | Applicable holidays |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-10 | FY2023 | 670.5 | 75 | 39 | 133 |  |  | 0 |
| 2022-11 | FY2023 | 481.5 | 54 | 31 | 132 | -28.2 |  | 0 |
| 2022-12 | FY2023 | 1,134.0 | 127 | 45 | 130 | 135.5 |  | 0 |
| 2023-01 | FY2023 | 540.0 | 60 | 26 | 130 | -52.4 |  | 3 |
| 2023-02 | FY2023 | 373.5 | 42 | 23 | 127 | -30.8 |  | 0 |
| 2023-03 | FY2023 | 423.0 | 47 | 23 | 126 | 13.3 |  | 3 |
| 2023-04 | FY2023 | 297.0 | 33 | 19 | 126 | -29.8 |  | 3 |
| 2023-05 | FY2023 | 351.0 | 39 | 20 | 127 | 18.2 |  | 2 |
| 2023-06 | FY2023 | 319.5 | 36 | 22 | 131 | -9.0 |  | 2 |
| 2023-07 | FY2023 | 231.5 | 27 | 16 | 130 | -27.5 |  | 0 |
| 2023-08 | FY2023 | 544.5 | 61 | 23 | 142 | 135.2 |  | 4 |
| 2023-09 | FY2023 | 395.5 | 45 | 24 | 145 | -27.4 |  | 4 |
| 2023-10 | FY2024 | 432.0 | 48 | 24 | 131 | 9.2 | -35.6 | 3 |
| 2023-11 | FY2024 | 603.0 | 70 | 29 | 129 | 39.6 | 25.2 | 5 |
| 2023-12 | FY2024 | 1,377.0 | 154 | 44 | 129 | 128.4 | 21.4 | 1 |
| 2024-01 | FY2024 | 531.0 | 59 | 20 | 126 | -61.4 | -1.7 | 4 |
| 2024-02 | FY2024 | 477.0 | 53 | 21 | 129 | -10.2 | 27.7 | 1 |
| 2024-03 | FY2024 | 351.0 | 39 | 21 | 131 | -26.4 | -17.0 | 3 |
| 2024-04 | FY2024 | 315.0 | 35 | 17 | 131 | -10.3 | 6.1 | 4 |
| 2024-05 | FY2024 | 270.0 | 31 | 19 | 129 | -14.3 | -23.1 | 3 |
| 2024-06 | FY2024 | 360.0 | 41 | 23 | 131 | 33.3 | 12.7 | 1 |
| 2024-07 | FY2024 | 243.0 | 28 | 16 | 134 | -32.5 | 5.0 | 1 |
| 2024-08 | FY2024 | 445.5 | 50 | 25 | 132 | 83.3 | -18.2 | 3 |
| 2024-09 | FY2024 | 441.0 | 49 | 22 | 138 | -1.0 | 11.5 | 2 |
| 2024-10 | FY2025 | 441.0 | 50 | 27 | 136 | 0.0 | 2.1 | 4 |
| 2024-11 | FY2025 | 805.5 | 91 | 36 | 134 | 82.7 | 33.6 | 4 |
| 2024-12 | FY2025 | 931.5 | 104 | 36 | 134 | 15.6 | -32.4 | 1 |
| 2025-01 | FY2025 | 657.0 | 74 | 25 | 138 | -29.5 | 23.7 | 4 |
| 2025-02 | FY2025 | 283.0 | 33 | 15 | 135 | -56.9 | -40.7 | 2 |
| 2025-03 | FY2025 | 288.0 | 33 | 18 | 142 | 1.8 | -17.9 | 2 |
| 2025-04 | FY2025 | 288.0 | 32 | 16 | 150 | 0.0 | -8.6 | 3 |
| 2025-05 | FY2025 | 360.0 | 40 | 19 | 152 | 25.0 | 33.3 | 2 |
| 2025-06 | FY2025 | 247.5 | 28 | 15 | 150 | -31.2 | -31.2 | 1 |
| 2025-07 | FY2025 | 333.0 | 37 | 8 | 102 | 34.5 | 37.0 | 0 |
| 2025-08 | FY2025 | 342.0 | 38 | 13 | 88 | 2.7 | -23.2 | 3 |
| 2025-09 (partial) | FY2025 | 144.0 | 16 | 8 | 89 | -57.9 | -67.3 | 2 |

September 2025 is partial because the data ends on 2025-09-20.

## F. Same-period year-on-year (Oct to Aug, 11 complete months)

| financial_year_label | vacation_hours | months | yoy_pct |
| --- | --- | --- | --- |
| FY2023 | 5,366.0 | 11 |  |
| FY2024 | 5,404.5 | 11 | 0.7 |
| FY2025 | 4,976.5 | 11 | -7.9 |

## G. Highest and lowest complete months

| period | type | month | vacation hours | vacation employees |
| --- | --- | --- | --- | --- |
| FY2023 | Highest | Dec 2022 | 1,134.0 | 45 |
| FY2023 | Lowest | Jul 2023 | 231.5 | 16 |
| FY2024 | Highest | Dec 2023 | 1,377.0 | 44 |
| FY2024 | Lowest | Jul 2024 | 243.0 | 16 |
| FY2025 | Highest | Dec 2024 | 931.5 | 36 |
| FY2025 | Lowest | Jun 2025 | 247.5 | 15 |
| All complete months | Highest | Dec 2023 | 1,377.0 | 44 |
| All complete months | Lowest | Jul 2023 | 231.5 | 16 |

## H. Locations and holiday applicability

**Employees and vacation hours by location**

| location | FY2023 employees | FY2023 vacation hours | FY2024 employees | FY2024 vacation hours | FY2025 employees | FY2025 vacation hours |
| --- | --- | --- | --- | --- | --- | --- |
| Bangalore | 78 | 2,155.5 | 57 | 1,971.0 | 55 | 1,619.5 |
| Chennai | 5 | 234.0 | 4 | 261.0 | 7 | 234.0 |
| Hyderabad | 92 | 3,372.0 | 101 | 3,568.5 | 117 | 3,267.0 |
| Mumbai | 1 | 0.0 | 1 | 45.0 | 1 | 0.0 |
| Pune | 0 | 0.0 | 0 | 0.0 | 1 | 0.0 |

**Holidays applicable to each location**

| location | CSV rows for location | FY2023 applicable holidays | FY2024 applicable holidays | FY2025 applicable holidays |
| --- | --- | --- | --- | --- |
| Bangalore | 2 | 19 | 30 | 27 |
| Chennai | 0 | 19 | 29 | 26 |
| Hyderabad | 3 | 20 | 29 | 27 |
| Mumbai | 3 | 20 | 30 | 26 |
| Pune | 3 | 20 | 30 | 26 |

**Holiday calendar coverage by FY window**

| financial year | window | CSV years needed | missing CSV years |
| --- | --- | --- | --- |
| FY2023 | Oct 2022 to Sep 2023 | 2022, 2023 | 2022 |
| FY2024 | Oct 2023 to Sep 2024 | 2023, 2024 | none |
| FY2025 | Oct 2024 to Sep 2025 | 2024, 2025 | none |

## I. Holiday-window patterns (weekdays, descriptive only)

| Holiday | Date | Window avg total vacation hours per day | Other days avg total vacation hours per day | Difference total vacation hours per day |
| --- | --- | --- | --- | --- |
| Christmas | 2023-12-25 | 100.12 | 20.76 | 79.36 |
| New Year's Day | 2024-01-01 | 74.25 | 21.03 | 53.22 |
| New Year's Day | 2025-01-01 | 61.07 | 21.22 | 39.85 |
| Christmas Holiday | 2024-12-25 | 58.50 | 21.25 | 37.25 |
| Uzhavar Thirunal | 2023-01-17 | 41.14 | 21.41 | 19.74 |
| Id-e-Milad | 2023-09-28 | 7.71 | 21.71 | -14.00 |
| Telangana State Formation Day | 2023-06-02 | 6.75 | 21.74 | -14.99 |
| Buddha Purnima | 2024-05-23 | 5.79 | 21.73 | -15.94 |
| Gandhi Jayanti | 2023-10-02 | 4.50 | 21.76 | -17.26 |
| Eid al-Adha (Bakrid) | 2023-06-29 | 0.00 | 21.78 | -21.78 |

79 holidays evaluated; 32 with a higher window average. Median difference -3.62 total vacation hours per day. Weekdays only, +/- 5 days. Descriptive timing association only.

## J. Insights

- On average, vacation hours were highest in December (1,147.5 h) and lowest in July (269.2 h) across the observed years.
- FY2025 vacation hours were 5,120.5 h, or 1.8% of recorded hours.
- Compared with FY2024, vacation hours are down 12.4% in FY2025.
- The most recent year-on-year change in a month was -23.2%.
- The holiday window with the highest average daily vacation was Christmas (2023-12-25), with +79.36 h per day against other days. This is a timing association only and does not show cause.

## K. Open issues and decisions

- **Dates**: Rows outside the Oct-Sep FY implied by their file name - FY23: 15,554; FY24: 15,199; FY25: 19,050 (assigned to the FY of the transaction date) (WARN)
- **Dates**: Source Year label vs Oct-Sep date rule - FY23: 167,797 rows; FY24: 15,199 rows; FY25: no Year column (WARN)
- **Holidays**: Calendar years covering FY23-FY25 windows - missing: 2022 (WARN)
- **Holidays**: Location holidays for Bangalore - 2 location-specific rows; none for 2025 (WARN)
- **Holidays**: Location holidays for Chennai - 0 location-specific rows; none for 2023, 2024, 2025 (WARN)
- **Holidays**: Location holidays for Hyderabad - 3 location-specific rows; none for 2024 (WARN)
- **Holidays**: Location holidays for Mumbai - 3 location-specific rows; none for 2025 (WARN)
- **Holidays**: Location holidays for Pune - 3 location-specific rows; none for 2025 (WARN)
- **Coverage**: Latest FY in scope fully covered - data ends 2025-09-20; FY2025 ends 2025-09-30 (WARN)
