# Vacation Forecasting POC

A standalone Python proof of concept that forecasts monthly vacation hours for one PAG from timesheet exports. It is a separate codebase: it does not read from or modify the Employee Calendar app.

## Scope

- Input: `.xlsb` timesheets (via the faster Calamine engine, with `pyxlsb` fallback). `.csv` files with the same columns are also accepted for offline use.
- Vacation is defined by TRC code only (default `105`). Descriptions are never used.
- Financial year runs October to September (FY2024 = Oct 2023 to Sep 2024).
- PAG spelling variants are normalized (for example `Solvay SA` to `Solvay`).
- Target: vacation hours per member-month, aggregated to PAG-month for reporting.
- Models: seasonal naive, trailing 3-month and 12-month averages, and XGBoost. All are compared on the same chronological holdout (one-step ahead).
- Revenue forecasting is out of scope.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Data

- Any `.xlsb` or `.csv` filename is accepted. Workbook sheet and header row are detected from the required timesheet columns (header rows 1-21 are scanned).
- Place timesheet files in `data/raw/`. Lock files (`~$*`) are ignored.
- `data/raw/holidays.csv` holds the holiday calendar with columns `date,holiday_name,location`. Use `ALL` for location to apply to every site. This file is excluded from timesheet discovery.
- Override the folder with `VF_RAW_DIR` or `--raw-dir`.

## Run

From the project root:

```powershell
$env:PYTHONPATH = "src"
python -m vacation_forecasting --pag Solvay --horizon 6
python -m vacation_forecasting --no-write      # run without writing files
```

Outputs go to `reports/output/` (PAG name lowercased, e.g. `solvay_*`):

- `solvay_backtest.csv` (Month, Actual, Forecast, Model, Error)
- `solvay_forecast.csv` (Month, Forecast, Lower, Upper)
- `solvay_member_forecast.csv`
- `solvay_vacation_report.md` (sections A to N)

With writes enabled, cleaned and rejected rows go to `data/processed/clean_timesheets.csv` and `data/processed/rejected_rows.csv`.

Optional Streamlit view:

```powershell
streamlit run app.py
```

## Tests

```powershell
python -m pytest -q
```

Tests use deterministic synthetic data, so no real timesheets are needed.

## Configuration

Settings live in `src/vacation_forecasting/config.py`. Environment overrides: `VF_PAG`, `VF_FORECAST_HORIZON`, `VF_RAW_DIR`, `VF_OUTPUT_DIR`. See `.env.example`.

## Limitations

- Results describe historical associations. They do not establish causes.
- With short histories, holdout error estimates and residual intervals are wide and indicative only.
- Real `.xlsb` files have not been validated in this environment; the loader was tested on synthetic CSV input. Check the column mapping in `COLUMN_ALIASES` against your export.
