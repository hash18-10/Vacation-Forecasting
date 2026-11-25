"""AI-Powered Vacation Forecasting demo. Run from the project root with: streamlit run app.py

Every number shown comes from the existing employee-level pipeline in src/vacation_forecasting.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vacation_forecasting.config import get_settings  # noqa: E402
from vacation_forecasting.data_loader import discover_timesheet_files  # noqa: E402
from vacation_forecasting.demo import DemoResult, build_demo  # noqa: E402
from vacation_forecasting.employee_forecasting import with_model_labels  # noqa: E402
from vacation_forecasting.employee_report import load_clean_all  # noqa: E402
from vacation_forecasting.holiday_analysis import load_holidays  # noqa: E402
from vacation_forecasting.vacation_analysis import available_pags, select_pag  # noqa: E402

DEFAULT_PAG = "Solvay"
VACATION_TYPE = "Vacation (TRC 105)"
# Validated in reports/model_validation.md on 6- and 12-month holdouts. Seasonal naive uses only actual history.
HORIZON_OPTIONS = (3, 6, 12)
SERIES_DOMAIN = ["Historical", "Forecast"]


def _save_uploads(uploads: list) -> Path:
    """Write uploaded timesheets to a temporary folder keyed by their names and sizes."""
    signature = repr(sorted((item.name, item.size) for item in uploads)).encode()
    folder = Path(tempfile.gettempdir()) / "vf_uploads" / hashlib.sha256(signature).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    for item in uploads:
        (folder / item.name).write_bytes(item.getbuffer())
    return folder


@st.cache_resource(show_spinner="Loading timesheets. The first load can take a few minutes...")
def _all_rows(raw_dir: Path) -> pd.DataFrame:
    """Clean, vacation-flagged rows for every PAG. Held once in memory, not copied per PAG."""
    return load_clean_all(replace(get_settings(), raw_dir=raw_dir))


@st.cache_data(show_spinner=False)
def _pag_names(raw_dir: Path) -> list[str]:
    return available_pags(_all_rows(raw_dir))


@st.cache_data(show_spinner="Running the employee-level forecast...")
def _demo(pag: str, horizon: int, raw_dir: Path) -> DemoResult:
    settings = replace(get_settings(), pag=pag, forecast_horizon=horizon, raw_dir=raw_dir)
    pag_df = select_pag(_all_rows(raw_dir), pag)
    holidays = load_holidays(settings.holidays_csv)
    return build_demo(pag_df, holidays, settings)


def _vacation_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> alt.Chart:
    """Vacation % by month. Solid line: historical. Dashed line: forecast."""
    data = pd.concat(
        [
            history[["month", "vacation_pct"]].assign(Series="Historical"),
            forecast[["month", "vacation_pct"]].assign(Series="Forecast"),
        ],
        ignore_index=True,
    ).dropna(subset=["vacation_pct"])
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("vacation_pct:Q", title="Vacation % of working days"),
            color=alt.Color(
                "Series:N", scale=alt.Scale(domain=SERIES_DOMAIN, range=["#1f77b4", "#ff7f0e"]), title="Series"
            ),
            strokeDash=alt.StrokeDash(
                "Series:N", scale=alt.Scale(domain=SERIES_DOMAIN, range=[[1, 0], [6, 4]]), legend=None
            ),
            tooltip=[
                alt.Tooltip("month:T", format="%b %Y", title="Month"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("vacation_pct:Q", format=".1f", title="Vacation %"),
            ],
        )
    )


def _metrics_view(frame: pd.DataFrame) -> pd.DataFrame:
    return with_model_labels(frame)[["model", "n_rows", "MAE", "RMSE", "WAPE", "Bias"]].round(2)


def _fmt(value: float, suffix: str = "") -> str:
    return "n/a" if pd.isna(value) else f"{value:.1f}{suffix}"


def _pp(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:+.1f} percentage points"


def _monthly_view(monthly: pd.DataFrame, days_label: str = "Vacation Days") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Month": monthly["month"].dt.strftime("%b %Y"),
            "Working Days": monthly["weekdays"],
            "Employee Count": monthly["employees"],
            days_label: monthly["vacation_days"].round(1),
            "Vacation %": monthly["vacation_pct"].round(1),
            "Availability %": monthly["availability_pct"].round(1),
        }
    )


st.set_page_config(page_title="AI-Powered Vacation Forecasting", layout="wide")
st.title("AI-Powered Vacation Forecasting")

uploads = st.file_uploader(
    "Timesheet files (.xlsb or .csv)", type=["xlsb", "csv"], accept_multiple_files=True
)
settings_default = get_settings()
if uploads:
    raw_dir = _save_uploads(uploads)
else:
    try:
        discover_timesheet_files(settings_default.raw_dir, exclude=settings_default.holidays_csv)
        raw_dir = settings_default.raw_dir
    except FileNotFoundError:
        st.info("Upload the timesheet files above to run the forecast.")
        st.stop()

pags = _pag_names(raw_dir)
if not pags:
    st.error("No PAGs found in the timesheet data.")
    st.stop()

left, right = st.columns(2)
pag = left.selectbox("PAG", pags, index=pags.index(DEFAULT_PAG) if DEFAULT_PAG in pags else 0)
horizon = right.selectbox(
    "Forecast period", HORIZON_OPTIONS, index=HORIZON_OPTIONS.index(6), format_func=lambda n: f"Next {n} months"
)
st.caption(f"Vacation type: {VACATION_TYPE}. Working days are Monday to Friday.")

try:
    demo = _demo(pag, int(horizon), raw_dir)
except (FileNotFoundError, ValueError, ImportError) as exc:
    st.error(str(exc))
    st.stop()

bv = demo.business
fs = bv.forecast_summary
fm = bv.forecast_monthly
if fs["months"] == 0:
    st.warning("No forecast is available for this PAG. It needs enough recent activity in the data.")
    st.stop()

st.caption(
    f"Forecast period: {fm['month'].min():%b %Y} to {fm['month'].max():%b %Y} ({fs['months']} months). "
    f"Last complete month in the data: {demo.run.window[1]:%b %Y}."
)

st.subheader(f"PAG: {pag}")
c1, c2, c3 = st.columns(3)
c1.metric("Forecast Vacation %", _fmt(fs["vacation_pct"], "%"))
c2.metric("Forecast Availability %", _fmt(fs["availability_pct"], "%"))
c3.metric("Number of Employees", f"{fs['employees']:,}")
c4, c5, c6 = st.columns(3)
c4.metric("Working Days", f"{fs['weekdays']:,}")
c5.metric("Forecast Vacation Days", f"{fs['vacation_days']:,.1f}")
c6.metric("Average Vacation Days / Employee", _fmt(fs["avg_vacation_days_per_employee"]))

same_last_year = bv.last_year_summary
c7, c8, c9 = st.columns(3)
if same_last_year is not None:
    c7.metric("Same Months Last Year Vacation %", _fmt(same_last_year["vacation_pct"], "%"))
    c8.metric(
        "Forecast vs Same Months Last Year",
        _pp(fs["vacation_pct"] - same_last_year["vacation_pct"]),
    )
else:
    c7.metric("Same Months Last Year Vacation %", "n/a")
    c8.metric("Forecast vs Same Months Last Year", "n/a")
c9.metric("Historical Average Vacation %", _fmt(bv.history_summary["vacation_pct"], "%"))
st.caption(
    "Employees are members with a timesheet row in the latest complete month, held constant over the forecast. "
    "Working days are Monday to Friday, summed over the forecast months."
)

st.subheader("Monthly Forecast")
st.dataframe(_monthly_view(fm, days_label="Forecast Vacation Days"), hide_index=True)
st.caption("Solid line: historical vacation %. Dashed line: forecast vacation %.")
st.altair_chart(_vacation_chart(bv.history_monthly, fm))

st.subheader("Historical Trend")
trend_rows = [
    ("Previous 12 months (actual)", bv.previous_period_summary),
    ("Same months last year (actual)", bv.last_year_summary),
    ("Forecast", fs),
]
trend = pd.DataFrame(
    [
        {
            "Period": name,
            "Vacation %": s["vacation_pct"],
            "Availability %": s["availability_pct"],
            "Avg Vacation Days / Employee": s["avg_vacation_days_per_employee"],
        }
        for name, s in trend_rows
        if s is not None
    ]
).round(1)
st.dataframe(trend, hide_index=True)
for line in bv.history_lines:
    st.markdown(f"- {line}")
with st.expander("Complete historical months"):
    st.dataframe(_monthly_view(bv.history_monthly), hide_index=True)

st.subheader("Forecast Insights")
for line in bv.forecast_lines:
    st.markdown(f"- {line}")


run = demo.run
st.header("Technical Details")
with st.expander("Settings and data checks"):
    holiday_text = "yes" if bv.exclude_public_holidays else "no (every Monday to Friday is a working day)"
    partial = ", ".join(f"{m:%b %Y}" for m in bv.excluded_partial_months) or "none"
    st.markdown(f"- Production model: {run.production_model} (same month last year), chosen in reports/model_validation.md")
    st.markdown(f"- Hours per vacation day: {bv.hours_per_vacation_day:g} h (full day; half days are 4.5 h)")
    st.markdown(f"- Public holidays excluded: {holiday_text}")
    st.markdown(f"- Partial months excluded from history: {partial}")
    if bv.issues:
        for issue in bv.issues:
            st.markdown(f"- {issue}")
    else:
        st.markdown("- No consistency issues found.")

with st.expander("Holdout evaluation"):
    st.caption(
        f"Holdout: {run.test_months.min():%b %Y} to {run.test_months.max():%b %Y}. Lower MAE is better. "
        "All models are scored on the same months."
    )
    st.subheader("PAG level (monthly totals)")
    st.dataframe(_metrics_view(run.pag_table()))
    st.subheader("Employee level (employee-months)")
    st.dataframe(_metrics_view(run.employee_table()))
    for line in demo.model_lines:
        st.markdown(f"- {line}")

with st.expander("Forecast hours and model inputs"):
    st.caption("Model output in hours before conversion to vacation days (hours / 9).")
    st.dataframe(
        run.future_monthly.assign(month=run.future_monthly["month"].dt.strftime("%b %Y")).rename(
            columns={"month": "Month", "forecast": "Forecast (h)", "employees": "Employees"}
        ).round(1),
        hide_index=True,
    )
    st.subheader("Recursive XGBoost forecast (comparison only, not used for the application forecast)")
    xgb_monthly = run.xgboost_future_employees.groupby("month")["forecast"].sum()
    compare = pd.DataFrame(
        {
            "Production forecast (h)": run.future_monthly.set_index("month")["forecast"],
            "Recursive XGBoost (h)": xgb_monthly,
        }
    ).round(1)
    compare.index = compare.index.strftime("%b %Y")
    st.dataframe(compare)
    st.caption(
        "Share of the XGBoost model's contribution by input, averaged over forecast months. These describe the "
        "fitted XGBoost model, not the application forecast, and do not show what causes vacation."
    )
    st.dataframe(
        demo.feature_weights.rename(columns={"signal": "Signal", "share_pct": "Share of contribution (%)"})[
            ["Signal", "Share of contribution (%)"]
        ].round(1)
    )

st.header("Limitations")
for line in [
    "Validation used two holdouts per horizon (6 and 12 months). Model differences are indicative only.",
    "The forecast repeats each member's vacation from the same month last year. It does not anticipate changes in headcount or planning.",
    "Vacation days assume 9 h per full day. Public holidays are not excluded unless the setting is turned on.",
    "Forecast employees are members with a timesheet row in the latest complete month, held constant over the forecast horizon. A different headcount changes the percentages.",
    "Some forecast inputs are built from the model's own predicted hours. The event-count inputs use observed data only.",
    "Explanations describe the fitted model. They are not causes.",
]:
    st.markdown(f"- {line}")
