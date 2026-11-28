"""AI-Powered Vacation Forecasting demo. Run from the project root with: streamlit run app.py

Every number shown comes from the existing employee-level pipeline in src/vacation_forecasting.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vacation_forecasting.config import LEAVE_TRC_CODES, get_settings  # noqa: E402
from vacation_forecasting.data_loader import SOURCE_FILE_COLUMN, discover_timesheet_files, load_raw_timesheets  # noqa: E402
from vacation_forecasting.demo import DemoResult, build_demo  # noqa: E402
from vacation_forecasting.employee_leave_forecast import (  # noqa: E402
    build_employee_monthly_forecast,
    build_leave_forecast,
    ranked_months,
)
from vacation_forecasting.employee_forecasting import with_model_labels  # noqa: E402
from vacation_forecasting.holiday_analysis import load_holidays  # noqa: E402
from vacation_forecasting.normalization import financial_year_end, financial_year_label  # noqa: E402
from vacation_forecasting.preprocessing import preprocess_timesheets  # noqa: E402
from vacation_forecasting.vacation_analysis import (
    add_vacation_flag,
    available_pags,
    employee_leave_lookup,
    select_pag,
)  # noqa: E402

FORECAST_HORIZON = 12
VACATION_TYPE = "Vacation (TRC 105)"
# Validated in reports/model_validation.md on 6- and 12-month holdouts. Seasonal naive uses only actual history.
SERIES_DOMAIN = ["Historical", "Forecast"]


@st.cache_resource(show_spinner=False)
def _load_timesheets(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate timesheets once, returning clean rows and per-file validation counts."""
    settings = replace(get_settings(), raw_dir=raw_dir)
    raw = load_raw_timesheets(settings)
    result = preprocess_timesheets(raw, settings)
    clean = add_vacation_flag(result.clean, settings.vacation_trc_codes)

    loaded_counts = raw.groupby(SOURCE_FILE_COLUMN).size().rename("Rows loaded")
    clean_counts = result.clean.groupby(SOURCE_FILE_COLUMN).size().rename("Rows accepted")
    rejected_counts = result.rejected.groupby(SOURCE_FILE_COLUMN).size().rename("Rows rejected")
    validation = pd.concat([loaded_counts, clean_counts, rejected_counts], axis=1).fillna(0).astype(int)
    return clean, validation.reset_index()


@st.cache_data(show_spinner="Running the employee-level forecast...")
def _demo(pag: str, manager: str | None, director: str | None, horizon: int, raw_dir: Path) -> DemoResult:
    settings = replace(get_settings(), pag=pag, forecast_horizon=horizon, raw_dir=raw_dir)
    all_rows, _ = _load_timesheets(raw_dir)
    pag_df = select_pag(all_rows, pag)
    if manager is not None:
        pag_df = pag_df.loc[pag_df["manager"].astype(str).str.strip().eq(manager)]
    if director is not None:
        pag_df = pag_df.loc[pag_df["director"].astype(str).str.strip().eq(director)]
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

settings_default = get_settings()
raw_dir = settings_default.raw_dir
try:
    discover_timesheet_files(raw_dir, exclude=settings_default.holidays_csv)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.status("Validating timesheet files...", expanded=False) as validation_status:
    st.write("Reading workbooks and checking required columns and row values. Large files may take a few minutes.")
    try:
        all_rows, validation = _load_timesheets(raw_dir)
    except Exception as exc:
        validation_status.update(label="Timesheet validation failed", state="error")
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    validation_status.update(label="Timesheet validation complete", state="complete")
    st.write(f"Validated {len(validation)} file(s): {validation['Rows loaded'].sum():,} rows loaded, "
             f"{validation['Rows accepted'].sum():,} accepted, "
             f"{validation['Rows rejected'].sum():,} rejected.")
    st.dataframe(validation, hide_index=True)

st.subheader("Employee Leave Lookup")
employee_directory = all_rows[["member_id", "member_name"]].dropna(subset=["member_id"]).copy()
employee_directory["member_name"] = employee_directory["member_name"].fillna("").astype(str).str.strip()
employee_directory = employee_directory.drop_duplicates("member_id")
employee_directory["display_name"] = employee_directory["member_name"].where(
    employee_directory["member_name"].ne(""), "Employee"
) + " (" + employee_directory["member_id"].astype(str) + ")"
employee_directory = employee_directory.sort_values("display_name")

if employee_directory.empty:
    st.info("No employees are available in the data.")
else:
    employee_options = employee_directory.set_index("display_name")["member_id"].to_dict()
    selected_employee = st.selectbox("Employee name", list(employee_options))
    check_date = pd.Timestamp.today().date()
    selected_member_id = str(employee_options[selected_employee])
    last_year_date = pd.Timestamp(check_date) - pd.DateOffset(years=1)
    day_rows, leave_history = employee_leave_lookup(all_rows, selected_member_id, check_date)
    leave_on_day = day_rows.loc[day_rows["trc_code"].isin(LEAVE_TRC_CODES)]

    st.caption(
        f"Checking {last_year_date:%d %b %Y}. Four-year history window: "
        f"{(pd.Timestamp(check_date) - pd.DateOffset(years=4)):%d %b %Y} to {pd.Timestamp(check_date):%d %b %Y}."
    )
    if not leave_on_day.empty:
        st.success(f"Leave was recorded on {last_year_date:%d %b %Y}: {leave_on_day['hours'].sum():g} hours.")
        st.dataframe(
            leave_on_day[["transaction_date", "trc_code", "trc_description", "hours"]].rename(
                columns={
                    "transaction_date": "Date",
                    "trc_code": "TRC",
                    "trc_description": "Description",
                    "hours": "Hours",
                }
            ),
            hide_index=True,
        )
    elif day_rows.empty:
        st.warning(f"No timesheet entry was found for {last_year_date:%d %b %Y}; leave cannot be confirmed.")
    else:
        st.info(f"Timesheet entries exist on {last_year_date:%d %b %Y}, but no leave was recorded.")

    if leave_history.empty:
        st.info("No leave records were found in this four-year window.")
    else:
        fy_end = financial_year_end(leave_history["transaction_date"])
        history_view = leave_history.assign(
            Year=fy_end.map(lambda y: f"{financial_year_label(y)} (Oct {int(y) - 1} - Sep {int(y)})"),
            Date=leave_history["transaction_date"].dt.strftime("%d %b %Y"),
        )
        yearly_by_type = history_view.pivot_table(
            index="Year", columns="trc_description", values="hours", aggfunc="sum", fill_value=0.0
        )
        yearly_by_type["Total leave hours"] = yearly_by_type.sum(axis=1)
        yearly_by_type["Leave entries"] = history_view.groupby("Year").size()
        st.dataframe(yearly_by_type.sort_index(ascending=False).reset_index(), hide_index=True)
        year_labels = sorted(history_view["Year"].unique(), reverse=True)
        month_numbers = {
            pd.Timestamp(2000, m, 1).strftime("%B"): m
            for m in sorted(history_view["transaction_date"].dt.month.unique())
        }
        filter_year, filter_month = st.columns(2)
        selected_year = filter_year.selectbox("Leave year", ["All years", *year_labels])
        selected_month = filter_month.selectbox("Leave month", ["All months", *month_numbers])
        detail_view = history_view
        if selected_year != "All years":
            detail_view = detail_view.loc[detail_view["Year"].eq(selected_year)]
        if selected_month != "All months":
            detail_view = detail_view.loc[detail_view["transaction_date"].dt.month.eq(month_numbers[selected_month])]
        if detail_view.empty:
            st.info("No leave entries match the selected year and month.")
        st.dataframe(
            detail_view[["Date", "trc_description", "hours", SOURCE_FILE_COLUMN]].rename(
                columns={
                    "trc_description": "Description",
                    "hours": "Hours",
                    SOURCE_FILE_COLUMN: "Source file",
                }
            ),
            hide_index=True,
        )

    st.markdown(f"#### Leave forecast: {selected_employee}")
    leave_forecast = build_leave_forecast(
        all_rows, selected_member_id, all_rows["month"].max(), pd.Timestamp.today().date()
    )
    ranked = ranked_months(leave_forecast.profile)
    if ranked.empty:
        st.info("No leave history for this employee, so no forecast can be made.")
    else:
        top = ranked.iloc[0]
        st.success(
            f"{top['Month']} usually has the most leave: {top['avg_leave_days']:.1f} days on average over "
            f"{int(top['years_observed'])} year(s). Expect similar leave in {top['Month']}."
        )
        st.dataframe(
            ranked.rename(
                columns={
                    "Month": "Calendar month",
                    "avg_leave_days": "Average leave days",
                    "years_observed": "Years of history",
                }
            ).round(1),
            hide_index=True,
        )
        employee_monthly = build_employee_monthly_forecast(
            all_rows,
            selected_member_id,
            all_rows["month"].max(),
            pd.Timestamp.today().date(),
            load_holidays(settings_default.holidays_csv),
        )
        st.markdown("##### Monthly Forecast")
        st.dataframe(_monthly_view(employee_monthly, days_label="Forecast Vacation Days"), hide_index=True)
        st.caption(
            "Forecast = average vacation days for the same calendar month over this employee's complete months "
            "in the data (1 day = 9 h). Vacation % = vacation days / working days (Monday to Friday)."
        )

st.caption(f"Vacation type: {VACATION_TYPE}. Working days are Monday to Friday.")

pags = available_pags(all_rows)
largest_pag = all_rows.groupby("pag")["member_id"].nunique().idxmax()
filter_pag, filter_manager, filter_director = st.columns(3)
selected_pag = filter_pag.selectbox("PAG", pags, index=pags.index(largest_pag))
pag_people = select_pag(all_rows, selected_pag)
managers = sorted(pag_people["manager"].dropna().astype(str).str.strip().loc[lambda s: s.ne("")].unique())
directors = sorted(pag_people["director"].dropna().astype(str).str.strip().loc[lambda s: s.ne("")].unique())
selected_manager = filter_manager.selectbox("Manager", ["All managers", *managers])
selected_director = filter_director.selectbox("Director", ["All directors", *directors])
manager_filter = None if selected_manager == "All managers" else selected_manager
director_filter = None if selected_director == "All directors" else selected_director
scope = " / ".join([selected_pag, selected_manager, selected_director])

try:
    demo = _demo(selected_pag, manager_filter, director_filter, FORECAST_HORIZON, raw_dir)
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

st.subheader(f"PAG Monthly Forecast: {scope}")
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
