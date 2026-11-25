"""Demo view model: runs the existing employee-level pipeline and shapes its outputs for the Streamlit app.

No forecast values are created here. Every number comes from the pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vacation_forecasting.business_view import BusinessView, build_business_view
from vacation_forecasting.config import Settings
from vacation_forecasting.employee_forecasting import EmployeeForecastRun, feature_weights, run_employee_level_forecast
from vacation_forecasting.insights import model_comparison_statements


@dataclass
class DemoResult:
    """Everything the app shows. Business tables come first; technical tables sit under Technical Details."""

    run: EmployeeForecastRun
    business: BusinessView
    feature_weights: pd.DataFrame
    model_lines: list[str]


def build_demo(pag_df: pd.DataFrame, holidays: pd.DataFrame, settings: Settings) -> DemoResult:
    """Run the employee-level pipeline, then the business view built from its outputs."""
    run = run_employee_level_forecast(pag_df, holidays, settings)
    business = build_business_view(pag_df, holidays, run, settings)
    return DemoResult(
        run=run,
        business=business,
        feature_weights=feature_weights(run.contributions_by_month),
        model_lines=model_comparison_statements(run.pag_table(), run.employee_table()),
    )
