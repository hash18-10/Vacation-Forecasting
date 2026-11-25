"""End-to-end pipeline test on a synthetic CSV timesheet (no real data required)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from vacation_forecasting.config import ModelSettings
from vacation_forecasting.pipeline import run_pipeline


def test_pipeline_runs_end_to_end_and_writes_outputs(settings_for, synthetic_raw: pd.DataFrame) -> None:
    synthetic_raw.to_csv(settings_for.raw_dir / "FY_synthetic.csv", index=False)
    (settings_for.raw_dir / "holidays.csv").write_text(
            "date,holiday_name,holiday_type,location,source_file,year\n"
            "2023-12-25,Christmas,Restricted/Optional,India,test.pdf,2023\n"
            "2024-12-25,Christmas Holiday,Restricted/Optional,India,test.pdf,2024\n"
    )
    settings = replace(settings_for, forecast_horizon=3, model=ModelSettings(n_estimators=40, max_depth=2))

    result = run_pipeline(settings, write_outputs=True)

    assert result.best in {"seasonal_naive", "trailing_3m_avg", "trailing_12m_avg", "xgboost"}
    assert len(result.forecast) == 3
    assert (result.forecast["lower"] <= result.forecast["forecast"]).all()
    assert result.report_path is not None and result.report_path.exists()
    text = result.report_path.read_text(encoding="utf-8")
    for letter in "ABCDEFGHIJKLMN":
        assert f"## {letter}. " in text
    assert len(result.csv_paths) == 3
    assert all(path.exists() for path in result.csv_paths)
