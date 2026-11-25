"""Shared test fixtures: deterministic synthetic timesheets (no real data is required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vacation_forecasting.config import Settings


def build_synthetic_timesheets(
    start: str = "2021-10-01", months: int = 36, members: int = 12, seed: int = 0
) -> pd.DataFrame:
    """Build a raw-style DataFrame with canonical headers and seasonal vacation (TRC 105) rows."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    month_starts = pd.date_range(start, periods=months, freq="MS")
    for month in month_starts:
        seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * (month.month - 1) / 12)
        for member in range(1, members + 1):
            for day in range(1, 29, 7):
                date = month + pd.Timedelta(days=day - 1)
                is_vacation = rng.random() < 0.1 * seasonal
                rows.append(
                    {
                        "Employee ID": f"E{member:03d}",
                        "Employee Name": f"Member {member}",
                        "Transaction Date": date,
                        "PAG": "Solvay SA" if member % 2 else "Solvay",
                        "TRC Code": 105 if is_vacation else 101,
                        "TRC Description": "Vacation" if is_vacation else "Project work",
                        "Hours": 8.0,
                        "Location": "DE",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    return build_synthetic_timesheets()


@pytest.fixture
def settings_for(tmp_path: Path) -> Settings:
    raw = tmp_path / "raw"
    raw.mkdir()
    return Settings(
        raw_dir=raw,
        holidays_csv=raw / "holidays.csv",
        output_dir=tmp_path / "out",
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )
