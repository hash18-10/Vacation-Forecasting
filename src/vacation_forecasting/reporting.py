"""Write forecast CSVs and the Markdown report (sections A to N)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def df_to_markdown(frame: pd.DataFrame, float_digits: int = 1) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table without requiring the tabulate package."""
    if frame.empty:
        return "_No rows._"

    def _fmt(value: object) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:,.{float_digits}f}"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m")
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def write_csv_outputs(output_dir: Path, pag: str, backtest: pd.DataFrame,
                      forecast: pd.DataFrame, member_forecast: pd.DataFrame) -> list[Path]:
    """Write the backtest, PAG forecast and member forecast CSVs. Returns the paths written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    comparison_rows = backtest.assign(Error=lambda d: d["forecast"] - d["actual"])
    comparison_rows = comparison_rows.rename(columns={"month": "Month", "actual": "Actual",
                                                      "forecast": "Forecast", "model": "Model"})
    path = output_dir / f"{pag.lower()}_backtest.csv"
    comparison_rows[["Month", "Actual", "Forecast", "Model", "Error"]].to_csv(path, index=False)
    written.append(path)

    path = output_dir / f"{pag.lower()}_forecast.csv"
    forecast.rename(columns={"month": "Month", "forecast": "Forecast", "lower": "Lower", "upper": "Upper"})[
        ["Month", "Forecast", "Lower", "Upper"]
    ].to_csv(path, index=False)
    written.append(path)

    path = output_dir / f"{pag.lower()}_member_forecast.csv"
    member_forecast.to_csv(path, index=False)
    written.append(path)
    return written


def render_report(title: str, sections: Sequence[tuple[str, str, str]]) -> str:
    """Assemble the Markdown report. Each section is (letter, heading, body_markdown)."""
    parts = [f"# {title}", ""]
    for letter, heading, body in sections:
        parts.extend([f"## {letter}. {heading}", "", body.strip(), ""])
    return "\n".join(parts)


def write_report(output_dir: Path, pag: str, markdown: str) -> Path:
    """Write the report next to the CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{pag.lower()}_vacation_report.md"
    path.write_text(markdown, encoding="utf-8")
    return path
