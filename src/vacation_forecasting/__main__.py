"""Command-line entry point: python -m vacation_forecasting [--pag NAME] [--horizon N] [--no-write]."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from vacation_forecasting.config import get_settings
from vacation_forecasting.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vacation_forecasting", description="Vacation forecasting POC")
    parser.add_argument("--pag", help="PAG to analyse (default: Solvay or VF_PAG)")
    parser.add_argument("--horizon", type=int, help="Forecast horizon in months (default: 6)")
    parser.add_argument("--raw-dir", help="Folder containing .xlsb/.csv timesheets")
    parser.add_argument("--no-write", action="store_true", help="Run without writing CSV or report files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    overrides = {}
    if args.pag:
        overrides["pag"] = args.pag
    if args.horizon:
        overrides["forecast_horizon"] = args.horizon
    if args.raw_dir:
        overrides["raw_dir"] = Path(args.raw_dir)
        overrides["holidays_csv"] = Path(args.raw_dir) / "holidays.csv"
    settings = replace(settings, **overrides)

    try:
        result = run_pipeline(settings, write_outputs=not args.no_write)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1

    print(f"PAG: {result.pag}")
    print(result.comparison.to_string(index=False))
    print(f"Best model (holdout MAE): {result.best}")
    print(result.forecast.to_string(index=False))
    if result.report_path:
        print(f"Report: {result.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
