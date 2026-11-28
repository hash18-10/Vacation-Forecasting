"""Read raw timesheet workbooks into one DataFrame, mapping per-year headers to canonical names."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from vacation_forecasting.config import COLUMN_ALIASES, REQUIRED_COLUMNS, Settings

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".xlsb", ".csv")
SOURCE_FILE_COLUMN = "source_file"
SOURCE_ROW_COLUMN = "source_row"
SOURCE_SHEET_COLUMN = "source_sheet"
RAW_CACHE_NAME = "raw_timesheets_cache.pkl"
MAX_HEADER_ROW_SCAN = 20
# Reference workbook with fiscal month dates; not a timesheet.
FISCAL_CALENDAR_NAME = "Fiscal_Calendar.xlsb"


def header_key(name: object) -> str:
    """Comparison key for headers: lower-case, alphanumerics only."""
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


def discover_timesheet_files(raw_dir: Path, exclude: Path | None = None) -> list[Path]:
    """List timesheet files in the raw directory, skipping Excel lock files and the holiday calendar."""
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    excluded = exclude.resolve() if exclude else None
    files = sorted(
        path
        for path in raw_dir.iterdir()
        if path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith("~$")
        and path.name != FISCAL_CALENDAR_NAME
        and path.resolve() != excluded
    )
    if not files:
        raise FileNotFoundError(f"No timesheet files (.xlsb/.csv) found in {raw_dir}")
    return files


def resolve_columns(columns: Iterable[object], source_label: str) -> dict[str, str]:
    """Map canonical column names to the actual header in one file. Raises if a required column is absent."""
    lookup = {header_key(column): str(column) for column in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for candidate in (canonical, *aliases):
            key = header_key(candidate)
            if key in lookup:
                mapping[canonical] = lookup[key]
                break
    missing = [column for column in REQUIRED_COLUMNS if column not in mapping]
    if missing:
        raise ValueError(
            f"{source_label}: missing required columns {missing}. Headers found: {list(lookup.values())}"
        )
    return mapping


def _xlsb_engine() -> str:
    try:
        import python_calamine  # noqa: F401
        return "calamine"
    except ImportError as exc:
        try:
            import pyxlsb  # noqa: F401
        except ImportError:
            raise ImportError(
                "python-calamine or pyxlsb is required to read .xlsb files: "
                "pip install python-calamine"
            ) from exc
        return "pyxlsb"


def read_workbook(path: Path, sheet_name: str | int, header_row: int) -> pd.DataFrame:
    """Read one .xlsb (prefer Calamine, fall back to pyxlsb) or .csv file."""
    if path.suffix.lower() == ".xlsb":
        return pd.read_excel(path, engine=_xlsb_engine(), sheet_name=sheet_name, header=header_row)
    return pd.read_csv(path, header=header_row)


def sheet_for(path: Path, settings: Settings) -> str | int:
    """Worksheet to read for a file: an explicit override by file stem, else the default sheet."""
    return settings.sheet_overrides.get(path.stem, settings.sheet_name)


def _header_rows(preferred: int) -> list[int]:
    return [preferred, *(row for row in range(MAX_HEADER_ROW_SCAN + 1) if row != preferred)]


def _find_csv_header(path: Path, settings: Settings) -> tuple[int, dict[str, str]]:
    last_headers: list[str] = []
    for header_row in _header_rows(settings.header_overrides.get(path.stem, settings.header_row)):
        headers = pd.read_csv(path, header=header_row, nrows=0).columns
        try:
            return header_row, resolve_columns(headers, path.name)
        except ValueError:
            last_headers = [str(column) for column in headers]
    raise ValueError(
        f"{path.name}: required timesheet columns not found in the first "
        f"{MAX_HEADER_ROW_SCAN + 1} rows. Headers found near the end of the scan: {last_headers}"
    )


def _find_excel_layout(
    workbook: pd.ExcelFile, path: Path, settings: Settings
) -> tuple[str | int, int, dict[str, str]]:
    preferred_sheet = sheet_for(path, settings)
    sheets = workbook.sheet_names
    ordered_sheets = [preferred_sheet, *(sheet for sheet in sheets if sheet != preferred_sheet)]
    preferred_header = settings.header_overrides.get(path.stem, settings.header_row)
    last_headers: list[str] = []

    for sheet in ordered_sheets:
        if sheet not in sheets:
            continue
        for header_row in _header_rows(preferred_header):
            try:
                headers = workbook.parse(sheet_name=sheet, header=header_row, nrows=0).columns
            except ValueError:
                continue  # sheet is shorter than this header row
            try:
                mapping = resolve_columns(headers, path.name)
            except ValueError:
                last_headers = [str(column) for column in headers]
                continue
            return sheet, header_row, mapping

    raise ValueError(
        f"{path.name}: required timesheet columns not found in any worksheet/header row "
        f"(scanned rows 1-{MAX_HEADER_ROW_SCAN + 1}). Worksheets: {sheets}. "
        f"Headers found near the end of the scan: {last_headers}"
    )


def load_timesheet_file(path: Path, settings: Settings) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read one timesheet file and rename its columns to canonical names.

    Returns the frame and the canonical-to-source column mapping that was used for this file.
    """
    if path.suffix.lower() == ".xlsb":
        with pd.ExcelFile(path, engine=_xlsb_engine()) as workbook:
            sheet, header_row, mapping = _find_excel_layout(workbook, path, settings)
            raw = workbook.parse(sheet_name=sheet, header=header_row)
    else:
        header_row, mapping = _find_csv_header(path, settings)
        sheet = 0
        raw = pd.read_csv(path, header=header_row)
    frame = raw.rename(columns={raw_name: canonical for canonical, raw_name in mapping.items()})
    frame[SOURCE_FILE_COLUMN] = path.name
    frame[SOURCE_SHEET_COLUMN] = str(sheet)
    frame[SOURCE_ROW_COLUMN] = range(header_row + 2, header_row + 2 + len(frame))
    unmapped = [column for column in raw.columns if str(column) not in mapping.values()]
    logger.info(
        "Loaded %s: %d rows, sheet=%s, canonical columns=%s, other columns kept=%d",
        path.name,
        len(frame),
        sheet,
        sorted(mapping),
        len(unmapped),
    )
    return frame, mapping


def _raw_cache_key(files: list[Path], settings: Settings) -> str:
    """Fingerprint of the source files (name, size, mtime) and the read settings."""
    signature = [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in files]
    read_settings = (
        str(settings.sheet_name),
        tuple(sorted(settings.sheet_overrides.items())),
        settings.header_row,
        tuple(sorted(settings.header_overrides.items())),
    )
    return hashlib.sha256(repr((signature, read_settings)).encode()).hexdigest()


def load_raw_timesheets(settings: Settings) -> pd.DataFrame:
    """Load every timesheet file in the raw directory into one DataFrame.

    Canonical columns are renamed; all other source columns are kept so no data is dropped silently.
    The combined frame is cached in processed_dir and reused while the source files are unchanged.
    """
    files = discover_timesheet_files(settings.raw_dir, exclude=settings.holidays_csv)
    key = _raw_cache_key(files, settings)
    cache_path = settings.processed_dir / RAW_CACHE_NAME
    if cache_path.is_file():
        cached = pd.read_pickle(cache_path)
        if cached["key"] == key:
            logger.info("Using cached raw timesheets from %s", cache_path)
            return cached["frame"]
    frames = [load_timesheet_file(path, settings)[0] for path in files]
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d file(s) in %s", len(combined), len(files), settings.raw_dir)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"key": key, "frame": combined}, cache_path)
    return combined
