"""Spreadsheets (.xlsx/.csv) -> TABLE-AWARE. Preserve rows/cols/units/formulas.

Critical: write each value to the `structured_facts` table so spec_lookup returns
EXACT numbers. Do NOT flatten a calc sheet into prose and lose the values.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..types import ParsedDoc, Table

# Matches unit annotation: "Temp [°C]" -> "°C"
_UNIT_RE = re.compile(r"\[([^\]]+)\]$")


def _detect_unit(col_name: str) -> str | None:
    m = _UNIT_RE.search(col_name.strip())
    return m.group(1) if m else None


def _df_to_table(df: pd.DataFrame, name: str) -> Table:
    """Convert a DataFrame to a Table, extracting units from bracket notation."""
    units: dict[str, str] = {}
    for col in df.columns:
        unit = _detect_unit(str(col))
        if unit:
            units[str(col)] = unit

    headers = [str(c) for c in df.columns]
    data_rows = [
        [str(v) if pd.notna(v) else "" for v in row]
        for _, row in df.iterrows()
    ]
    return Table(name=name, rows=[headers] + data_rows, units=units)


def _df_summary(df: pd.DataFrame, sheet_name: str) -> str:
    """Short prose summary of a sheet for semantic retrieval."""
    cols = ", ".join(str(c) for c in df.columns)
    return f"Tabelle '{sheet_name}': {len(df)} Zeilen, Spalten: {cols}"


def parse(path: str | Path) -> ParsedDoc:
    """Parse .xlsx or .csv into a ParsedDoc with Table objects and prose summaries.

    Each sheet (xlsx) or the whole file (csv) becomes one Table.
    Unit annotations in column headers like 'Temp [°C]' are stored in Table.units.
    The text field contains short summaries so the sheet is also reachable via
    semantic search, while the tables flow into structured_facts for exact lookup.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        tables = [_df_to_table(df, path.stem)]
        text = _df_summary(df, path.stem)
    elif suffix == ".xlsx":
        xl = pd.ExcelFile(path)
        tables: list[Table] = []
        summaries: list[str] = []
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            tables.append(_df_to_table(df, sheet_name))
            summaries.append(_df_summary(df, sheet_name))
        text = "\n\n".join(summaries)
    else:
        raise ValueError(f"sheet_parser: unsupported extension '{suffix}'")

    return ParsedDoc(
        text=text,
        tables=tables,
        metadata={"filename": path.name, "type": "sheet"},
        source_ref={"filename": path.name},
    )
