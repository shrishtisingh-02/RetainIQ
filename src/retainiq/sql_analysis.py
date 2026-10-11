"""Run the SQL insight queries against the cleaned data using SQLite."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from . import config

_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
            "UnitPrice", "CustomerID", "Country", "Revenue"]


def parse_queries(sql_text: str) -> dict[str, str]:
    """Split a SQL file into {name: query} using '-- name: x' markers."""
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", sql_text, flags=re.MULTILINE)
    # parts = [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def run_sql_insights(clean: pd.DataFrame,
                     sql_file: Path | str = config.SQL_FILE) -> dict[str, pd.DataFrame]:
    """Load the cleaned data into in-memory SQLite and execute every named query."""
    queries = parse_queries(Path(sql_file).read_text(encoding="utf-8"))
    if not queries:
        raise ValueError(f"No '-- name:' query blocks found in {sql_file}")
    table = clean[_COLUMNS].copy()
    table["InvoiceDate"] = table["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(":memory:") as conn:
        table.to_sql("transactions", conn, index=False)
        return {name: pd.read_sql_query(sql, conn) for name, sql in queries.items()}
