"""Loading and validating the raw transaction file."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def find_raw_file(directory: Path | str = config.DATA_RAW) -> Path:
    """Return the first known raw data file inside `directory`."""
    directory = Path(directory)
    for name in config.RAW_CANDIDATES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No dataset found in '{directory}'. Download the UCI 'Online Retail' dataset "
        "(https://archive.ics.uci.edu/dataset/352/online+retail), unzip it and place "
        f"'Online Retail.xlsx' in that folder. Expected one of: {config.RAW_CANDIDATES}"
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column variants to the canonical names."""
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in config.COLUMN_ALIASES:
            renamed[col] = config.COLUMN_ALIASES[key]
    return df.rename(columns=renamed)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a clear error if the dataframe is empty or lacks required columns."""
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. Found columns: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("Dataset contains no rows.")


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw CSV/XLSX file and return a dataframe with canonical column names."""
    path = find_raw_file() if path is None else Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)  # the UCI file is ~540k rows; this can take a minute
    elif suffix == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .xlsx or .csv.")

    df = standardize_columns(df)
    validate_schema(df)
    return df
