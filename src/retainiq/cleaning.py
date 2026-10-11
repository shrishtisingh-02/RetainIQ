"""Data validation and cleaning for raw retail transactions."""
from __future__ import annotations

import pandas as pd

from . import config


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean raw transactions.

    Returns the cleaned dataframe (with a `Revenue` column) and a report dict that
    states how many rows each step removed, so the cleaning is auditable.
    """
    report: dict = {"raw_rows": int(len(raw))}
    df = raw.copy()

    # --- type coercion -----------------------------------------------------
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    if "Country" not in df.columns:
        df["Country"] = "Unknown"
    if "Description" not in df.columns:
        df["Description"] = ""

    def drop(mask: pd.Series, key: str) -> None:
        nonlocal df
        report[key] = int(mask.sum())
        df = df.loc[~mask].copy()

    drop(df["InvoiceDate"].isna() | df["Quantity"].isna() | df["UnitPrice"].isna(),
         "removed_unparseable_values")
    drop(df["CustomerID"].isna(), "removed_missing_customer_id")
    drop(df["InvoiceNo"].str.upper().str.startswith("C"), "removed_cancellations")
    drop((df["Quantity"] <= 0) | (df["UnitPrice"] <= 0), "removed_non_positive_qty_or_price")
    drop(df["StockCode"].isin(config.NON_PRODUCT_CODES), "removed_non_product_codes")
    drop(df.duplicated(), "removed_exact_duplicates")

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df = df.sort_values("InvoiceDate").reset_index(drop=True)

    report["clean_rows"] = int(len(df))
    report["clean_customers"] = int(df["CustomerID"].nunique())
    report["clean_invoices"] = int(df["InvoiceNo"].nunique())
    if len(df):
        report["date_min"] = str(df["InvoiceDate"].min())
        report["date_max"] = str(df["InvoiceDate"].max())
    return df, report
