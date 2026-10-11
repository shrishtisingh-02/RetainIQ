"""Monthly cohort retention analysis and monthly revenue summaries."""
from __future__ import annotations

import pandas as pd


def monthly_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """Revenue, orders and active customers per calendar month."""
    month = clean["InvoiceDate"].dt.to_period("M")
    summary = clean.groupby(month).agg(
        revenue=("Revenue", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
    )
    summary.index = summary.index.astype(str)
    summary.index.name = "month"
    return summary.reset_index()


def cohort_retention(clean: pd.DataFrame) -> pd.DataFrame:
    """Share of each acquisition cohort that purchased again N months later.

    Rows = acquisition month, columns = months since first purchase (0, 1, 2, ...).
    """
    if clean.empty:
        return pd.DataFrame()
    df = clean[["CustomerID", "InvoiceDate"]].copy()
    df["order_month"] = df["InvoiceDate"].dt.to_period("M")
    df["cohort_month"] = df.groupby("CustomerID")["order_month"].transform("min")
    df["period"] = (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)
    counts = (
        df.groupby(["cohort_month", "period"])["CustomerID"].nunique().unstack(fill_value=0)
    )
    sizes = counts[0]
    retention = counts.div(sizes, axis=0)
    # Blank out cells that are in the future relative to the last observed month.
    last_month = df["order_month"].max()
    for cohort in retention.index:
        max_period = (last_month - cohort).n
        retention.loc[cohort, retention.columns > max_period] = float("nan")
    retention.index = retention.index.astype(str)
    retention.index.name = "cohort_month"
    return retention
