"""Customer-level feature engineering with strict time-based leakage control.

Features are computed ONLY from transactions on or before a `cutoff` timestamp.
Targets are computed ONLY from transactions in (cutoff, cutoff + horizon].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_order_value",
    "tenure_days",
    "purchase_velocity",
    "avg_days_between_orders",
    "product_breadth",
    "items_per_order",
    "revenue_last_30d",
    "revenue_last_90d",
    "revenue_last_180d",
    "orders_last_90d",
]


def build_customer_features(clean: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per customer who purchased on/before `cutoff`, indexed by CustomerID."""
    cutoff = pd.Timestamp(cutoff)
    hist = clean[clean["InvoiceDate"] <= cutoff]
    if hist.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["country"]).rename_axis("CustomerID")

    invoices = (
        hist.groupby(["CustomerID", "InvoiceNo"])
        .agg(InvoiceDate=("InvoiceDate", "min"), invoice_value=("Revenue", "sum"))
        .reset_index()
        .sort_values(["CustomerID", "InvoiceDate"])
    )
    per_customer = invoices.groupby("CustomerID")
    feats = per_customer.agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("invoice_value", "sum"),
        avg_order_value=("invoice_value", "mean"),
    )
    feats["recency_days"] = (cutoff - feats["last_purchase"]).dt.total_seconds() / 86400
    feats["tenure_days"] = (cutoff - feats["first_purchase"]).dt.total_seconds() / 86400
    feats["purchase_velocity"] = feats["frequency"] / (feats["tenure_days"] / 30).clip(lower=1)

    # Mean gap between consecutive orders (NaN for one-order customers; tree models handle NaN).
    gaps = invoices.groupby("CustomerID")["InvoiceDate"].diff().dt.total_seconds() / 86400
    feats["avg_days_between_orders"] = gaps.groupby(invoices["CustomerID"]).mean()

    feats["product_breadth"] = hist.groupby("CustomerID")["StockCode"].nunique()
    feats["items_per_order"] = hist.groupby("CustomerID")["Quantity"].sum() / feats["frequency"]

    for window in (30, 90, 180):
        recent = invoices[invoices["InvoiceDate"] > cutoff - pd.Timedelta(days=window)]
        feats[f"revenue_last_{window}d"] = (
            recent.groupby("CustomerID")["invoice_value"].sum().reindex(feats.index).fillna(0.0)
        )
    recent90 = invoices[invoices["InvoiceDate"] > cutoff - pd.Timedelta(days=90)]
    feats["orders_last_90d"] = (
        recent90.groupby("CustomerID")["InvoiceNo"].nunique().reindex(feats.index).fillna(0)
    )

    feats["country"] = hist.groupby("CustomerID")["Country"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"
    )
    return feats[FEATURE_COLUMNS + ["country"]]


def add_targets(feats: pd.DataFrame, clean: pd.DataFrame, cutoff: pd.Timestamp,
                horizon_days: int) -> pd.DataFrame:
    """Attach `future_revenue` and `is_active_future` for the window after `cutoff`."""
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    future = clean[(clean["InvoiceDate"] > cutoff) & (clean["InvoiceDate"] <= end)]
    revenue = future.groupby("CustomerID")["Revenue"].sum()
    out = feats.copy()
    out["future_revenue"] = revenue.reindex(out.index).fillna(0.0)
    out["is_active_future"] = (out["future_revenue"] > 0).astype(int)
    return out


def build_training_table(clean: pd.DataFrame, cutoff: pd.Timestamp,
                         horizon_days: int) -> pd.DataFrame:
    """Features as of `cutoff` + targets over the following `horizon_days`."""
    return add_targets(build_customer_features(clean, cutoff), clean, cutoff, horizon_days)


def log_target(y: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype=float))
