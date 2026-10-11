import pandas as pd
import pytest

from retainiq.features import FEATURE_COLUMNS, add_targets, build_customer_features


def _tx(rows):
    df = pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "StockCode", "Quantity",
                                     "InvoiceDate", "Revenue", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_known_values_for_hand_computed_customer():
    tx = _tx([
        (1, "A", "S1", 2, "2011-01-01", 10.0, "UK"),
        (1, "A", "S2", 1, "2011-01-01", 5.0, "UK"),
        (1, "B", "S1", 3, "2011-01-11", 30.0, "UK"),
    ])
    cutoff = pd.Timestamp("2011-01-21")
    f = build_customer_features(tx, cutoff).loc[1]
    assert f["frequency"] == 2
    assert f["monetary"] == pytest.approx(45.0)
    assert f["avg_order_value"] == pytest.approx(22.5)
    assert f["recency_days"] == pytest.approx(10.0)
    assert f["tenure_days"] == pytest.approx(20.0)
    assert f["avg_days_between_orders"] == pytest.approx(10.0)
    assert f["product_breadth"] == 2
    assert f["items_per_order"] == pytest.approx(3.0)
    assert f["revenue_last_30d"] == pytest.approx(45.0)


def test_features_do_not_leak_future_data(clean_df):
    cutoff = clean_df["InvoiceDate"].max() - pd.Timedelta(days=120)
    before = build_customer_features(clean_df, cutoff)
    # Remove everything after the cutoff: features must be identical.
    truncated = clean_df[clean_df["InvoiceDate"] <= cutoff]
    after = build_customer_features(truncated, cutoff)
    pd.testing.assert_frame_equal(before, after)


def test_targets_only_use_window_after_cutoff():
    tx = _tx([
        (1, "A", "S1", 1, "2011-01-01", 10.0, "UK"),
        (1, "B", "S1", 1, "2011-02-01", 20.0, "UK"),   # inside window
        (1, "C", "S1", 1, "2011-06-01", 99.0, "UK"),   # beyond window
        (2, "D", "S1", 1, "2011-01-02", 7.0, "UK"),    # churned: nothing after cutoff
    ])
    cutoff = pd.Timestamp("2011-01-15")
    feats = build_customer_features(tx, cutoff)
    out = add_targets(feats, tx, cutoff, horizon_days=60)
    assert out.loc[1, "future_revenue"] == pytest.approx(20.0)
    assert out.loc[1, "is_active_future"] == 1
    assert out.loc[2, "future_revenue"] == 0.0
    assert out.loc[2, "is_active_future"] == 0


def test_empty_history_returns_empty_frame(clean_df):
    out = build_customer_features(clean_df, pd.Timestamp("2000-01-01"))
    assert out.empty
    assert set(FEATURE_COLUMNS) <= set(out.columns)
