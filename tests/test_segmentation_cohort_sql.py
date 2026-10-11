import numpy as np
import pandas as pd

from retainiq import config
from retainiq.cohort import cohort_retention, monthly_summary
from retainiq.features import build_customer_features
from retainiq.segmentation import (
    ACTION_MATRIX, assign_rfm_segment, build_scores, rfm_scores, risk_bucket, value_tier,
)
from retainiq.sql_analysis import parse_queries, run_sql_insights


def test_every_rfm_combination_gets_a_segment():
    names = {assign_rfm_segment(r, f) for r in range(1, 6) for f in range(1, 6)}
    assert "Champions" in names and "Hibernating" in names
    assert assign_rfm_segment(5, 5) == "Champions"
    assert assign_rfm_segment(1, 5) == "Can't Lose"
    assert assign_rfm_segment(1, 1) == "Hibernating"


def test_action_matrix_covers_all_tier_risk_pairs():
    for tier in ("High", "Mid", "Low"):
        for risk in ("High", "Medium", "Low"):
            assert (tier, risk) in ACTION_MATRIX


def test_value_tier_proportions_and_risk_buckets():
    s = pd.Series(np.arange(100, dtype=float))
    tiers = value_tier(s)
    assert (tiers == "High").sum() == 20 and (tiers == "Mid").sum() == 30
    assert (tiers == "Low").sum() == 50
    p = pd.Series([0.1, 0.3, 0.59, 0.6, 0.95])
    assert list(risk_bucket(p)) == ["Low", "Medium", "Medium", "High", "High"]
    assert config.HIGH_RISK_THRESHOLD == 0.6


def test_rfm_scores_direction(clean_df):
    feats = build_customer_features(clean_df, clean_df["InvoiceDate"].max())
    scores = rfm_scores(feats)
    assert scores.min().min() >= 1 and scores.max().max() <= 5
    most_recent = feats["recency_days"].idxmin()
    assert scores.loc[most_recent, "R_score"] == 5   # lowest recency_days = best score


def test_build_scores_columns(clean_df):
    feats = build_customer_features(clean_df, clean_df["InvoiceDate"].max())
    n = len(feats)
    out = build_scores(feats, np.linspace(0, 500, n), np.linspace(0, 1, n))
    for col in ("rfm_segment", "value_tier", "risk_bucket", "recommended_action"):
        assert out[col].notna().all()


def test_monthly_summary_matches_total_revenue(clean_df):
    m = monthly_summary(clean_df)
    assert m["revenue"].sum() == np.float64(clean_df["Revenue"].sum()) or np.isclose(
        m["revenue"].sum(), clean_df["Revenue"].sum())


def test_cohort_retention_properties(clean_df):
    c = cohort_retention(clean_df)
    assert (c[c.columns[0]].dropna() == 1.0).all()          # month 0 is 100% by definition
    values = c.to_numpy().ravel()
    values = values[~np.isnan(values)]
    assert len(values) > 0 and ((values >= 0) & (values <= 1)).all()


def test_sql_queries_agree_with_pandas(clean_df):
    results = run_sql_insights(clean_df)
    assert set(results) >= {"monthly_revenue", "top_countries", "top_products",
                            "repeat_purchase_rate", "revenue_concentration"}
    assert np.isclose(results["monthly_revenue"]["revenue"].sum(), clean_df["Revenue"].sum(),
                      rtol=1e-3)
    orders = clean_df.groupby("CustomerID")["InvoiceNo"].nunique()
    expected_repeat = (orders > 1).sum()
    assert results["repeat_purchase_rate"].loc[0, "repeat_customers"] == expected_repeat
    assert np.isclose(results["revenue_concentration"]["revenue_share_pct"].sum(), 100, atol=0.1)


def test_parse_queries():
    q = parse_queries("-- header\n-- name: a\nSELECT 1;\n\n-- name: b\nSELECT 2;")
    assert q == {"a": "SELECT 1;", "b": "SELECT 2;"}
