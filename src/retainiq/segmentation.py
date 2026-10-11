"""RFM segmentation and retention decision rules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _quintile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Score 1..5 using rank-based quintiles (robust to many tied values)."""
    ranks = series.rank(method="first", ascending=ascending)
    n_bins = min(5, max(1, series.nunique()))
    return (pd.qcut(ranks, n_bins, labels=False) + 1).astype(int)


def rfm_scores(feats: pd.DataFrame) -> pd.DataFrame:
    """Add R, F, M scores (5 = best). Low recency_days is good, so it is ranked reversed."""
    out = pd.DataFrame(index=feats.index)
    out["R_score"] = _quintile_score(feats["recency_days"], ascending=False)
    out["F_score"] = _quintile_score(feats["frequency"], ascending=True)
    out["M_score"] = _quintile_score(feats["monetary"], ascending=True)
    return out


def assign_rfm_segment(r: int, f: int) -> str:
    """Map recency / frequency scores to a named segment."""
    if r >= 4 and f >= 4:
        return "Champions"
    if r <= 2 and f >= 4:
        return "Can't Lose"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r <= 2 and f == 3:
        return "At Risk"
    if r >= 4 and f <= 2:
        return "New / Promising"
    if r <= 2:
        return "Hibernating"
    return "Needs Attention"


def value_tier(predicted_revenue: pd.Series) -> pd.Series:
    """High = top 20% by predicted revenue, Mid = next 30%, Low = the rest."""
    pct = predicted_revenue.rank(method="first", ascending=False, pct=True)
    return pd.Series(
        np.select(
            [pct <= config.HIGH_VALUE_TOP_PCT, pct <= config.MID_VALUE_TOP_PCT],
            ["High", "Mid"], default="Low",
        ),
        index=predicted_revenue.index,
    )


def risk_bucket(churn_probability: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [churn_probability >= config.HIGH_RISK_THRESHOLD,
             churn_probability >= config.MEDIUM_RISK_THRESHOLD],
            ["High", "Medium"], default="Low",
        ),
        index=churn_probability.index,
    )


ACTION_MATRIX = {
    ("High", "High"): "Priority win-back: personal outreach with an incentive",
    ("High", "Medium"): "Proactive loyalty offer before they lapse",
    ("High", "Low"): "Reward and upsell: VIP perks, early access",
    ("Mid", "High"): "Targeted reactivation email campaign",
    ("Mid", "Medium"): "Nurture: personalised product recommendations",
    ("Mid", "Low"): "Cross-sell related products",
    ("Low", "High"): "Low-cost automated win-back only",
    ("Low", "Medium"): "Include in regular newsletter",
    ("Low", "Low"): "Maintain; no extra retention spend",
}


def recommend_action(tier: str, risk: str) -> str:
    return ACTION_MATRIX[(tier, risk)]


def build_scores(feats: pd.DataFrame, predicted_revenue: np.ndarray,
                 churn_probability: np.ndarray) -> pd.DataFrame:
    """Combine features, model outputs, segments and recommended actions."""
    out = feats.copy()
    out["predicted_revenue"] = np.asarray(predicted_revenue, dtype=float)
    out["churn_probability"] = np.asarray(churn_probability, dtype=float)
    scores = rfm_scores(feats)
    out = out.join(scores)
    out["rfm_segment"] = [assign_rfm_segment(r, f)
                          for r, f in zip(out["R_score"], out["F_score"])]
    out["value_tier"] = value_tier(out["predicted_revenue"])
    out["risk_bucket"] = risk_bucket(out["churn_probability"])
    out["recommended_action"] = [recommend_action(t, r)
                                 for t, r in zip(out["value_tier"], out["risk_bucket"])]
    return out
