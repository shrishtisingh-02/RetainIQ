import json

import pandas as pd
import pytest

from retainiq.pipeline import run_pipeline
from retainiq.synthetic import generate_synthetic_transactions


@pytest.fixture(scope="module")
def pipeline_out(tmp_path_factory, raw_df):
    out = tmp_path_factory.mktemp("outputs")
    result = run_pipeline(raw_df, out, horizon_days=90, n_iter=5, cv=3, source_label="synthetic")
    return out, result


def test_pipeline_writes_all_outputs(pipeline_out):
    out, _ = pipeline_out
    for name in ("customer_scores.csv", "feature_importance.csv", "monthly_revenue.csv",
                 "cohort_retention.csv", "metrics.json", "cleaning_report.json",
                 "run_info.json", "models/revenue_model.joblib", "models/activity_model.joblib",
                 "sql_insights/monthly_revenue.csv"):
        assert (out / name).exists(), name


def test_scores_are_valid(pipeline_out):
    out, _ = pipeline_out
    s = pd.read_csv(out / "customer_scores.csv")
    assert s["CustomerID"].is_unique
    assert s["predicted_revenue"].ge(0).all()
    assert s["churn_probability"].between(0, 1).all()
    assert s["recommended_action"].notna().all()


def test_metrics_json_and_model_beats_mean_baseline(pipeline_out):
    out, _ = pipeline_out
    m = json.loads((out / "metrics.json").read_text())
    rev = m["revenue_model"]
    assert {"baseline_mean", "baseline_persistence", "ridge",
            "gradient_boosting_tuned"} <= set(rev)
    assert rev["gradient_boosting_tuned"]["R2_log_scale"] > rev["baseline_mean"]["R2_log_scale"]
    assert 0.5 < m["activity_model"]["roc_auc"] <= 1.0


def test_run_info_marks_synthetic(pipeline_out):
    out, _ = pipeline_out
    assert json.loads((out / "run_info.json").read_text())["source"] == "synthetic"


def test_pipeline_rejects_insufficient_history(tmp_path, raw_df):
    with pytest.raises(ValueError, match="history"):
        run_pipeline(raw_df, tmp_path, horizon_days=200, n_iter=2, cv=3)


def test_pipeline_rejects_all_invalid_data(tmp_path):
    bad = generate_synthetic_transactions(n_customers=5, seed=1)
    bad["CustomerID"] = None
    with pytest.raises(ValueError, match="No valid transactions"):
        run_pipeline(bad, tmp_path, n_iter=2, cv=3)


def test_pipeline_rejects_tiny_dataset(tmp_path):
    tiny = generate_synthetic_transactions(n_customers=10, seed=3)
    with pytest.raises(ValueError):
        run_pipeline(tiny, tmp_path, n_iter=2, cv=3)
