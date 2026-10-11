"""End-to-end pipeline: clean -> features -> train -> score -> segment -> export."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .cleaning import clean_transactions
from .cohort import cohort_retention, monthly_summary
from .features import FEATURE_COLUMNS, build_customer_features, build_training_table
from .modeling import (
    predict_revenue, refit_on_all, train_activity_model, train_revenue_model,
)
from .segmentation import build_scores
from .sql_analysis import run_sql_insights


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    return obj


def run_pipeline(raw: pd.DataFrame, out_dir: Path | str,
                 horizon_days: int = config.DEFAULT_HORIZON_DAYS, n_iter: int = 30,
                 cv: int = 5, source_label: str = "real") -> dict:
    """Run the full workflow on a raw transactions dataframe and write all outputs.

    Time-based protocol (no leakage):
      * train table : features at snapshot - 2*horizon, targets over the next `horizon` days
      * test table  : features at snapshot - 1*horizon, targets over the final `horizon` days
      * scoring     : features at snapshot (latest date); predictions cover the NEXT horizon
    """
    out_dir = Path(out_dir)
    (out_dir / "sql_insights").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    clean, cleaning_report = clean_transactions(raw)
    if clean.empty:
        raise ValueError("No valid transactions remain after cleaning.")

    snapshot = clean["InvoiceDate"].max()
    horizon = pd.Timedelta(days=horizon_days)
    train_cutoff, test_cutoff = snapshot - 2 * horizon, snapshot - horizon
    history_days = (train_cutoff - clean["InvoiceDate"].min()).days
    if history_days < config.MIN_HISTORY_DAYS:
        raise ValueError(
            f"Only {history_days} days of history before the training cutoff "
            f"(need >= {config.MIN_HISTORY_DAYS}). Reduce --horizon-days or use more data."
        )

    train = build_training_table(clean, train_cutoff, horizon_days)
    test = build_training_table(clean, test_cutoff, horizon_days)

    revenue = train_revenue_model(train, test, n_iter=n_iter, cv=cv, horizon_days=horizon_days)
    activity = train_activity_model(train, test, n_iter=max(10, n_iter // 2), cv=cv)

    # Refit on both windows, then score every customer as of the latest date.
    both = pd.concat([train, test])
    revenue_model = refit_on_all(revenue["model"], both, "future_revenue", log=True)
    activity_model = refit_on_all(activity["model"], both, "is_active_future", log=False)

    snapshot_feats = build_customer_features(clean, snapshot)
    X = snapshot_feats[FEATURE_COLUMNS]
    predicted_revenue = predict_revenue(revenue_model, X)
    churn_probability = 1.0 - activity_model.predict_proba(X)[:, 1]
    scores = build_scores(snapshot_feats, predicted_revenue, churn_probability)

    # ---- exports ----------------------------------------------------------
    scores.reset_index().to_csv(out_dir / "customer_scores.csv", index=False)
    revenue["feature_importance"].to_csv(out_dir / "feature_importance.csv", index=False)
    monthly = monthly_summary(clean)
    monthly.to_csv(out_dir / "monthly_revenue.csv", index=False)
    cohort_retention(clean).to_csv(out_dir / "cohort_retention.csv")
    for name, frame in run_sql_insights(clean).items():
        frame.to_csv(out_dir / "sql_insights" / f"{name}.csv", index=False)
    joblib.dump(revenue_model, out_dir / "models" / "revenue_model.joblib")
    joblib.dump(activity_model, out_dir / "models" / "activity_model.joblib")

    metrics = {
        "revenue_model": revenue["metrics"],
        "revenue_best_params": revenue["best_params"],
        "revenue_cv_best_rmse_log_scale": revenue["cv_best_rmse_log_scale"],
        "revenue_search": revenue["search_settings"],
        "activity_model": activity["metrics"],
        "activity_cv_best_roc_auc": activity["cv_best_roc_auc"],
        "train_customers": int(len(train)),
        "test_customers": int(len(test)),
    }
    run_info = {
        "source": source_label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "horizon_days": horizon_days,
        "snapshot_date": str(snapshot),
        "train_cutoff": str(train_cutoff),
        "test_cutoff": str(test_cutoff),
        "customers_scored": int(len(scores)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2))
    (out_dir / "cleaning_report.json").write_text(json.dumps(cleaning_report, indent=2))
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    return {"metrics": metrics, "run_info": run_info, "cleaning_report": cleaning_report,
            "scores": scores}
