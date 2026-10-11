"""Revenue regression and retention (activity) classification models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score, mean_absolute_error, r2_score,
    roc_auc_score, root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURE_COLUMNS, log_target

_PARAM_DISTRIBUTIONS = {
    "learning_rate": loguniform(0.02, 0.3),
    "max_iter": randint(100, 500),
    "max_depth": [None, 3, 4, 6],
    "max_leaf_nodes": randint(8, 48),
    "min_samples_leaf": randint(10, 60),
    "l2_regularization": loguniform(1e-3, 10),
}


def predict_revenue(model, X: pd.DataFrame, max_log: float | None = None) -> np.ndarray:
    """Models are trained on log1p(revenue); convert predictions back to currency.

    `max_log` optionally caps the log-scale prediction (used for the linear baseline,
    whose unbounded extrapolation would otherwise explode after exponentiation).
    """
    return np.expm1(np.clip(model.predict(X), 0, max_log))


def top_quantile_capture(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.2) -> float:
    """Share of actual revenue earned by the customers ranked in the top `q` by prediction."""
    y_true = np.asarray(y_true, dtype=float)
    total = y_true.sum()
    if total <= 0:
        return float("nan")
    n = max(1, int(np.ceil(len(y_true) * q)))
    idx = np.argsort(-np.asarray(y_pred))[:n]
    return float(y_true[idx].sum() / total)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "R2_log_scale": float(r2_score(np.log1p(y_true), np.log1p(np.clip(y_pred, 0, None)))),
        "top20pct_revenue_capture": top_quantile_capture(y_true, y_pred, 0.2),
    }


def _check_table(df: pd.DataFrame, name: str) -> None:
    if len(df) < config.MIN_CUSTOMERS:
        raise ValueError(
            f"{name} table has only {len(df)} customers (need >= {config.MIN_CUSTOMERS}). "
            "Use a larger dataset or a shorter horizon."
        )


def train_revenue_model(train: pd.DataFrame, test: pd.DataFrame, n_iter: int = 30,
                        cv: int = 5, random_state: int = config.RANDOM_STATE,
                        horizon_days: int = config.DEFAULT_HORIZON_DAYS) -> dict:
    """Tune a gradient-boosting regressor with RandomizedSearchCV and compare to baselines."""
    _check_table(train, "Training")
    _check_table(test, "Test")
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    y_train, y_test = train["future_revenue"].to_numpy(), test["future_revenue"].to_numpy()

    results: dict = {}

    # Baseline 1: predict the training-set mean for everyone.
    results["baseline_mean"] = regression_metrics(
        y_test, np.full(len(y_test), y_train.mean()))
    # Baseline 2: "persistence" -- assume the next window looks like the last 90 days.
    persistence = test["revenue_last_90d"].to_numpy() * (horizon_days / 90)
    results["baseline_persistence"] = regression_metrics(y_test, persistence)

    # Baseline 3: regularised linear model.
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(X_train, log_target(y_train))
    results["ridge"] = regression_metrics(
        y_test, predict_revenue(ridge, X_test, max_log=float(log_target(y_train).max())))

    # Main model: tuned histogram gradient boosting.
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=random_state),
        param_distributions=_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=KFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X_train, log_target(y_train))
    best = search.best_estimator_
    results["gradient_boosting_tuned"] = regression_metrics(y_test, predict_revenue(best, X_test))

    # Permutation importance on the held-out (later) window.
    imp = permutation_importance(best, X_test, log_target(y_test), n_repeats=10,
                                 random_state=random_state, scoring="neg_root_mean_squared_error")
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return {
        "model": best,
        "metrics": results,
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "cv_best_rmse_log_scale": float(-search.best_score_),
        "search_settings": {"n_iter": n_iter, "cv_folds": cv},
        "feature_importance": importance,
    }


def train_activity_model(train: pd.DataFrame, test: pd.DataFrame, n_iter: int = 20,
                         cv: int = 5, random_state: int = config.RANDOM_STATE) -> dict:
    """Classifier for P(customer purchases again within the horizon)."""
    _check_table(train, "Training")
    _check_table(test, "Test")
    y_train, y_test = train["is_active_future"].to_numpy(), test["is_active_future"].to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training target has a single class; cannot train a classifier.")

    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=random_state),
        param_distributions=_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(train[FEATURE_COLUMNS], y_train)
    best = search.best_estimator_
    proba = best.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    metrics = {
        "base_rate_active": float(y_test.mean()),
        "accuracy_at_0.5": float(((proba >= 0.5).astype(int) == y_test).mean()),
        "pr_auc": float(average_precision_score(y_test, proba)) if y_test.sum() else float("nan"),
    }
    metrics["roc_auc"] = (float(roc_auc_score(y_test, proba))
                          if len(np.unique(y_test)) == 2 else float("nan"))
    return {
        "model": best,
        "metrics": metrics,
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "cv_best_roc_auc": float(search.best_score_),
    }


def refit_on_all(model, table: pd.DataFrame, target: str, log: bool):
    """Refit a tuned estimator (same hyper-parameters) on a combined table."""
    fitted = clone(model)
    y = log_target(table[target]) if log else table[target].to_numpy()
    fitted.fit(table[FEATURE_COLUMNS], y)
    return fitted
