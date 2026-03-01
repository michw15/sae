"""pytest tests for src/evaluation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


from src.evaluation import compare_models, compute_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def binary_predictions():
    """Return a simple set of true labels and predicted probabilities."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    # Slightly informative predictions
    y_pred_proba = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, 200), 0.0, 1.0)
    return y_true, y_pred_proba


@pytest.fixture()
def simple_models():
    """Return a dict with one trained LogisticRegression model."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        random_state=42,
        weights=[0.8, 0.2],
    )
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)
    return {"logistic_regression": model}, X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_metrics_valid_range(binary_predictions) -> None:
    """All returned metric values should be in [0, 1]."""
    y_true, y_pred_proba = binary_predictions

    metrics = compute_metrics(y_true, y_pred_proba)

    for key, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"Metric {key} = {value} out of [0, 1]"


def test_optimal_threshold_youden(binary_predictions) -> None:
    """The Youden-optimal threshold should lie strictly between 0 and 1."""
    y_true, y_pred_proba = binary_predictions

    metrics = compute_metrics(y_true, y_pred_proba)

    assert 0.0 < metrics["optimal_threshold"] < 1.0


def test_compute_metrics_returns_expected_keys(binary_predictions) -> None:
    """compute_metrics should return all expected metric keys."""
    y_true, y_pred_proba = binary_predictions

    metrics = compute_metrics(y_true, y_pred_proba)

    expected_keys = {
        "auc_roc", "auc_pr", "sensitivity", "specificity",
        "ppv", "npv", "brier_score", "f1", "optimal_threshold",
    }
    assert expected_keys.issubset(metrics.keys())


def test_compare_models_returns_dataframe(simple_models) -> None:
    """compare_models should return a DataFrame with a column per metric."""
    models_dict, X, y = simple_models
    y_series = pd.Series(y)

    result = compare_models(models_dict, X, y_series)

    assert isinstance(result, pd.DataFrame)
    assert "auc_roc" in result.columns
    assert len(result) == len(models_dict)
