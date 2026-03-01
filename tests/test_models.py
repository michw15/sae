"""pytest tests for src/models.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models import (
    apply_smote,
    load_model,
    save_model,
    train_logistic_regression,
    train_random_forest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_dataset():
    """Return a small imbalanced classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        weights=[0.85, 0.15],
        random_state=42,
    )
    return X.astype(np.float32), y.astype(int)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_logistic_regression_trains_and_predicts(small_dataset) -> None:
    """Logistic Regression should train and return probabilities of correct shape."""
    X, y = small_dataset

    model = train_logistic_regression(X, y, cv=2)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_random_forest_trains(small_dataset) -> None:
    """Random Forest should fit without error and produce valid predictions."""
    X, y = small_dataset

    model = train_random_forest(X, y, cv=2)

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_smote_increases_minority_class(small_dataset) -> None:
    """After SMOTE, both classes should have equal counts."""
    X, y = small_dataset

    X_res, y_res = apply_smote(X, y, random_state=42)

    counts = np.bincount(y_res)
    # Both classes should be equal after SMOTE
    assert counts[0] == counts[1]
    assert len(y_res) > len(y)


def test_save_and_load_model(small_dataset, tmp_path) -> None:
    """save_model / load_model should roundtrip a fitted estimator."""
    X, y = small_dataset
    model = train_logistic_regression(X, y, cv=2)

    model_path = str(tmp_path / "models" / "test_lr.joblib")
    save_model(model, model_path)

    loaded = load_model(model_path)

    # Predictions should be identical
    np.testing.assert_array_equal(
        model.predict(X), loaded.predict(X)
    )
