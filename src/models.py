"""Model training utilities for the SAE clinical trial ML pipeline.

Implements training functions for Logistic Regression, Random Forest,
XGBoost, and LightGBM with cross-validated hyperparameter search, as well
as SMOTE oversampling and joblib model persistence.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> LogisticRegression:
    """Train a Logistic Regression classifier with grid-searched regularisation.

    Args:
        X_train: Training feature matrix (already preprocessed / numeric).
        y_train: Binary target vector.
        cv: Number of cross-validation folds for hyperparameter search.

    Returns:
        Fitted ``LogisticRegression`` estimator with the best ``C`` value.
    """
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    base = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info(
        "LogisticRegression best params: %s  |  CV AUC: %.4f",
        grid.best_params_,
        grid.best_score_,
    )
    return grid.best_estimator_


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with grid-searched hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Binary target vector.
        cv: Number of cross-validation folds.

    Returns:
        Fitted ``RandomForestClassifier`` with the best hyperparameters.
    """
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
    }
    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info(
        "RandomForest best params: %s  |  CV AUC: %.4f",
        grid.best_params_,
        grid.best_score_,
    )
    return grid.best_estimator_


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> Any:
    """Train an XGBoost classifier with grid-searched hyperparameters.

    The ``scale_pos_weight`` parameter is set automatically from the class
    ratio to handle class imbalance.

    Args:
        X_train: Training feature matrix.
        y_train: Binary target vector.
        cv: Number of cross-validation folds.

    Returns:
        Fitted ``XGBClassifier`` with the best hyperparameters.
    """
    from xgboost import XGBClassifier

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info(
        "XGBoost best params: %s  |  CV AUC: %.4f",
        grid.best_params_,
        grid.best_score_,
    )
    return grid.best_estimator_


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> Any:
    """Train a LightGBM classifier with grid-searched hyperparameters.

    ``is_unbalance=True`` is set to compensate for class imbalance.

    Args:
        X_train: Training feature matrix.
        y_train: Binary target vector.
        cv: Number of cross-validation folds.

    Returns:
        Fitted ``LGBMClassifier`` with the best hyperparameters.
    """
    from lightgbm import LGBMClassifier

    param_grid = {
        "n_estimators": [100, 200],
        "num_leaves": [31, 63],
        "learning_rate": [0.05, 0.1],
    }
    base = LGBMClassifier(
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    grid = GridSearchCV(base, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    logger.info(
        "LightGBM best params: %s  |  CV AUC: %.4f",
        grid.best_params_,
        grid.best_score_,
    )
    return grid.best_estimator_


# ---------------------------------------------------------------------------
# SMOTE oversampling
# ---------------------------------------------------------------------------


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to balance classes in the training set.

    Args:
        X_train: Training feature matrix (numeric numpy array).
        y_train: Binary target vector.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled) with balanced class distribution.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(
        "SMOTE: %d → %d samples | class distribution: %s",
        len(y_train),
        len(y_res),
        dict(zip(*np.unique(y_res, return_counts=True))),
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------


def save_model(model: Any, path: str) -> None:
    """Persist a fitted model to disk using joblib.

    Args:
        model: Any fitted scikit-learn compatible estimator.
        path: Destination file path (e.g. ``'models/rf.joblib'``).
    """
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def load_model(path: str) -> Any:
    """Load a persisted model from disk.

    Args:
        path: Path to a joblib file previously created by :func:`save_model`.

    Returns:
        The deserialised estimator object.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> dict[str, Any]:
    """Train all four models and return them in a dictionary.

    Args:
        X_train: Training feature matrix.
        y_train: Binary target vector.
        cv: Cross-validation folds passed to each trainer.

    Returns:
        Dictionary mapping model names to fitted estimators:
        ``{'logistic_regression': ..., 'random_forest': ...,
        'xgboost': ..., 'lightgbm': ...}``.
    """
    models: dict[str, Any] = {}

    logger.info("Training Logistic Regression …")
    models["logistic_regression"] = train_logistic_regression(X_train, y_train, cv=cv)

    logger.info("Training Random Forest …")
    models["random_forest"] = train_random_forest(X_train, y_train, cv=cv)

    logger.info("Training XGBoost …")
    models["xgboost"] = train_xgboost(X_train, y_train, cv=cv)

    logger.info("Training LightGBM …")
    models["lightgbm"] = train_lightgbm(X_train, y_train, cv=cv)

    logger.info("All models trained successfully.")
    return models
