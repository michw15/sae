"""Explainability utilities for the SAE clinical trial ML pipeline.

Provides SHAP-based global and local explanations as well as LIME-based
individual patient explanations and feature-importance plots for tree-based
models.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they do not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "tree",
) -> Any:
    """Compute SHAP values for *model* on *X*.

    Args:
        model: Fitted scikit-learn–compatible estimator.
        X: Feature matrix (DataFrame).
        model_type: ``'tree'`` uses ``shap.TreeExplainer``; any other string
            falls back to ``shap.KernelExplainer`` on a 50-sample background.

    Returns:
        SHAP values array (or ``shap.Explanation`` object for tree models).
    """
    import shap

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
    else:
        background = shap.sample(X, min(50, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)

    logger.info("SHAP values computed (model_type=%s, n=%d)", model_type, len(X))
    return shap_values


def plot_shap_summary(
    shap_values: Any,
    X: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot a SHAP beeswarm summary plot.

    Args:
        shap_values: SHAP values from :func:`compute_shap_values`.
        X: Feature matrix used to compute the SHAP values.
        save_path: If provided, save the figure to this path.
    """
    import shap

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot saved to %s", save_path)
    plt.close("all")


def plot_shap_bar(
    shap_values: Any,
    X: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """Plot a SHAP global feature-importance bar chart.

    Args:
        shap_values: SHAP values from :func:`compute_shap_values`.
        X: Feature matrix used to compute the SHAP values.
        save_path: If provided, save the figure to this path.
    """
    import shap

    shap.plots.bar(shap_values, show=False, max_display=20)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP bar plot saved to %s", save_path)
    plt.close("all")


def plot_shap_waterfall(
    shap_values: Any,
    X: pd.DataFrame,
    idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """Plot a SHAP waterfall chart for a single patient.

    Args:
        shap_values: SHAP values from :func:`compute_shap_values`.
        X: Feature matrix used to compute the SHAP values.
        idx: Row index of the patient to explain.
        save_path: If provided, save the figure to this path.
    """
    import shap

    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP waterfall plot saved to %s (idx=%d)", save_path, idx)
    plt.close("all")


def plot_shap_dependence(
    shap_values: Any,
    X: pd.DataFrame,
    feature: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot a SHAP dependence scatter plot for a single feature.

    Args:
        shap_values: SHAP values from :func:`compute_shap_values`.
        X: Feature matrix used to compute the SHAP values.
        feature: Name of the feature to plot on the x-axis.
        save_path: If provided, save the figure to this path.
    """
    import shap

    shap.plots.scatter(shap_values[:, feature], show=False)
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP dependence plot saved to %s (feature=%s)", save_path, feature)
    plt.close("all")


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


def explain_with_lime(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    idx: int,
    feature_names: list[str],
    save_path: Optional[str] = None,
) -> Any:
    """Generate a LIME explanation for a single test instance.

    Args:
        model: Fitted estimator with a ``predict_proba`` method.
        X_train: Training data used to initialise the LIME explainer.
        X_test: Test data containing the instance to explain.
        idx: Row index in *X_test* to explain.
        feature_names: Ordered list of feature names.
        save_path: If provided, save the explanation figure to this path.

    Returns:
        The ``lime.explanation.Explanation`` object.
    """
    import lime
    import lime.lime_tabular

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=["No SAE", "SAE"],
        mode="classification",
        random_state=42,
    )

    instance = np.array(X_test)[idx]
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=10,
    )

    if save_path:
        _ensure_dir(save_path)
        fig = explanation.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("LIME explanation saved to %s (idx=%d)", save_path, idx)

    return explanation


# ---------------------------------------------------------------------------
# Tree feature importance
# ---------------------------------------------------------------------------


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot built-in feature importances for tree-based models.

    Args:
        model: Fitted tree-based estimator with a ``feature_importances_``
            attribute.
        feature_names: Ordered list of feature names matching the training
            columns.
        model_name: Title label for the plot.
        save_path: If provided, save the figure to this path.

    Raises:
        AttributeError: If *model* does not have ``feature_importances_``.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(indices)), importances[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(
        [feature_names[i] for i in indices], rotation=45, ha="right"
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance – {model_name}")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance plot saved to %s", save_path)
    plt.close(fig)
