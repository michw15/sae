"""Model evaluation utilities for the SAE clinical trial ML pipeline.

Provides metric computation, comparison tables, and publication-quality
diagnostic plots (ROC, PR, confusion matrix, calibration).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: Optional[float] = None,
) -> dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    The optimal threshold is determined via Youden's J statistic
    (``sensitivity + specificity − 1`` maximised) unless *threshold* is
    supplied explicitly.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_pred_proba: Predicted probabilities for the positive class.
        threshold: Decision threshold.  If *None* the Youden-optimal
            threshold is used.

    Returns:
        Dictionary containing: ``auc_roc``, ``auc_pr``, ``sensitivity``,
        ``specificity``, ``ppv``, ``npv``, ``brier_score``, ``f1``,
        ``optimal_threshold``.
    """
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)

    # Youden-optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[youden_idx])

    if threshold is None:
        threshold = optimal_threshold

    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "brier_score": float(brier),
        "f1": float(f1),
        "optimal_threshold": float(optimal_threshold),
    }

    logger.info("Metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(path) if "." in os.path.basename(path) else path, exist_ok=True)


def plot_roc_curves(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """Plot ROC curves for multiple models on a single axes.

    Args:
        models_dict: Mapping of model name → fitted estimator.
        X_test: Test feature matrix.
        y_test: True binary labels.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
        logger.info("ROC curve saved to %s", save_path)
    plt.close(fig)


def plot_pr_curves(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """Plot Precision-Recall curves for multiple models.

    Args:
        models_dict: Mapping of model name → fitted estimator.
        X_test: Test feature matrix.
        y_test: True binary labels.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = y_test.mean()

    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    ax.axhline(baseline, color="k", linestyle="--", linewidth=0.8, label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
        logger.info("PR curve saved to %s", save_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Plot and optionally save a confusion-matrix heatmap.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        model_name: Name used in the plot title.
        save_path: If provided, save the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No SAE", "SAE"],
        yticklabels=["No SAE", "SAE"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
        logger.info("Confusion matrix saved to %s", save_path)
    plt.close(fig)


def plot_calibration_curve(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """Plot a calibration reliability diagram for multiple models.

    Args:
        models_dict: Mapping of model name → fitted estimator.
        X_test: Test feature matrix.
        y_test: True binary labels.
        save_path: If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        fraction_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_pred, fraction_pos, marker="o", label=name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Reliability Diagram")
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
        logger.info("Calibration curve saved to %s", save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_models(
    models_dict: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame.

    Args:
        models_dict: Mapping of model name → fitted estimator.
        X_test: Test feature matrix.
        y_test: True binary labels.

    Returns:
        DataFrame with one row per model and columns for each metric.
    """
    rows = []
    for name, model in models_dict.items():
        proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(np.array(y_test), proba)
        metrics["model"] = name
        rows.append(metrics)

    df_cmp = pd.DataFrame(rows).set_index("model")
    df_cmp = df_cmp.sort_values("auc_roc", ascending=False)

    print("\n=== Model Comparison ===")
    print(df_cmp.to_string(float_format="{:.4f}".format))

    return df_cmp
