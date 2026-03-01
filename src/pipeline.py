"""End-to-end CLI pipeline for SAE prediction.

Usage::

    python src/pipeline.py --data data/synthetic/clinical_trial_data.csv \\
                           --model all --output reports/ --explain

The pipeline:
    1. Loads configuration from ``configs/config.yaml``.
    2. Loads and preprocesses the data.
    3. Trains the requested model(s).
    4. Evaluates on the test set and saves diagnostic plots.
    5. Optionally generates SHAP / LIME explanations.
    6. Saves ``reports/model_comparison.csv`` and prints a summary table.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Ensure the project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    compare_models,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
)
from src.explainability import (
    compute_shap_values,
    explain_with_lime,
    plot_feature_importance,
    plot_shap_bar,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.models import (
    apply_smote,
    save_model,
    train_all_models,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.preprocessing import (
    build_preprocessing_pipeline,
    load_data,
    prepare_data,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(log_dir: str) -> None:
    """Configure root logger to write to console and a rotating log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAE Prediction ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/synthetic/clinical_trial_data.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["lr", "rf", "xgboost", "lgbm", "all"],
        help="Model(s) to train.",
    )
    parser.add_argument(
        "--output",
        default="reports/",
        help="Directory for output reports and figures.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE oversampling before training.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate SHAP and LIME explanations.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    if not os.path.exists(path):
        logger.warning("Config file not found at %s; using defaults.", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", path)
    return cfg or {}


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full SAE prediction pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    cfg = _load_config(args.config)

    output_dir = args.output
    figures_dir = cfg.get("evaluation", {}).get(
        "output_dir", os.path.join(output_dir, "figures/")
    )
    models_dir = cfg.get("output", {}).get("models_dir", "models/")
    log_dir = cfg.get("output", {}).get("logs_dir", "logs/")

    _setup_logging(log_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    logger.info("Loading data from %s …", args.data)
    df = load_data(args.data)

    data_cfg = cfg.get("data", {})
    target_col = data_cfg.get("target_col", "sae_occurred")
    test_size = data_cfg.get("test_size", 0.2)
    random_state = data_cfg.get("random_state", 42)

    X_train, X_test, y_train, y_test = prepare_data(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # 2. Preprocessing pipeline (fit on train, transform both splits)
    # ------------------------------------------------------------------
    feat_cfg = cfg.get("features", {})
    numeric_cols = feat_cfg.get(
        "numeric",
        X_train.select_dtypes(include=[np.number]).columns.tolist(),
    )
    # Filter to columns that actually exist after feature engineering
    numeric_cols = [c for c in numeric_cols if c in X_train.columns]
    categorical_cols = feat_cfg.get(
        "categorical",
        X_train.select_dtypes(include=["object", "category"]).columns.tolist(),
    )
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]

    # Binary columns: kept as-is (pass-through via ColumnTransformer remainder)
    all_handled = set(numeric_cols + categorical_cols)
    binary_cols = [c for c in X_train.columns if c not in all_handled]

    preproc_pipe = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_train_proc = preproc_pipe.fit_transform(X_train)
    X_test_proc = preproc_pipe.transform(X_test)

    # ------------------------------------------------------------------
    # 3. Optional SMOTE
    # ------------------------------------------------------------------
    if args.smote:
        logger.info("Applying SMOTE …")
        X_train_proc, y_train_arr = apply_smote(X_train_proc, np.array(y_train))
        y_train_fit = pd.Series(y_train_arr)
    else:
        y_train_fit = y_train

    # ------------------------------------------------------------------
    # 4. Train model(s)
    # ------------------------------------------------------------------
    TRAINER_MAP = {
        "lr": ("logistic_regression", train_logistic_regression),
        "rf": ("random_forest", train_random_forest),
        "xgboost": ("xgboost", train_xgboost),
        "lgbm": ("lightgbm", train_lightgbm),
    }

    trained_models: dict[str, Any] = {}

    if args.model == "all":
        trained_models = train_all_models(X_train_proc, y_train_fit)
    else:
        name, trainer = TRAINER_MAP[args.model]
        trained_models[name] = trainer(X_train_proc, y_train_fit)

    # Persist models
    for model_name, model in trained_models.items():
        save_model(model, os.path.join(models_dir, f"{model_name}.joblib"))

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    logger.info("Evaluating models …")
    comparison_df = compare_models(trained_models, X_test_proc, y_test)

    csv_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(csv_path)
    logger.info("Model comparison saved to %s", csv_path)

    # Diagnostic plots
    plot_roc_curves(
        trained_models,
        X_test_proc,
        y_test,
        save_path=os.path.join(figures_dir, "roc_curves.png"),
    )
    plot_pr_curves(
        trained_models,
        X_test_proc,
        y_test,
        save_path=os.path.join(figures_dir, "pr_curves.png"),
    )
    plot_calibration_curve(
        trained_models,
        X_test_proc,
        y_test,
        save_path=os.path.join(figures_dir, "calibration_curve.png"),
    )

    for model_name, model in trained_models.items():
        proba = model.predict_proba(X_test_proc)[:, 1]
        from src.evaluation import compute_metrics

        metrics = compute_metrics(np.array(y_test), proba)
        y_pred = (proba >= metrics["optimal_threshold"]).astype(int)
        plot_confusion_matrix(
            np.array(y_test),
            y_pred,
            model_name=model_name,
            save_path=os.path.join(figures_dir, f"confusion_{model_name}.png"),
        )

    # ------------------------------------------------------------------
    # 6. Explainability
    # ------------------------------------------------------------------
    if args.explain:
        logger.info("Generating explainability plots …")
        # Use best model (highest AUC-ROC)
        best_name = comparison_df["auc_roc"].idxmax()
        best_model = trained_models[best_name]
        logger.info("Explaining model: %s", best_name)

        shap_cfg = cfg.get("explainability", {})
        sample_size = shap_cfg.get("shap_sample_size", 200)

        X_shap = X_test_proc[:sample_size]
        model_type = "tree" if best_name in ("random_forest", "xgboost", "lightgbm") else "kernel"

        try:
            shap_vals = compute_shap_values(best_model, pd.DataFrame(X_shap), model_type=model_type)
            plot_shap_summary(
                shap_vals, pd.DataFrame(X_shap),
                save_path=os.path.join(figures_dir, f"shap_summary_{best_name}.png"),
            )
            plot_shap_bar(
                shap_vals, pd.DataFrame(X_shap),
                save_path=os.path.join(figures_dir, f"shap_bar_{best_name}.png"),
            )
            plot_shap_waterfall(
                shap_vals, pd.DataFrame(X_shap), idx=0,
                save_path=os.path.join(figures_dir, f"shap_waterfall_{best_name}.png"),
            )
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s", exc)

        try:
            # Reconstruct feature names from the preprocessor
            ohe = preproc_pipe.named_steps["preprocessor"].named_transformers_["cat"]["encoder"]
            cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
            feature_names = numeric_cols + cat_feature_names + binary_cols

            explain_with_lime(
                best_model,
                X_train_proc,
                X_test_proc,
                idx=0,
                feature_names=feature_names[: X_train_proc.shape[1]],
                save_path=os.path.join(figures_dir, f"lime_{best_name}.png"),
            )
        except Exception as exc:
            logger.warning("LIME explanation failed: %s", exc)

        if hasattr(best_model, "feature_importances_"):
            try:
                ohe = preproc_pipe.named_steps["preprocessor"].named_transformers_["cat"]["encoder"]
                cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
                feature_names = numeric_cols + cat_feature_names + binary_cols
                plot_feature_importance(
                    best_model,
                    feature_names=feature_names[: X_train_proc.shape[1]],
                    model_name=best_name,
                    save_path=os.path.join(figures_dir, f"feature_importance_{best_name}.png"),
                )
            except Exception as exc:
                logger.warning("Feature importance plot failed: %s", exc)

    # ------------------------------------------------------------------
    # 7. Print final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(float_format="{:.4f}".format))
    print(f"\nFull report saved to: {csv_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the pipeline."""
    args = _parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
