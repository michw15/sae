"""Preprocessing utilities for the SAE clinical trial ML pipeline.

Provides functions and sklearn-compatible pipelines for loading, cleaning,
feature engineering, and splitting clinical trial data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset and perform basic validation.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Loaded dataset is empty: {path}")

    logger.info("Loaded dataset: %s rows × %s cols from %s", *df.shape, path)
    return df


# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric columns, mode for categorical.

    Args:
        df: Input DataFrame (may contain NaNs).

    Returns:
        DataFrame with no missing values.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Imputed %s with median=%.4f", col, median_val)

    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.debug("Imputed %s with mode=%s", col, mode_val)

    return df


# ---------------------------------------------------------------------------
# Outlier capping
# ---------------------------------------------------------------------------


def cap_outliers(
    df: pd.DataFrame,
    cols: Optional[list[str]] = None,
    method: str = "iqr",
) -> pd.DataFrame:
    """Cap extreme values using IQR-based (Tukey) fences.

    Args:
        df: Input DataFrame.
        cols: List of numeric column names to cap.  If *None*, all numeric
              columns are used.
        method: Only ``'iqr'`` is currently supported.

    Returns:
        DataFrame with outliers capped at [Q1 - 1.5·IQR, Q3 + 1.5·IQR].

    Raises:
        ValueError: If *method* is not ``'iqr'``.
    """
    if method != "iqr":
        raise ValueError(f"Unsupported outlier method: {method!r}.  Use 'iqr'.")

    df = df.copy()

    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
        logger.debug("Capped %s to [%.3f, %.3f]", col, lower, upper)

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived / interaction features to the dataset.

    New columns added:

    * ``age_group`` – ordinal age bins (``'<30'``, ``'30-44'``, ``'45-64'``,
      ``'65+'``).
    * ``lab_risk_score`` – composite lab score
      ``(ALT/40 + AST/40 + creatinine/1.2) / 3``.
    * ``age_comorbidity_interaction`` – ``age * comorbidity_score``.
    * ``treatment_prior_sae`` – interaction flag
      ``treatment_arm * prior_sae``.
    * ``bmi_category`` – WHO BMI bins (``'underweight'``, ``'normal'``,
      ``'overweight'``, ``'obese'``).

    Args:
        df: Input DataFrame containing the expected raw columns.

    Returns:
        DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 29, 44, 64, 200],
        labels=["<30", "30-44", "45-64", "65+"],
    ).astype(str)

    # Composite lab risk score
    df["lab_risk_score"] = (
        df["alt"] / 40.0 + df["ast"] / 40.0 + df["creatinine"] / 1.2
    ) / 3.0

    # Age × comorbidity interaction
    df["age_comorbidity_interaction"] = df["age"] * df["comorbidity_score"]

    # Treatment × prior SAE interaction
    df["treatment_prior_sae"] = df["treatment_arm"] * df["prior_sae"]

    # BMI categories
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25.0, 30.0, np.inf],
        labels=["underweight", "normal", "overweight", "obese"],
    ).astype(str)

    logger.info(
        "Engineered features: age_group, lab_risk_score, "
        "age_comorbidity_interaction, treatment_prior_sae, bmi_category"
    )
    return df


# ---------------------------------------------------------------------------
# sklearn Pipeline construction
# ---------------------------------------------------------------------------


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """Build an sklearn preprocessing Pipeline with ColumnTransformer.

    Numeric columns are imputed (median) then standardised.
    Categorical columns are imputed (most frequent) then one-hot encoded.

    Args:
        numeric_cols: Names of numeric feature columns.
        categorical_cols: Names of categorical (string) feature columns.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline`` object.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline


# ---------------------------------------------------------------------------
# Full data preparation
# ---------------------------------------------------------------------------


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "sae_occurred",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare data for model training: engineer features and stratified split.

    Steps performed:

    1. Handle missing values.
    2. Engineer derived features.
    3. Stratified train/test split.

    Args:
        df: Raw input DataFrame.
        target_col: Name of the binary target column.
        test_size: Fraction of samples reserved for the test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    df = handle_missing_values(df)
    df = engineer_features(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Train size: %d | Test size: %d | Positive rate (train): %.3f",
        len(X_train),
        len(X_test),
        y_train.mean(),
    )

    return X_train, X_test, y_train, y_test
