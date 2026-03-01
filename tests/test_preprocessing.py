"""pytest tests for src/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_preprocessing_pipeline,
    cap_outliers,
    engineer_features,
    handle_missing_values,
    load_data,
    prepare_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a minimal clinical-trial DataFrame for testing."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "sex": rng.choice(["M", "F"], n),
            "weight": rng.uniform(50, 120, n),
            "height": rng.uniform(150, 195, n),
            "bmi": rng.uniform(18, 35, n),
            "alt": rng.uniform(5, 200, n),
            "ast": rng.uniform(5, 200, n),
            "creatinine": rng.uniform(0.5, 5.0, n),
            "hemoglobin": rng.uniform(8, 18, n),
            "platelets": rng.uniform(50, 600, n),
            "wbc": rng.uniform(2, 20, n),
            "systolic_bp": rng.uniform(90, 180, n),
            "diastolic_bp": rng.uniform(60, 110, n),
            "heart_rate": rng.uniform(50, 120, n),
            "temperature": rng.uniform(36, 39, n),
            "diabetes": rng.integers(0, 2, n),
            "hypertension": rng.integers(0, 2, n),
            "renal_impairment": rng.integers(0, 2, n),
            "hepatic_impairment": rng.integers(0, 2, n),
            "prior_sae": rng.integers(0, 2, n),
            "cardiac_history": rng.integers(0, 2, n),
            "treatment_arm": rng.integers(0, 2, n),
            "study_phase": rng.choice([2, 3], n),
            "visit_number": rng.integers(1, 11, n),
            "dose_level": rng.integers(1, 5, n),
            "comorbidity_score": rng.integers(0, 6, n),
            "sae_occurred": rng.integers(0, 2, n),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_data_returns_dataframe(tmp_path: "pathlib.Path", sample_df: pd.DataFrame) -> None:
    """load_data should return a non-empty DataFrame."""
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    df = load_data(str(csv_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == sample_df.shape


def test_load_data_raises_for_missing_file() -> None:
    """load_data should raise FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        load_data("/tmp/does_not_exist_sae.csv")


def test_handle_missing_values_no_nulls_remaining(sample_df: pd.DataFrame) -> None:
    """After imputation, the DataFrame should contain no NaN values."""
    # Introduce some NaNs
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "age"] = np.nan
    df_with_nan.loc[1, "sex"] = np.nan
    df_with_nan.loc[2, "creatinine"] = np.nan

    result = handle_missing_values(df_with_nan)

    assert result.isna().sum().sum() == 0


def test_cap_outliers_within_bounds(sample_df: pd.DataFrame) -> None:
    """cap_outliers should constrain values to within IQR-based fences."""
    # Introduce extreme values
    df_extreme = sample_df.copy()
    df_extreme.loc[0, "alt"] = 99999.0
    df_extreme.loc[1, "creatinine"] = -99999.0

    numeric_cols = ["alt", "creatinine"]
    result = cap_outliers(df_extreme, cols=numeric_cols)

    # Verify that the extreme values have been clipped using bounds from df_extreme
    for col in numeric_cols:
        q1 = df_extreme[col].quantile(0.25)
        q3 = df_extreme[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        assert result[col].max() <= upper + 1e-6
        assert result[col].min() >= lower - 1e-6


def test_engineer_features_adds_columns(sample_df: pd.DataFrame) -> None:
    """engineer_features should add the five expected columns."""
    result = engineer_features(sample_df)

    expected = {
        "age_group",
        "lab_risk_score",
        "age_comorbidity_interaction",
        "treatment_prior_sae",
        "bmi_category",
    }
    assert expected.issubset(result.columns)


def test_prepare_data_stratified_split(sample_df: pd.DataFrame) -> None:
    """prepare_data should return correctly sized, stratified train/test sets."""
    X_train, X_test, y_train, y_test = prepare_data(
        sample_df, target_col="sae_occurred", test_size=0.2, random_state=42
    )

    total = len(sample_df)
    expected_test = int(total * 0.2)

    assert len(X_test) == pytest.approx(expected_test, abs=2)
    assert len(X_train) + len(X_test) == total

    # Stratification: positive rate should be close in both splits
    assert abs(y_train.mean() - y_test.mean()) < 0.15
