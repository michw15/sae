"""pytest tests for src/data_adapter.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_adapter import (
    build_pipeline_dataset,
    extract_demographics,
    extract_lab_values,
    extract_medical_history,
    extract_sae_flags,
    extract_vital_signs,
)


# ---------------------------------------------------------------------------
# Fixtures – minimal SDTM domain DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture()
def ae_df() -> pd.DataFrame:
    """Minimal AE domain DataFrame."""
    return pd.DataFrame(
        {
            "USUBJID": ["S001", "S002", "S003", "S004"],
            "AETERM": ["Nausea", "Hospitalisation", "Headache", "Death"],
            "AESER": ["N", "Y", "N", "Y"],
        }
    )


@pytest.fixture()
def dm_df() -> pd.DataFrame:
    """Minimal DM domain DataFrame."""
    return pd.DataFrame(
        {
            "USUBJID": ["S001", "S002", "S003", "S004"],
            "AGE": [45, 62, 38, 71],
            "SEX": ["M", "F", "M", "F"],
            "ARM": ["PLACEBO", "ACTIVE DRUG", "ACTIVE DRUG", "PLACEBO"],
            "PHASE": ["PHASE 3", "PHASE 3", "PHASE 2", "PHASE 3"],
            "VISITNUM": [1, 2, 1, 3],
        }
    )


@pytest.fixture()
def lb_df() -> pd.DataFrame:
    """Minimal LB domain DataFrame (long format) with varying values per subject."""
    # Different values per subject to exercise aggregation and real-world variation
    subject_labs = {
        "S001": {"ALT": 35.0, "AST": 28.0, "CREATININE": 1.1, "HEMOGLOBIN": 14.0, "PLATELETS": 220.0, "WBC": 7.5},
        "S002": {"ALT": 85.0, "AST": 70.0, "CREATININE": 2.3, "HEMOGLOBIN": 10.5, "PLATELETS": 110.0, "WBC": 12.0},
        "S003": {"ALT": 22.0, "AST": 18.0, "CREATININE": 0.9, "HEMOGLOBIN": 16.0, "PLATELETS": 300.0, "WBC": 6.0},
        "S004": {"ALT": 150.0, "AST": 130.0, "CREATININE": 4.5, "HEMOGLOBIN": 8.5, "PLATELETS": 60.0, "WBC": 18.0},
    }
    rows = []
    for subj, tests in subject_labs.items():
        for test, val in tests.items():
            rows.append({"USUBJID": subj, "LBTESTCD": test, "LBSTRESN": val})
    return pd.DataFrame(rows)


@pytest.fixture()
def vs_df() -> pd.DataFrame:
    """Minimal VS domain DataFrame (long format)."""
    rows = []
    for subj in ["S001", "S002", "S003", "S004"]:
        for test, val in [("SYSBP", 120.0), ("DIABP", 80.0),
                          ("HR", 70.0), ("TEMP", 36.6)]:
            rows.append({"USUBJID": subj, "VSTESTCD": test, "VSSTRESN": val})
    return pd.DataFrame(rows)


@pytest.fixture()
def mh_df() -> pd.DataFrame:
    """Minimal MH domain DataFrame."""
    return pd.DataFrame(
        {
            "USUBJID": ["S001", "S002", "S002", "S003"],
            "MHDECOD": ["DIABETES MELLITUS", "HYPERTENSION", "RENAL IMPAIRMENT", "DIABETES"],
        }
    )


# ---------------------------------------------------------------------------
# extract_sae_flags
# ---------------------------------------------------------------------------


def test_extract_sae_flags_identifies_serious_events(ae_df: pd.DataFrame) -> None:
    """AESER='Y' subjects should have sae_occurred=1."""
    result = extract_sae_flags(ae_df)
    assert result.loc[result["subject_id"] == "S002", "sae_occurred"].iloc[0] == 1
    assert result.loc[result["subject_id"] == "S004", "sae_occurred"].iloc[0] == 1


def test_extract_sae_flags_non_serious_zero(ae_df: pd.DataFrame) -> None:
    """AESER='N' subjects should have sae_occurred=0."""
    result = extract_sae_flags(ae_df)
    assert result.loc[result["subject_id"] == "S001", "sae_occurred"].iloc[0] == 0


def test_extract_sae_flags_missing_aeser_raises() -> None:
    """Missing AESER column should raise ValueError."""
    bad_df = pd.DataFrame({"USUBJID": ["S001"], "AETERM": ["Nausea"]})
    with pytest.raises(ValueError, match="AESER"):
        extract_sae_flags(bad_df)


def test_extract_sae_flags_missing_usubjid_raises() -> None:
    """Missing USUBJID column should raise ValueError."""
    bad_df = pd.DataFrame({"AETERM": ["Nausea"], "AESER": ["N"]})
    with pytest.raises(ValueError, match="USUBJID"):
        extract_sae_flags(bad_df)


# ---------------------------------------------------------------------------
# extract_demographics
# ---------------------------------------------------------------------------


def test_extract_demographics_age_and_sex(dm_df: pd.DataFrame) -> None:
    """Age and sex columns should be correctly extracted."""
    result = extract_demographics(dm_df)
    assert "age" in result.columns
    assert "sex" in result.columns
    assert result["age"].iloc[0] == 45
    assert result["sex"].iloc[0] == "M"


def test_extract_demographics_treatment_arm_encoding(dm_df: pd.DataFrame) -> None:
    """PLACEBO arm should map to 0; active arm should map to 1."""
    result = extract_demographics(dm_df)
    placebo_subjects = result[result["treatment_arm"] == 0]["subject_id"].tolist()
    active_subjects = result[result["treatment_arm"] == 1]["subject_id"].tolist()
    assert "S001" in placebo_subjects  # ARM = PLACEBO
    assert "S002" in active_subjects   # ARM = ACTIVE DRUG


def test_extract_demographics_study_phase(dm_df: pd.DataFrame) -> None:
    """Study phase should be extracted as integer (2 or 3)."""
    result = extract_demographics(dm_df)
    phases = set(result["study_phase"].tolist())
    assert phases.issubset({2, 3})


def test_extract_demographics_missing_usubjid_raises() -> None:
    """Missing USUBJID should raise ValueError."""
    bad_df = pd.DataFrame({"AGE": [45], "SEX": ["M"]})
    with pytest.raises(ValueError, match="USUBJID"):
        extract_demographics(bad_df)


# ---------------------------------------------------------------------------
# extract_lab_values
# ---------------------------------------------------------------------------


def test_extract_lab_values_returns_wide_format(lb_df: pd.DataFrame) -> None:
    """Lab values should be pivoted to one row per subject."""
    result = extract_lab_values(lb_df)
    assert len(result) == 4  # 4 subjects
    assert "alt" in result.columns
    assert "creatinine" in result.columns


def test_extract_lab_values_median_aggregation(lb_df: pd.DataFrame) -> None:
    """Duplicate entries should be aggregated by median."""
    # Add a duplicate ALT for S001
    extra = pd.DataFrame({"USUBJID": ["S001"], "LBTESTCD": ["ALT"], "LBSTRESN": [65.0]})
    lb_dup = pd.concat([lb_df, extra], ignore_index=True)
    result = extract_lab_values(lb_dup)
    s001_alt = result.loc[result["subject_id"] == "S001", "alt"].iloc[0]
    assert s001_alt == pytest.approx(np.median([35.0, 65.0]))


# ---------------------------------------------------------------------------
# extract_vital_signs
# ---------------------------------------------------------------------------


def test_extract_vital_signs_wide_format(vs_df: pd.DataFrame) -> None:
    """Vital signs should be pivoted to one row per subject."""
    result = extract_vital_signs(vs_df)
    assert len(result) == 4
    assert "systolic_bp" in result.columns
    assert "heart_rate" in result.columns


# ---------------------------------------------------------------------------
# extract_medical_history
# ---------------------------------------------------------------------------


def test_extract_medical_history_binary_flags(mh_df: pd.DataFrame) -> None:
    """Known comorbidities should produce binary flag columns."""
    result = extract_medical_history(mh_df)
    assert "diabetes" in result.columns
    assert "hypertension" in result.columns
    # S002 has hypertension
    assert result.loc[result["subject_id"] == "S002", "hypertension"].iloc[0] == 1


def test_extract_medical_history_absent_flag_zero(mh_df: pd.DataFrame) -> None:
    """Subjects without a specific condition should have flag=0 (via outer join fill)."""
    result = extract_medical_history(mh_df)
    # S001 has diabetes only; hypertension should not appear in result for S001
    # (or should be 0 if it does appear)
    if "hypertension" in result.columns:
        s001_hyp = result.loc[result["subject_id"] == "S001", "hypertension"]
        if not s001_hyp.empty:
            assert s001_hyp.iloc[0] == 0


# ---------------------------------------------------------------------------
# build_pipeline_dataset (integration)
# ---------------------------------------------------------------------------


def test_build_pipeline_dataset_returns_dataframe(
    ae_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    lb_df: pd.DataFrame,
    vs_df: pd.DataFrame,
    mh_df: pd.DataFrame,
) -> None:
    """Full merge should return a non-empty DataFrame."""
    result = build_pipeline_dataset(ae_df, dm_df, lb_df, vs_df, mh_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4


def test_build_pipeline_dataset_has_sae_column(
    ae_df: pd.DataFrame, dm_df: pd.DataFrame
) -> None:
    """Result must contain 'sae_occurred' column."""
    result = build_pipeline_dataset(ae_df, dm_df)
    assert "sae_occurred" in result.columns


def test_build_pipeline_dataset_no_subject_id_column(
    ae_df: pd.DataFrame, dm_df: pd.DataFrame
) -> None:
    """subject_id helper column should be dropped from the final dataset."""
    result = build_pipeline_dataset(ae_df, dm_df)
    assert "subject_id" not in result.columns


def test_build_pipeline_dataset_missing_optional_domains_no_error(
    ae_df: pd.DataFrame, dm_df: pd.DataFrame
) -> None:
    """Omitting LB, VS, MH should not raise an error."""
    result = build_pipeline_dataset(ae_df, dm_df, lb_df=None, vs_df=None, mh_df=None)
    assert len(result) == 4


def test_build_pipeline_dataset_comorbidity_defaults_zero(
    ae_df: pd.DataFrame, dm_df: pd.DataFrame
) -> None:
    """Without MH domain, all comorbidity flags should default to 0."""
    result = build_pipeline_dataset(ae_df, dm_df, mh_df=None)
    for col in ["diabetes", "hypertension", "renal_impairment",
                "hepatic_impairment", "cardiac_history"]:
        assert result[col].sum() == 0


def test_build_pipeline_dataset_bmi_computed(
    ae_df: pd.DataFrame, dm_df: pd.DataFrame
) -> None:
    """BMI should be computed when weight and height are present."""
    dm_with_wh = dm_df.copy()
    dm_with_wh["WEIGHT"] = [70.0, 80.0, 65.0, 90.0]
    dm_with_wh["HEIGHT"] = [170.0, 175.0, 165.0, 180.0]
    result = build_pipeline_dataset(ae_df, dm_with_wh)
    assert "bmi" in result.columns
    assert result["bmi"].notna().all()
