"""CDISC SDTM → pipeline schema adapter.

Maps standard CDISC SDTM domain columns to the column names expected by the
SAE prediction pipeline (``src/preprocessing.py``).  Supports the five most
relevant domains: AE (Adverse Events), DM (Demographics), LB (Laboratory
Tests), VS (Vital Signs), and MH (Medical History).

Typical usage::

    python src/data_adapter.py \\
        --ae  data/raw/ae.csv  \\
        --dm  data/raw/dm.csv  \\
        --lb  data/raw/lb.csv  \\
        --vs  data/raw/vs.csv  \\
        --mh  data/raw/mh.csv  \\
        --out data/raw/clinical_trial_data.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name maps  (SDTM standard → pipeline schema)
# ---------------------------------------------------------------------------

#: Key demographic columns expected in the DM domain.
_DM_MAP: dict[str, str] = {
    "USUBJID": "subject_id",
    "AGE": "age",
    "SEX": "sex",
    "ACTARM": "treatment_arm_label",
    "ARM": "treatment_arm_label",
    "VISITNUM": "visit_number",
    "STUDYPHASE": "study_phase_label",
    "PHASE": "study_phase_label",
}

#: Lab test name → pipeline column name.
_LB_TEST_MAP: dict[str, str] = {
    "ALT": "alt",
    "ALAT": "alt",
    "ALANINE AMINOTRANSFERASE": "alt",
    "AST": "ast",
    "ASAT": "ast",
    "ASPARTATE AMINOTRANSFERASE": "ast",
    "CREATININE": "creatinine",
    "HEMOGLOBIN": "hemoglobin",
    "HAEMOGLOBIN": "hemoglobin",
    "HGB": "hemoglobin",
    "PLATELETS": "platelets",
    "PLATELET COUNT": "platelets",
    "PLT": "platelets",
    "WBC": "wbc",
    "LEUKOCYTES": "wbc",
    "WHITE BLOOD CELL COUNT": "wbc",
}

#: Vital sign test name → pipeline column name.
_VS_TEST_MAP: dict[str, str] = {
    "SYSBP": "systolic_bp",
    "SYSTOLIC BLOOD PRESSURE": "systolic_bp",
    "DIABP": "diastolic_bp",
    "DIASTOLIC BLOOD PRESSURE": "diastolic_bp",
    "HR": "heart_rate",
    "PULSE RATE": "heart_rate",
    "HEART RATE": "heart_rate",
    "TEMP": "temperature",
    "TEMPERATURE": "temperature",
}

#: Medical history terms → pipeline binary flags.
_MH_TERM_MAP: dict[str, str] = {
    "DIABETES": "diabetes",
    "DIABETES MELLITUS": "diabetes",
    "HYPERTENSION": "hypertension",
    "HIGH BLOOD PRESSURE": "hypertension",
    "RENAL IMPAIRMENT": "renal_impairment",
    "RENAL FAILURE": "renal_impairment",
    "KIDNEY DISEASE": "renal_impairment",
    "HEPATIC IMPAIRMENT": "hepatic_impairment",
    "LIVER DISEASE": "hepatic_impairment",
    "HEPATIC FAILURE": "hepatic_impairment",
    "CARDIAC DISORDER": "cardiac_history",
    "HEART FAILURE": "cardiac_history",
    "MYOCARDIAL INFARCTION": "cardiac_history",
    "CORONARY ARTERY DISEASE": "cardiac_history",
}


# ---------------------------------------------------------------------------
# Domain loaders
# ---------------------------------------------------------------------------


def _load_csv(path: Optional[str], domain: str) -> Optional[pd.DataFrame]:
    """Load a CSV file and return a DataFrame, or *None* if path is absent.

    Args:
        path: Filesystem path to the CSV file, or ``None``.
        domain: Human-readable domain name used in log messages.

    Returns:
        Loaded DataFrame, or ``None`` when *path* is ``None``.

    Raises:
        FileNotFoundError: If *path* is provided but the file does not exist.
    """
    if path is None:
        logger.debug("Domain %s not provided – skipping.", domain)
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"{domain} file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %s domain: %d rows from %s", domain, len(df), path)
    return df


# ---------------------------------------------------------------------------
# AE domain → SAE flag + prior-SAE flag
# ---------------------------------------------------------------------------


def extract_sae_flags(ae_df: pd.DataFrame) -> pd.DataFrame:
    """Extract SAE outcome and prior-SAE flag from the AE domain.

    Args:
        ae_df: AE domain DataFrame containing at minimum the columns
               ``USUBJID`` and ``AESER`` (Serious Adverse Event flag,
               ``'Y'`` / ``'N'``).

    Returns:
        DataFrame with columns ``['subject_id', 'sae_occurred', 'prior_sae']``.
    """
    ae = ae_df.copy()
    ae.columns = ae.columns.str.upper()

    if "USUBJID" not in ae.columns:
        raise ValueError("AE domain must contain 'USUBJID' column.")
    if "AESER" not in ae.columns:
        raise ValueError("AE domain must contain 'AESER' column (serious flag).")

    ae["is_sae"] = ae["AESER"].str.upper().eq("Y").astype(int)

    per_subject = ae.groupby("USUBJID")["is_sae"].max().reset_index()
    per_subject.columns = ["subject_id", "sae_occurred"]

    # prior_sae: ideally derived from temporal data (AESTDTC vs current visit).
    # Without reliable visit-date context, we default to 0 rather than copy
    # sae_occurred, which would introduce data leakage into the feature set.
    per_subject["prior_sae"] = 0

    logger.info(
        "SAE rate: %.3f (%d / %d subjects)",
        per_subject["sae_occurred"].mean(),
        per_subject["sae_occurred"].sum(),
        len(per_subject),
    )
    return per_subject[["subject_id", "sae_occurred", "prior_sae"]]


# ---------------------------------------------------------------------------
# DM domain → demographics
# ---------------------------------------------------------------------------


def extract_demographics(dm_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and map demographic columns from the DM domain.

    Args:
        dm_df: DM domain DataFrame.

    Returns:
        DataFrame with ``subject_id``, ``age``, ``sex``, ``treatment_arm``,
        ``study_phase``, and ``visit_number`` columns.
    """
    dm = dm_df.copy()
    dm.columns = dm.columns.str.upper()

    if "USUBJID" not in dm.columns:
        raise ValueError("DM domain must contain 'USUBJID' column.")

    result = pd.DataFrame()
    result["subject_id"] = dm["USUBJID"]

    # Age
    result["age"] = pd.to_numeric(dm.get("AGE"), errors="coerce")

    # Sex – normalise to M/F
    if "SEX" in dm.columns:
        result["sex"] = dm["SEX"].str.upper().map(
            {"M": "M", "MALE": "M", "F": "F", "FEMALE": "F"}
        )
    else:
        result["sex"] = np.nan

    # Weight / Height (optional)
    result["weight"] = pd.to_numeric(dm.get("WEIGHT", np.nan), errors="coerce")
    result["height"] = pd.to_numeric(dm.get("HEIGHT", np.nan), errors="coerce")

    # Treatment arm: encode as binary (0 = control/placebo, 1 = active)
    arm_col = next((c for c in ("ACTARM", "ARM") if c in dm.columns), None)
    if arm_col:
        arm_vals = dm[arm_col].str.upper().fillna("")
        result["treatment_arm"] = (
            ~arm_vals.str.contains("PLACEBO|CONTROL|VEHICLE", regex=True)
        ).astype(int)
    else:
        result["treatment_arm"] = 0

    # Study phase: use numeric part of PHASE string (e.g. "PHASE 3" → 3)
    phase_col = next((c for c in ("PHASE", "STUDYPHASE") if c in dm.columns), None)
    if phase_col:
        result["study_phase"] = (
            dm[phase_col]
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
            .fillna(3)
            .astype(int)
        )
    else:
        result["study_phase"] = 3

    # Visit number (first visit per subject)
    if "VISITNUM" in dm.columns:
        result["visit_number"] = pd.to_numeric(dm["VISITNUM"], errors="coerce").fillna(1).astype(int)
    else:
        result["visit_number"] = 1

    # Dose level (optional; default 1 if absent)
    if "EXDOSE" in dm.columns:
        result["dose_level"] = pd.to_numeric(dm["EXDOSE"], errors="coerce").fillna(1)
    else:
        result["dose_level"] = 1

    return result.drop_duplicates(subset="subject_id")


# ---------------------------------------------------------------------------
# LB domain → laboratory values (wide pivot)
# ---------------------------------------------------------------------------


def extract_lab_values(lb_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the LB domain from long to wide format, mapping test names.

    Args:
        lb_df: LB domain DataFrame with columns ``USUBJID``, ``LBTESTCD`` or
               ``LBTEST``, and ``LBSTRESN`` (numeric result).

    Returns:
        Wide DataFrame with one row per subject and columns for each lab value.
    """
    lb = lb_df.copy()
    lb.columns = lb.columns.str.upper()

    if "USUBJID" not in lb.columns:
        raise ValueError("LB domain must contain 'USUBJID' column.")

    # Prefer LBTESTCD (short code); fall back to LBTEST (long name)
    test_col = "LBTESTCD" if "LBTESTCD" in lb.columns else "LBTEST"
    value_col = "LBSTRESN" if "LBSTRESN" in lb.columns else "LBORRES"

    lb["test_mapped"] = lb[test_col].str.upper().map(_LB_TEST_MAP)
    lb = lb[lb["test_mapped"].notna()].copy()
    lb[value_col] = pd.to_numeric(lb[value_col], errors="coerce")

    # Aggregate: median per subject per test
    wide = (
        lb.groupby(["USUBJID", "test_mapped"])[value_col]
        .median()
        .unstack("test_mapped")
        .reset_index()
        .rename(columns={"USUBJID": "subject_id"})
    )

    logger.info("Lab columns extracted: %s", list(wide.columns[1:]))
    return wide


# ---------------------------------------------------------------------------
# VS domain → vital signs (wide pivot)
# ---------------------------------------------------------------------------


def extract_vital_signs(vs_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the VS domain from long to wide format.

    Args:
        vs_df: VS domain DataFrame with ``USUBJID``, ``VSTESTCD`` / ``VSTEST``,
               and ``VSSTRESN``.

    Returns:
        Wide DataFrame with one row per subject and vital sign columns.
    """
    vs = vs_df.copy()
    vs.columns = vs.columns.str.upper()

    if "USUBJID" not in vs.columns:
        raise ValueError("VS domain must contain 'USUBJID' column.")

    test_col = "VSTESTCD" if "VSTESTCD" in vs.columns else "VSTEST"
    value_col = "VSSTRESN" if "VSSTRESN" in vs.columns else "VSORRES"

    vs["test_mapped"] = vs[test_col].str.upper().map(_VS_TEST_MAP)
    vs = vs[vs["test_mapped"].notna()].copy()
    vs[value_col] = pd.to_numeric(vs[value_col], errors="coerce")

    wide = (
        vs.groupby(["USUBJID", "test_mapped"])[value_col]
        .median()
        .unstack("test_mapped")
        .reset_index()
        .rename(columns={"USUBJID": "subject_id"})
    )

    logger.info("Vital sign columns extracted: %s", list(wide.columns[1:]))
    return wide


# ---------------------------------------------------------------------------
# MH domain → binary comorbidity flags
# ---------------------------------------------------------------------------


def extract_medical_history(mh_df: pd.DataFrame) -> pd.DataFrame:
    """Create binary comorbidity flags from the MH domain.

    Args:
        mh_df: MH domain DataFrame with ``USUBJID`` and ``MHDECOD`` or
               ``MHTERM`` (medical history term).

    Returns:
        DataFrame with one row per subject and binary flag columns.
    """
    mh = mh_df.copy()
    mh.columns = mh.columns.str.upper()

    if "USUBJID" not in mh.columns:
        raise ValueError("MH domain must contain 'USUBJID' column.")

    term_col = "MHDECOD" if "MHDECOD" in mh.columns else "MHTERM"
    mh["flag"] = mh[term_col].str.upper().map(_MH_TERM_MAP)
    mh = mh[mh["flag"].notna()].copy()
    mh["value"] = 1

    flags_df = (
        mh.groupby(["USUBJID", "flag"])["value"]
        .max()
        .unstack("flag")
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"USUBJID": "subject_id"})
    )

    logger.info("Medical history flags extracted: %s", list(flags_df.columns[1:]))
    return flags_df


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------


def build_pipeline_dataset(
    ae_df: pd.DataFrame,
    dm_df: pd.DataFrame,
    lb_df: Optional[pd.DataFrame] = None,
    vs_df: Optional[pd.DataFrame] = None,
    mh_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge SDTM domains into a single flat DataFrame for the pipeline.

    Missing domains produce zero-filled or median-filled columns so the
    pipeline always receives the expected schema.

    Args:
        ae_df: AE domain DataFrame (required).
        dm_df: DM domain DataFrame (required).
        lb_df: LB domain DataFrame (optional).
        vs_df: VS domain DataFrame (optional).
        mh_df: MH domain DataFrame (optional).

    Returns:
        Merged DataFrame ready to be passed to ``src/preprocessing.prepare_data``.
    """
    sae_flags = extract_sae_flags(ae_df)
    demographics = extract_demographics(dm_df)

    merged = demographics.merge(sae_flags, on="subject_id", how="left")
    # Default SAE = 0 for subjects absent from AE domain
    merged["sae_occurred"] = merged["sae_occurred"].fillna(0).astype(int)
    merged["prior_sae"] = merged["prior_sae"].fillna(0).astype(int)

    if lb_df is not None:
        labs = extract_lab_values(lb_df)
        merged = merged.merge(labs, on="subject_id", how="left")

    if vs_df is not None:
        vitals = extract_vital_signs(vs_df)
        merged = merged.merge(vitals, on="subject_id", how="left")

    if mh_df is not None:
        history = extract_medical_history(mh_df)
        merged = merged.merge(history, on="subject_id", how="left")

    # Fill binary comorbidity flags with 0 if absent
    comorbidity_cols = [
        "diabetes", "hypertension", "renal_impairment",
        "hepatic_impairment", "cardiac_history",
    ]
    for col in comorbidity_cols:
        if col not in merged.columns:
            merged[col] = 0
        else:
            merged[col] = merged[col].fillna(0).astype(int)

    # Derive BMI if weight and height are present
    if "weight" in merged.columns and "height" in merged.columns:
        height_m = merged["height"] / 100.0
        merged["bmi"] = (merged["weight"] / height_m**2).round(2)
    else:
        merged["bmi"] = np.nan

    # Comorbidity score
    merged["comorbidity_score"] = merged[
        [c for c in comorbidity_cols if c in merged.columns]
    ].sum(axis=1)

    # Drop internal helper column
    merged = merged.drop(columns=["subject_id"], errors="ignore")

    logger.info(
        "Final dataset: %d rows × %d columns | SAE rate: %.3f",
        len(merged),
        merged.shape[1],
        merged["sae_occurred"].mean(),
    )
    return merged


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt CDISC SDTM domain CSVs to the SAE pipeline schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ae", required=True, help="Path to AE domain CSV.")
    parser.add_argument("--dm", required=True, help="Path to DM domain CSV.")
    parser.add_argument("--lb", default=None, help="Path to LB domain CSV (optional).")
    parser.add_argument("--vs", default=None, help="Path to VS domain CSV (optional).")
    parser.add_argument("--mh", default=None, help="Path to MH domain CSV (optional).")
    parser.add_argument(
        "--out",
        default="data/raw/clinical_trial_data.csv",
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point for the SDTM data adapter."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    args = _parse_args()

    ae_df = _load_csv(args.ae, "AE")
    dm_df = _load_csv(args.dm, "DM")
    lb_df = _load_csv(args.lb, "LB")
    vs_df = _load_csv(args.vs, "VS")
    mh_df = _load_csv(args.mh, "MH")

    result = build_pipeline_dataset(ae_df, dm_df, lb_df, vs_df, mh_df)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Dataset saved to {args.out}  ({len(result)} rows × {result.shape[1]} columns)")


if __name__ == "__main__":
    main()
