"""Generate a realistic synthetic clinical trial dataset for SAE prediction.

This script creates a dataset of 2000 patients with demographics, lab values,
vital signs, medical history, and trial metadata. The target variable
``sae_occurred`` is generated via a logistic function based on clinically
meaningful risk factors.

Usage::

    python data/synthetic/generate_synthetic.py
"""

import os

import numpy as np
import pandas as pd


def generate_synthetic_data(n_patients: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic clinical trial dataset.

    Args:
        n_patients: Number of patient records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame containing the synthetic clinical trial data.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Demographics
    # ------------------------------------------------------------------
    age = rng.uniform(18, 80, n_patients)
    sex = rng.choice(["M", "F"], n_patients)
    weight = rng.uniform(50, 120, n_patients)
    height = rng.uniform(150, 195, n_patients)
    bmi = weight / ((height / 100) ** 2)

    # ------------------------------------------------------------------
    # Lab values
    # ------------------------------------------------------------------
    alt = rng.uniform(5, 200, n_patients)
    ast = rng.uniform(5, 200, n_patients)
    creatinine = rng.uniform(0.5, 5.0, n_patients)
    hemoglobin = rng.uniform(8, 18, n_patients)
    platelets = rng.uniform(50, 600, n_patients)
    wbc = rng.uniform(2, 20, n_patients)

    # ------------------------------------------------------------------
    # Vital signs
    # ------------------------------------------------------------------
    systolic_bp = rng.uniform(90, 180, n_patients)
    diastolic_bp = rng.uniform(60, 110, n_patients)
    heart_rate = rng.uniform(50, 120, n_patients)
    temperature = rng.uniform(36, 39, n_patients)

    # ------------------------------------------------------------------
    # Medical history flags (binary)
    # ------------------------------------------------------------------
    diabetes = rng.binomial(1, 0.15, n_patients)
    hypertension = rng.binomial(1, 0.30, n_patients)
    renal_impairment = rng.binomial(1, 0.10, n_patients)
    hepatic_impairment = rng.binomial(1, 0.08, n_patients)
    prior_sae = rng.binomial(1, 0.12, n_patients)
    cardiac_history = rng.binomial(1, 0.20, n_patients)

    # ------------------------------------------------------------------
    # Trial metadata
    # ------------------------------------------------------------------
    treatment_arm = rng.binomial(1, 0.5, n_patients)
    study_phase = rng.choice([2, 3], n_patients, p=[0.4, 0.6])
    visit_number = rng.integers(1, 11, n_patients)
    dose_level = rng.integers(1, 5, n_patients)

    # ------------------------------------------------------------------
    # Comorbidity score
    # ------------------------------------------------------------------
    comorbidity_score = (
        diabetes
        + hypertension
        + renal_impairment
        + hepatic_impairment
        + prior_sae
        + cardiac_history
    )

    # ------------------------------------------------------------------
    # Target variable: sae_occurred (~15% positive rate)
    # ------------------------------------------------------------------
    log_odds = (
        -4.0
        + 0.04 * (age - 65) * (age > 65)
        + 1.5 * prior_sae
        + 0.3 * (creatinine - 1.0)
        + 0.015 * (alt - 40)
        + 0.4 * treatment_arm
        + 0.2 * renal_impairment
        + 0.15 * hepatic_impairment
        + 0.1 * comorbidity_score
    )
    prob_sae = 1 / (1 + np.exp(-log_odds))
    sae_occurred = rng.binomial(1, prob_sae, n_patients)

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "age": np.round(age, 1),
            "sex": sex,
            "weight": np.round(weight, 1),
            "height": np.round(height, 1),
            "bmi": np.round(bmi, 2),
            "alt": np.round(alt, 1),
            "ast": np.round(ast, 1),
            "creatinine": np.round(creatinine, 2),
            "hemoglobin": np.round(hemoglobin, 1),
            "platelets": np.round(platelets, 0),
            "wbc": np.round(wbc, 1),
            "systolic_bp": np.round(systolic_bp, 0),
            "diastolic_bp": np.round(diastolic_bp, 0),
            "heart_rate": np.round(heart_rate, 0),
            "temperature": np.round(temperature, 2),
            "diabetes": diabetes,
            "hypertension": hypertension,
            "renal_impairment": renal_impairment,
            "hepatic_impairment": hepatic_impairment,
            "prior_sae": prior_sae,
            "cardiac_history": cardiac_history,
            "treatment_arm": treatment_arm,
            "study_phase": study_phase,
            "visit_number": visit_number,
            "dose_level": dose_level,
            "comorbidity_score": comorbidity_score,
            "sae_occurred": sae_occurred,
        }
    )

    return df


def main() -> None:
    """Entry point: generate data, print summary, and save to CSV."""
    df = generate_synthetic_data(n_patients=2000, seed=42)

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "clinical_trial_data.csv"
    )
    df.to_csv(output_path, index=False)

    print("=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nNumeric statistics:\n{df.describe().T.to_string()}")
    print(f"\nClass distribution (sae_occurred):\n{df['sae_occurred'].value_counts()}")
    print(
        f"\nPositive rate: {df['sae_occurred'].mean():.3f} "
        f"({df['sae_occurred'].sum()} / {len(df)})"
    )
    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
