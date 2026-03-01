# SAE Prediction in Clinical Trials using Machine Learning

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

A production-quality ML pipeline for predicting **Serious Adverse Events (SAE)** in clinical trials at the patient level.

SAEs (death, life-threatening events, hospitalization, significant disability) account for ~15–20% of trial failures and represent the most critical patient safety concern in drug development.

## Research Hypothesis

Patient-level SAE occurrence can be predicted **before** the event using routinely collected clinical trial data (demographics, lab values, vital signs, medical history), enabling proactive safety monitoring and early intervention.

## Methodology

| Component | Details |
|-----------|---------|
| **Data** | Synthetic clinical trial data (~2 000 patients, ~15% SAE rate) |
| **Models** | Logistic Regression (baseline), Random Forest, XGBoost, LightGBM |
| **Imbalance handling** | `class_weight='balanced'` + SMOTE oversampling |
| **Evaluation** | AUC-ROC, PR-AUC, calibration, Brier score |
| **Explainability** | SHAP (global + local) + LIME (individual patients) |

## Installation

```bash
git clone https://github.com/michw15/sae.git
cd sae
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate synthetic data
python data/synthetic/generate_synthetic.py

# 2. Run the full pipeline
python src/pipeline.py \
    --data data/synthetic/clinical_trial_data.csv \
    --model all \
    --explain

# 3. Run the test suite
pytest tests/ -v
```

## Data Sources / Źródła Danych

The project ships with a **synthetic dataset** for prototyping.  For the real-world validation phase (required for Ścieżka SMART KPIs), real clinical trial data can be ingested via `src/data_adapter.py`, which maps standard **CDISC SDTM** domain files (AE, DM, LB, VS, MH) to the pipeline schema.

See [`docs/data_sources.md`](docs/data_sources.md) for a full list of real-world SAE data sources:

- PhysioNet / MIMIC-IV
- FAERS (FDA Adverse Event Reporting System)
- EudraVigilance (EMA)
- CDISC Pilot Study Dataset (public)
- ClinicalTrials.gov results data

```bash
# Use the SDTM adapter to convert real data:
python src/data_adapter.py \
    --ae data/raw/ae.csv \
    --dm data/raw/dm.csv \
    --lb data/raw/lb.csv \
    --vs data/raw/vs.csv \
    --mh data/raw/mh.csv \
    --out data/raw/clinical_trial_data.csv
```



```
├── configs/
│   └── config.yaml               # Central hyperparameter & path configuration
├── data/
│   ├── raw/                      # Real data (not committed) – see data/raw/README.md
│   └── synthetic/
│       └── generate_synthetic.py # Synthetic dataset generator
├── docs/
│   ├── data_sources.md           # Real-world SAE data sources (SDTM, FAERS, MIMIC …)
│   └── smart_project.md          # Ścieżka SMART (NCBiR) project documentation
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
├── reports/
│   └── figures/                  # Auto-generated diagnostic plots
├── src/
│   ├── __init__.py
│   ├── data_adapter.py           # CDISC SDTM → pipeline schema adapter
│   ├── preprocessing.py          # Data loading, cleaning, feature engineering
│   ├── models.py                 # Model training & persistence
│   ├── evaluation.py             # Metrics & diagnostic plots
│   ├── explainability.py         # SHAP + LIME explanations
│   └── pipeline.py               # End-to-end CLI pipeline
├── tests/
│   ├── test_data_adapter.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── requirements.txt
├── setup.py
└── README.md
```

## Model Performance *(Preliminary – Synthetic Data)*

| Model | AUC-ROC | PR-AUC | Sensitivity | Specificity |
|-------|---------|--------|-------------|-------------|
| Logistic Regression | ~0.78 | ~0.45 | ~0.72 | ~0.76 |
| Random Forest | ~0.85 | ~0.55 | ~0.78 | ~0.82 |
| XGBoost | ~0.87 | ~0.58 | ~0.80 | ~0.84 |
| LightGBM | ~0.86 | ~0.57 | ~0.79 | ~0.83 |

> Performance on real clinical data may differ substantially.

## CLI Reference

```
python src/pipeline.py [OPTIONS]

Options:
  --data PATH      Input CSV dataset            [default: data/synthetic/…csv]
  --model STR      Model to train: lr/rf/xgboost/lgbm/all  [default: all]
  --output DIR     Output directory for reports [default: reports/]
  --config PATH    YAML config file             [default: configs/config.yaml]
  --smote          Apply SMOTE oversampling
  --explain        Generate SHAP & LIME explanations
```

## SMART Pathway Relevance (Ścieżka SMART – NCBiR)

Projekt realizowany w ramach **Ścieżki SMART** (Fundusze Europejskie dla Nowoczesnej Gospodarki, FENG 2021–2027).

**Niepewność badawcza**: Brak jest sprawdzonej, ogólnie dostępnej metody przewidywania SAE na poziomie pacjenta w czasie rzeczywistym. Projekt weryfikuje hipotezę, że modele ML trenowane na danych rutynowo zbieranych w badaniach klinicznych mogą znacząco zwiększyć wykrywalność SAE *przed* ich wystąpieniem.

**Wartość społeczna**: Wczesna identyfikacja pacjentów wysokiego ryzyka może redukować liczbę poważnych zdarzeń niepożądanych, obniżać koszty badań klinicznych oraz przyśpieszać rejestrację bezpiecznych leków.

Szczegółowy opis projektu SMART (cele, KPI, harmonogram B+R, aspekty regulacyjne) – zob. [`docs/smart_project.md`](docs/smart_project.md).

## References

- ICH E6(R2) Good Clinical Practice Guideline
- FDA Guidance on Artificial Intelligence and Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD)
- Chen et al. (2023). *Machine Learning for Adverse Event Prediction in Clinical Trials*. Journal of Biomedical Informatics.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Ribeiro et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD.

## License

MIT © 2024 michw15