# Real Clinical Trial Data

Place real-world clinical trial data files here. This directory is **not committed** to version control (see `.gitignore`).

## Expected Input Files

When using real CDISC SDTM data, place the following domain exports in this directory:

| File | SDTM Domain | Required | Description |
|------|-------------|----------|-------------|
| `ae.csv` | AE – Adverse Events | **Yes** | Contains `AESER` flag for SAE classification |
| `dm.csv` | DM – Demographics | **Yes** | Age, sex, treatment arm, study phase |
| `lb.csv` | LB – Laboratory Tests | Recommended | ALT, AST, creatinine, hemoglobin, platelets, WBC |
| `vs.csv` | VS – Vital Signs | Recommended | Systolic/diastolic BP, heart rate, temperature |
| `mh.csv` | MH – Medical History | Recommended | Prior comorbidities (diabetes, hypertension, etc.) |

## How to Generate the Pipeline-Ready CSV

Once you have placed the raw domain files, run the data adapter:

```bash
python src/data_adapter.py \
    --ae  data/raw/ae.csv  \
    --dm  data/raw/dm.csv  \
    --lb  data/raw/lb.csv  \
    --vs  data/raw/vs.csv  \
    --mh  data/raw/mh.csv  \
    --out data/raw/clinical_trial_data.csv
```

The resulting `clinical_trial_data.csv` can then be passed directly to the main pipeline:

```bash
python src/pipeline.py \
    --data data/raw/clinical_trial_data.csv \
    --model all \
    --explain
```

## Data Sources

See [`docs/data_sources.md`](../../docs/data_sources.md) for a full list of real-world SAE data sources including:

- PhysioNet / MIMIC-IV
- FAERS (FDA Adverse Event Reporting System)
- EudraVigilance (EMA)
- CDISC Pilot Study Dataset (public, no registration required)
- ClinicalTrials.gov results data

## Privacy & Ethics

> ⚠️ Real clinical data is sensitive personal data under GDPR / HIPAA. Before placing any files here:
>
> 1. Obtain approval from an Ethics Committee / IRB.
> 2. Sign a Data Use Agreement (DUA) with the data owner.
> 3. Anonymise or pseudonymise the data before use.
> 4. **Never commit** patient-level data to version control.
