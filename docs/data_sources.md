# Źródła Danych dla Predykcji SAE / Data Sources for SAE Prediction

## Skąd wziąć dane? / Where to Get Data?

Projekt używa **syntetycznych danych** do prototypowania. Poniżej opisano rzeczywiste źródła danych klinicznych, które można podłączyć do pipeline'u po podpisaniu odpowiednich umów i uzyskaniu zgód etycznych.

The project ships with **synthetic data** for prototyping. Below are real-world clinical data sources that can be plugged into the pipeline after signing the appropriate data-sharing agreements and obtaining ethical approvals.

---

## 1. Dane syntetyczne (dostępne od razu) / Synthetic Data (Available Immediately)

| Źródło | Opis | Dostęp |
|--------|------|--------|
| `data/synthetic/generate_synthetic.py` | 2 000 syntetycznych pacjentów, ~15% SAE | Brak wymagań – uruchom skrypt |

```bash
python data/synthetic/generate_synthetic.py
```

---

## 2. Publiczne bazy danych klinicznych / Public Clinical Databases

### 2a. PhysioNet / MIMIC
- **URL**: <https://physionet.org/content/mimiciv/> 
- **Opis**: Dane ICU ~40 000 pacjentów (MIMIC-IV). Zawiera kody ICD, wyniki laboratoryjne, dane życiowe, leki.
- **Format**: PostgreSQL / CSV (tabele w schemacie OMOP/własnym)
- **Dostęp**: Rejestracja + kurs CITI (bezpłatne, ~4 h)
- **Zmienne SAE**: `diagnoses_icd` (ICD-9/10 kodujące zdarzenia niepożądane), tabela `admissions`

### 2b. eICU Collaborative Research Database
- **URL**: <https://physionet.org/content/eicu-crd/>
- **Opis**: ~200 000 pobytów na OIT z 208 szpitali USA
- **Format**: CSV / PostgreSQL
- **Dostęp**: Rejestracja PhysioNet + kurs CITI

### 2c. FAERS (FDA Adverse Event Reporting System)
- **URL**: <https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers>
- **Opis**: Raporty zdarzeń niepożądanych leków w USA (post-market)
- **Format**: CSV/ASCII (kwartalne pliki), ~20 mln rekordów
- **Dostęp**: Publiczny – pobierz bezpośrednio

### 2d. EudraVigilance (EMA)
- **URL**: <https://www.ema.europa.eu/en/human-regulatory/research-development/pharmacovigilance/eudravigilance>
- **Opis**: Europejski odpowiednik FAERS; dane post-market z EU/EEA
- **Format**: XML (ICSR E2B(R3))
- **Dostęp**: Publiczny portal; pełny dostęp dla sponsorów i regulatorów po rejestracji

### 2e. OpenFDA
- **URL**: <https://open.fda.gov/apis/drug/event/>
- **Opis**: REST API do FAERS; ~10 mln zdarzeń niepożądanych
- **Format**: JSON via REST API
- **Dostęp**: Publiczny (klucz API opcjonalny)

---

## 3. Dane z badań klinicznych (CDISC SDTM) / Clinical Trial Data (CDISC SDTM)

Sponsor badania klinicznego przesyła dane do FDA/EMA w formacie **CDISC SDTM** (Study Data Tabulation Model). Najważniejsze domeny do predykcji SAE:

| Domain | Zawartość | Tabela SDTM |
|--------|-----------|-------------|
| DM | Dane demograficzne | `dm.xpt` |
| LB | Wyniki laboratoryjne | `lb.xpt` |
| VS | Znaki życiowe | `vs.xpt` |
| MH | Historia medyczna | `mh.xpt` |
| AE | Zdarzenia niepożądane (w tym SAE) | `ae.xpt` |
| CM | Leki towarzyszące | `cm.xpt` |
| EX | Ekspozycja na lek | `ex.xpt` |

### Jak załadować dane SDTM do pipeline'u?

```bash
# 1. Eksportuj domenę AE i DM z SAS Transport Files (.xpt) do CSV:
python -c "
import pandas as pd
ae = pd.read_sas('ae.xpt', format='xport', encoding='utf-8')
dm = pd.read_sas('dm.xpt', format='xport', encoding='utf-8')
ae.to_csv('data/raw/ae.csv', index=False)
dm.to_csv('data/raw/dm.csv', index=False)
"

# 2. Użyj adaptera SDTM -> pipeline:
python src/data_adapter.py \
    --ae data/raw/ae.csv \
    --dm data/raw/dm.csv \
    --lb data/raw/lb.csv \
    --vs data/raw/vs.csv \
    --mh data/raw/mh.csv \
    --out data/raw/clinical_trial_data.csv

# 3. Uruchom pipeline z prawdziwymi danymi:
python src/pipeline.py --data data/raw/clinical_trial_data.csv --model all --explain
```

### Publiczne zestawy SDTM (pilotaż)
- **CDISC Pilot Study Dataset** – <https://github.com/cdisc-org/sdtm-adam-pilot-project>
- Zawiera domeny: DM, AE, LB, VS, MH, CM, EX
- Używany przez CDISC do testowania narzędzi

---

## 4. Polskie i europejskie rejestry badań klinicznych

| Rejestr | URL | Uwagi |
|---------|-----|-------|
| ClinicalTrials.gov (NIH) | <https://clinicaltrials.gov/> | Dane wyników po zakończeniu badania (`results_section`) |
| EU Clinical Trials Register | <https://www.clinicaltrialsregister.eu/> | Dane unijne |
| POLCRO / GCPpl | <https://gcppl.pl/> | Polskie badania kliniczne |

---

## 5. Format danych wymagany przez pipeline

Dane wejściowe muszą być w formacie CSV z kolumnami opisanymi w sekcji `features` pliku `configs/config.yaml`. Adapter (`src/data_adapter.py`) automatycznie mapuje kolumny SDTM na wymagany schemat.

Minimalne wymagane kolumny:

```
age, sex, weight, height, bmi, alt, ast, creatinine, hemoglobin, platelets,
wbc, systolic_bp, diastolic_bp, heart_rate, temperature, diabetes,
hypertension, renal_impairment, hepatic_impairment, prior_sae,
cardiac_history, treatment_arm, study_phase, visit_number, dose_level,
comorbidity_score, sae_occurred
```

---

## 6. Aspekty etyczne i prawne / Ethics & Legal

> ⚠️ Dane kliniczne są wrażliwe. Przed ich użyciem należy:
>
> 1. Uzyskać zgodę Komisji Bioetycznej / IRB
> 2. Podpisać **Data Use Agreement (DUA)** z właścicielem danych
> 3. Zanonimizować/pseudoanomizować dane (RODO / HIPAA / GDPR)
> 4. Przechowywać dane lokalnie – **nie commituj** do repozytorium

W ramach projektu **Ścieżki SMART** (patrz `docs/smart_project.md`) dane syntetyczne są wystarczające do wykazania niepewności badawczej i osiągnięcia KPI I fazy projektu.
