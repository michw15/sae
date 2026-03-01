# Projekt w ramach Ścieżki SMART (NCBiR / FENG 2021–2027)

## Informacje ogólne

| Pozycja | Wartość |
|---------|---------|
| **Program** | Fundusze Europejskie dla Nowoczesnej Gospodarki (FENG) 2021–2027 |
| **Działanie** | 1.1 Ścieżka SMART |
| **Instytucja Pośrednicząca** | NCBiR (Narodowe Centrum Badań i Rozwoju) |
| **Typ projektu** | Prace B+R (moduł obowiązkowy) + wdrożenie (opcjonalnie) |
| **Tytuł projektu** | Predykcja Poważnych Zdarzeń Niepożądanych (SAE) w Badaniach Klinicznych z Wykorzystaniem Uczenia Maszynowego |

---

## 1. Uzasadnienie badawcze (niepewność badawcza)

> Kryterium kwalifikowalności w Ścieżce SMART: projekt musi zawierać **prace badawczo-rozwojowe** o charakterze eksperymentalnym i eliminować **niepewność naukową/techniczną**.

### Zdefiniowana niepewność badawcza

**Pytanie badawcze**: Czy modele uczenia maszynowego trenowane na rutynowo zbieranych danych z badań klinicznych (dane demograficzne, wyniki laboratoryjne, znaki życiowe, historia medyczna) mogą przewidywać ryzyko wystąpienia SAE na poziomie pacjenta *przed* jego wystąpieniem z wystarczającą czułością (≥ 80%) i precyzją AUC-ROC ≥ 0,85, umożliwiając proaktywne interwencje bezpieczeństwa?

**Dlaczego istnieje niepewność**:
1. Brak sprawdzonej, ogólnodostępnej metody predykcji SAE w czasie rzeczywistym w fazie klinicznej
2. Wysoka heterogeniczność danych klinicznych (różne protokoły, populacje, fazy badań)
3. Silna nierównowaga klas (~15% zdarzeń) utrudnia standardowe podejścia ML
4. Brak walidacji krzyżowej między obszarami terapeutycznymi (onkologia vs kardiologia vs neurologia)

---

## 2. Cele projektu i KPI

### Cel główny
Opracowanie i walidacja modelu predykcyjnego SAE osiągającego AUC-ROC ≥ 0,85 na niezależnym zbiorze testowym.

### Wskaźniki rezultatu (KPI)

| KPI | Wartość docelowa | Metoda pomiaru |
|-----|-----------------|----------------|
| AUC-ROC najlepszego modelu | ≥ 0,85 | `compare_models()` w `src/evaluation.py` |
| Czułość (sensitivity) | ≥ 0,75 | Optymalny próg Youdena |
| Czas predykcji na pacjenta | < 100 ms | Benchmark na CPU |
| Pokrycie XAI (odsetek wyjaśnionych predykcji) | 100% | SHAP global + LIME local |
| Liczba walidowanych modeli | ≥ 4 | LR, RF, XGBoost, LightGBM |

---

## 3. Plan prac B+R (harmonogram)

### Etap I – Prace badawcze (miesiące 1–12)

| Zadanie | Opis | Plik |
|---------|------|------|
| T1.1 | Przegląd literatury i analiza dostępnych zbiorów danych SAE | `docs/data_sources.md` |
| T1.2 | Opracowanie syntetycznego datasetu referencyjnego | `data/synthetic/generate_synthetic.py` |
| T1.3 | Inżynieria cech i preprocessing | `src/preprocessing.py` |
| T1.4 | Trening i tuning modeli ML | `src/models.py` |
| T1.5 | Ewaluacja i benchmarking | `src/evaluation.py` |
| T1.6 | Wyjaśnialność (XAI) – SHAP i LIME | `src/explainability.py` |

### Etap II – Walidacja na danych rzeczywistych (miesiące 13–24)

| Zadanie | Opis |
|---------|------|
| T2.1 | Pozyskanie danych CDISC SDTM od partnera przemysłowego (CRO/sponsor) |
| T2.2 | Adaptacja pipeline'u do formatu SDTM (`src/data_adapter.py`) |
| T2.3 | Walidacja zewnętrzna na ≥ 2 niezależnych zbiorach danych |
| T2.4 | Analiza prospektywna (jeśli możliwa) |

### Etap III – Wdrożenie (opcjonalny moduł SMART, miesiące 25–36)

| Zadanie | Opis |
|---------|------|
| T3.1 | Opracowanie API REST do predykcji (FastAPI/Flask) |
| T3.2 | Integracja z systemem EDC (Electronic Data Capture) |
| T3.3 | Pilot u partnera przemysłowego |
| T3.4 | Walidacja kliniczna i dokumentacja regulacyjna (FDA 21 CFR Part 11) |

---

## 4. Dane wejściowe projektu

> Szczegóły – patrz `docs/data_sources.md`

- **Faza I (B+R)**: Dane syntetyczne generowane przez `data/synthetic/generate_synthetic.py`
- **Faza II (walidacja)**: Dane CDISC SDTM od partnera lub z publicznych repozytoriów (CDISC Pilot Dataset, PhysioNet MIMIC-IV)
- **Dane zastrzeżone**: Przechowywane poza repozytorium w `data/raw/` (patrz `.gitignore`)

---

## 5. Metodologia (zgodność z ICH E6 R2 / GCP)

```
Dane kliniczne (CDISC SDTM)
        │
        ▼
src/data_adapter.py   ← mapowanie SDTM → schemat pipeline'u
        │
        ▼
src/preprocessing.py  ← imputacja, outlier capping, feature engineering
        │
        ▼
src/models.py         ← LR / RF / XGBoost / LightGBM + SMOTE
        │
        ▼
src/evaluation.py     ← AUC-ROC, PR-AUC, kalibracja, Brier score
        │
        ▼
src/explainability.py ← SHAP (global) + LIME (local)
        │
        ▼
reports/              ← wykresy, tabele, model_comparison.csv
```

---

## 6. Kwestie regulacyjne i etyczne

| Aspekt | Wymaganie | Status |
|--------|-----------|--------|
| Ochrona danych (RODO) | Dane syntetyczne nie wymagają zgody | ✅ Spełnione (faza I) |
| Komisja Bioetyczna | Wymagana dla danych rzeczywistych | 🔲 Faza II |
| ICH E6(R2) GCP | Pipeline zgodny z zasadami GCP | ✅ |
| FDA 21 CFR Part 11 | Wymagane dla systemów decyzyjnych | 🔲 Faza III |
| ISO 13485 / MDR | Jeśli narzędzie klasyfikowane jako SaMD | 🔲 Opcjonalne |

---

## 7. Wkład własny i finansowanie

W ramach Ścieżki SMART dofinansowanie B+R wynosi do **80% kosztów kwalifikowalnych** dla prac przemysłowych. Wkład własny: min. 20%.

Kosztami kwalifikowalnymi są m.in.:
- Wynagrodzenia zespołu B+R (data scientists, biostatystycy, lekarze)
- Licencje oprogramowania (jeśli wymagane)
- Usługi doradcze w zakresie regulacyjnym
- Koszty pozyskania danych (DUA, dostęp do baz danych)
- Infrastruktura obliczeniowa (serwery GPU/cloud)

---

## 8. Referencje

- [Ścieżka SMART – NCBiR](https://www.ncbir.pl/fundusze-europejskie/feng/sciezka-smart/)
- [FENG 2021–2027 – PARP](https://www.parp.gov.pl/feng)
- ICH E6(R2) Good Clinical Practice Guideline
- FDA Guidance on AI/ML-Based Software as a Medical Device (SaMD)
- CDISC SDTM Implementation Guide v3.3
