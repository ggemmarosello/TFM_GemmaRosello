# A Data-Driven Framework for Chronicity and Mortality Risk Prediction

**Master's Thesis – MSc in Fundamental Principles of Data Science**  
**Universitat de Barcelona**

**Author:** Gemma Roselló Fontanals

---

## Overview

This repository contains the complete data engineering and machine learning pipeline developed for my Master's Thesis. The project develops and evaluates machine learning models for predicting 12-month mortality in patients aged 70 years and older using routinely collected Electronic Health Records (EHRs). It also investigates the relationship between predicted mortality risk and chronicity (PCC and MACA) to assess whether mortality prediction can complement existing chronic patient identification strategies.

---

## Repository Structure

```text
TFM_GemmaRosello/
│
├── notebooks/
│   ├── 01_data_engineering_patient_level.ipynb
│   ├── 02_data_engineering_healthcare.ipynb
│   └── 03_modeling_pipeline.ipynb
│
├── src/
│   ├── utils.py
│   └── utils_modeling.py
│
└── README.md
```

---

## Pipeline

The repository follows the workflow illustrated below:

```text
Patient-level preprocessing
            │
            ▼
Healthcare feature engineering
            │
            ▼
Machine learning pipeline
```

### notebooks/01_data_engineering_patient_level.ipynb

Builds the study cohort and performs patient-level preprocessing, including:

- data extraction from the database
- patient identifier harmonization
- study cohort definition
- mortality data cleaning
- duplicate resolution
- data cleaning and preprocessing
- observation window generation
- demographic feature engineering

---

### notebooks/02_data_engineering_healthcare.ipynb

Extracts healthcare information from multiple data sources and aggregates longitudinal records into patient-level features.

Processed data include:

- laboratory tests
- diagnoses
- prescribed medications
- primary care visits
- hospitalizations
- emergency department visits

The resulting dataset is used as input for the machine learning models.

---

### notebooks/03_modeling_pipeline.ipynb

Implements the complete experimental pipeline, including:

- train/validation/test split
- hyperparameter optimization
- threshold optimization
- baseline model comparison
- probability calibration
- class imbalance experiments
- final model evaluation
- chronicity analyses
- SHAP explainability

Evaluated models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

---

## Source Code

### src/utils.py

Utility functions for:

- SQL data extraction
- parallel database queries
- patient chunk generation
- temporary SQL tables
- column names cleaning

### src/utils_modeling.py

Utility functions for:

- stratified cross-validation
- randomized hyperparameter search
- threshold optimization
- model evaluation
- ROC-AUC and PR-AUC curves
- probability calibration
- calibration curves

---

## Dataset

The Electronic Health Records (EHRs) used in this work were provided by the **Institut Català de la Salut (ICS) Camp de Tarragona**.

The study cohort includes patients aged **70 years and older**, with information extracted from routinely collected primary care records, including demographics, diagnoses, laboratory tests, prescribed medications, and healthcare utilization.

The data contain sensitive patient information and **cannot be shared publicly**. Consequently, this repository contains only the source code used to perform the analyses.

---

## Main Results

The best-performing model was an **Isotonic-Calibrated LightGBM**, achieving:

| Metric | Value |
|---------|------:|
| PR-AUC | **0.720** |
| ROC-AUC | **0.970** |

The model captured clinically meaningful patterns associated with multimorbidity, frailty, and healthcare complexity while showing well-calibrated mortality probabilitie.

---


