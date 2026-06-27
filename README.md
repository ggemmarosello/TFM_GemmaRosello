# A Data-Driven Framework for Chronicity and Mortality Risk Prediction

````markdown
# A Data-Driven Framework for Chronicity and Mortality Risk Prediction

**Master's Thesis – MSc in Fundamental Principles of Data Science**  
**Universitat de Barcelona**

**Author:** Gemma Roselló Fontanals

---

## Overview

This repository contains the code developed for my Master's Thesis:

> **A Data-Driven Framework for Chronicity and Mortality Risk Prediction**

The project develops and evaluates machine learning models for predicting **12-month mortality** in patients aged 70 years and older using routinely collected Electronic Health Records (EHRs). It also investigates the relationship between predicted mortality risk and chronicity (PCC and MACA) in order to assess whether mortality prediction can complement existing chronic patient identification strategies.

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
├── README.md
└── requirements.txt
```

---

## Workflow

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

- patient identifier harmonization
- mortality data cleaning
- duplicate resolution
- chronicity preprocessing
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
- baseline model comparison
- hyperparameter optimization
- probability calibration
- class imbalance experiments
- threshold optimization
- final model evaluation
- SHAP explainability
- chronicity analyses

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
- dataframe preprocessing
- column cleaning

### src/utils_modeling.py

Utility functions for:

- stratified cross-validation
- randomized hyperparameter search
- threshold optimization
- model evaluation
- ROC and Precision–Recall curves
- probability calibration
- calibration curves

---

## Dataset

The Electronic Health Records (EHRs) used in this work were provided by the **Institut Català de la Salut (ICS) Camp de Tarragona**.

The data contain sensitive patient information and **cannot be shared publicly**. Consequently, only the source code is included in this repository.

---

## Main Results

The best-performing model was an **Isotonic-Calibrated LightGBM**, achieving:

| Metric | Value |
|---------|------:|
| PR-AUC | **0.720** |
| ROC-AUC | **0.970** |

The model captured clinically meaningful patterns associated with multimorbidity, frailty, and healthcare complexity while showing a strong relationship between predicted mortality risk and chronicity labels.

---

## Requirements

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

Main libraries include:

- pandas
- numpy
- scikit-learn
- LightGBM
- XGBoost
- CatBoost
- imbalanced-learn
- SHAP
- matplotlib

---

## Running the Project

Execute the notebooks in the following order:

1. `notebooks/01_data_engineering_patient_level.ipynb`
2. `notebooks/02_data_engineering_healthcare.ipynb`
3. `notebooks/03_modeling_pipeline.ipynb`

---

## Citation

If you use this repository, please cite:

```text
Roselló Fontanals, G.
A Data-Driven Framework for Chronicity and Mortality Risk Prediction.
MSc Thesis, Universitat de Barcelona, 2026.
```

---

## License

This repository is intended for academic and research purposes.

The healthcare data used in this work are not publicly available due to ethical and privacy restrictions.
````



