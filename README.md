# XAI Heart-Disease Risk Prediction

Explainable AI pipeline for heart-disease risk prediction using the **MIMIC-III** clinical database.  
Trains three classifiers and uses **SHAP** to explain which features drive each prediction.

---

## Pipeline overview

```
PostgreSQL (MIMIC-III)
        │
        ▼
extract_dataset.py   →  data/heart_disease_dataset.csv   (18,776 patients × 35 features)
        │
        ▼
train_models.py      →  outputs/best_model.joblib        (Gradient Boosting, AUC = 0.852)
        │
        ▼
shap_analysis.py     →  outputs/figures/shap_summary.png
                         outputs/tables/shap_top15_features.csv
```

Run everything in one command:
```bash
python src/run_pipeline.py
```

---

## Features extracted from MIMIC-III

| Category | Features |
|---|---|
| Demographics | age, gender, ethnicity, admission type |
| Vitals (first 24 h) | heart rate, SBP, DBP, MBP, SpO2, resp. rate, temperature |
| Labs (first 24 h) | creatinine, BUN, sodium, potassium, chloride, bicarbonate, calcium, glucose, platelet, WBC, hemoglobin, hematocrit, lactate |
| Stay info | ICU LOS, hospital LOS, first care unit |
| Target | `heart_disease` — ICD-9 codes 410–414, 427, 428 |

---

## Models & results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.77 | 0.838 |
| Random Forest | 0.80 | 0.852 |
| **Gradient Boosting** ✓ | **0.80** | **0.852** |

Best model saved to `outputs/best_model.joblib`.

---

## Top 15 features (SHAP)

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | age | 0.793 |
| 2 | first_careunit | 0.696 |
| 3 | sbp | 0.187 |
| 4 | creatinine | 0.173 |
| 5 | bun | 0.142 |
| 6 | hosp_los_days | 0.117 |
| 7 | icu_los_days | 0.117 |
| 8 | resp_rate | 0.092 |
| 9 | bicarbonate | 0.077 |
| 10 | gender | 0.066 |

---

## Project structure

```
XAI/
├── sql/
│   └── create_heart_disease_dataset.sql   # Builds the ML dataset from MIMIC-III
├── src/
│   ├── db_connection.py                   # PostgreSQL connection (SQLAlchemy + dotenv)
│   ├── extract_dataset.py                 # Runs SQL, exports CSV
│   ├── train_models.py                    # Trains & evaluates 3 classifiers
│   ├── shap_analysis.py                   # SHAP explainability
│   └── run_pipeline.py                    # Runs all steps in order
├── .env                                   # DB credentials (not committed)
├── .gitignore
└── requirements.txt
```

---

## Setup

### 1. Prerequisites
- Python 3.10+
- PostgreSQL 16 with MIMIC-III loaded
- `psql` on your system PATH

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
Create a `.env` file in the project root:
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
DB_SCHEMA=public
```

### 4. Run
```bash
python src/run_pipeline.py
```

Or run individual steps:
```bash
python src/extract_dataset.py    # extract data from PostgreSQL
python src/train_models.py       # train models
python src/shap_analysis.py      # run SHAP analysis
```

---

## Data access

MIMIC-III is a restricted-access dataset. To use this project you must:
1. Complete the [CITI Program training](https://physionet.org/about/citi-course/)
2. Apply for access at [physionet.org/content/mimiciii](https://physionet.org/content/mimiciii/)

Raw data and model artefacts are excluded from this repository (see `.gitignore`).

---

## Tech stack

`Python` · `PostgreSQL` · `SQLAlchemy` · `pandas` · `scikit-learn` · `SHAP` · `matplotlib`
