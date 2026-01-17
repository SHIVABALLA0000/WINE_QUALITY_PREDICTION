#PROBLEM STATEMENT
Wine quality is traditionally evaluated by domain experts, which is costly, time‑consuming, and often inconsistent. 
This project builds a production‑ready machine learning system to predict wine quality using physicochemical properties, enabling scalable and reproducible quality assessment.
The solution integrates SQL-based data ingestion, robust preprocessing, statistically sound evaluation, and Bayesian hyperparameter optimization to ensure reliable performance, even under severe class imbalance.

#DATA_SET
red_wine and white_wine two data sets
column space/features/dimensions
1.fixed_acidity:it is a non-volatile which implies doesnot evapourate and gives sourness to wine.
2.volatile_acidity:acids that evapourate mainly acetic acid.
3.citric_acid:a organic acid found in small amounts which adds freshness to wine.
4.residual_sugar:sugar remaining after fermentation.which decides dry wine or sweet wine.
5.chlorides:amount of salts .
6.free_sulfur_dioxide:free so2 which prevents bacteria and oxidation.
7.total_sulfur_dioxide:free so2+bound s02.
8.density:mass per unit volume of wine.
9.ph:measures acidity/alkalinity of wine.
10.sulphates:acts as antimicrobial and enhances flavor intensity.
11.alcohol:alcohol percentage in wine.
12.quality:target coulmn/output variable.

#DEPENDENCIES
numpy
pandas
scikit-learn
lightgbm
xgboost
joblib


#ARCHITECTURE
┌──────────────┐
│  CSV / DB    │
│ (raw data)   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ data_utils.py    │
│ Load & validate  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ preprocess.py    │
│ Feature pipeline │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ model.py         │
│ Model factory    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ train.py         │
│ Nested CV +      │
│ Optuna + Train   │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ wine_artifacts/  │
│ Saved models &   │
│ reports          │
└──────────────────┘

#FILE_STRUCTURE

WINE_QUALITY_PREDICTION/
│
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│
├── data_analysis/
│   ├── wine_quality.db
│   ├── wine_sql.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── preprocess.py
│   ├── model.py
│   ├── metrics.py
│   ├── train.py
│
├── wine_artifacts/
│   ├── wine_quality_model.joblib
│   ├── label_encoder.joblib
│   ├── nestedcv_report.json
│   ├── model_card.json
│
├── venv/
│
├── wine_quality_prediction.egg-info/
│
├── run_train.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore


 a)src/__init__.py:Marks src as a python package.
 b)src/config.py:For a fixed trails and no randomnesss.
 c)src/data_utils.py:Load CSV,separate features & target.
 d)src/preprocess.py:pre-processing the data using advanced techinques in a pipeline and column tranformer.
 e)src/model.py:Establishing a different models to select dynamically one by one.
 f)src/metrics.py:For checking model performance by using different metrics based on constraints.
 g)src/tuning.py:Hyperparameter search space where different parameters are given using optuna.
 h)src/train.py:It is a nested k-fold  cross-validation where it main block of the training the data.
 g)run_train.py:It is a entry point where the whole code is run through this file.
 h)setup.py:It creates src as package and can be used in other files and th whole project can downloaded.
 
#IMPLEMENTATION AND DESIGN
A)Data Ingestion & Cleaning
Raw CSV data is loaded into an SQLite database.
SQL queries are used for validation, consistency checks, and exploratory cleaning.
Cleaned data is passed to Pandas‑based preprocessing pipelines.
B)Modeling Strategy
Treated as a multiclass classification problem
Implemented a model factory to dynamically switch between models.
C)Class Imbalance Handling
Severe imbalance handled using class‑weighted loss functions
Synthetic oversampling (SMOTE) intentionally avoided to prevent overfitting on extremely rare classes
D)Hyperparameter Optimization
Optuna (TPE sampler) used for Bayesian hyperparameter optimization
Optimization performed inside cross‑validation loops to prevent data leakage
D)Evaluation Methodology
Stratified K‑Fold Cross‑Validation to preserve class distribution
F1‑macro selected as the primary metric to ensure equal importance to rare and majority classes
E)Artifacts & Reproducibility
Trained model and label encoder are serialized using Joblib
Cross‑validation results and configuration stored as JSON reports
Fixed random seeds ensure reproducibility across runs
