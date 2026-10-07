# Early Sepsis Detection: XGBoost vs Bayesian Network 🩺

*A comparison of two machine learning models for hourly sepsis prediction in ICU patients, with a prototype front-end dashboard.*

## Motivation ⚙️

### Chosen data-related problem

> **"Can we flag ICU patients at risk of sepsis early, and how does an interpretable Bayesian Network compare with a gradient boosting model?"**

Sepsis is a life-threatening condition where every hour of delay matters. This project predicts sepsis onset hour by hour, compares two very different approaches, and presents the output in a clinical-style early-warning dashboard.

### Chosen dataset

The dataset used was the PhysioNet/CinC 2019 Sepsis Challenge **training set A** (20,336 ICU patients, one record per hour), downloaded through the Kaggle API. The target variable is `SepsisLabel`, which is set 6 hours before clinical suspicion is recorded, so an accurate hourly prediction acts as an early-warning signal. Features include vital signs, laboratory values and demographics.

### Chosen machine learning algorithms

| Model | Why we chose it |
|---|---|
| **Bayesian Network** | Interpretable, shows probabilistic dependencies between clinical variables and gives calibrated risk estimates |
| **XGBoost** | Gradient boosting, often the strongest model on tabular data, with built-in handling of class imbalance |

Both models were trained on a **patient-level** 70/15/15 train, validation and test split (stratified on sepsis outcome) so that no patient appears in more than one set.

## Project Objectives 🎯

1. Clean the data without leakage (forward-fill only, implausible values removed).
2. Explore the data (EDA): class imbalance, distributions and differences between septic and non-septic hours.
3. Engineer trend features: hour-to-hour change and a rolling 6-hour mean.
4. Train and compare a Bayesian Network and an XGBoost model.
5. Evaluate with ROC-AUC, PR-AUC, precision, recall, calibration and lead time before diagnosis.
6. Present the results in a front-end early-warning dashboard.

## Environment 👩🏻‍💻

<p align="center">
  <img src="https://img.shields.io/badge/jupyter-F37626?style=flat&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white" alt="Google Colab"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" alt="GitHub"/>
</p>

## Stack 🛠️

<p align="center">
  <img src="https://img.shields.io/badge/python-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/matplotlib-11557C?style=flat" alt="matplotlib"/>
  <img src="https://img.shields.io/badge/seaborn-4C72B0?style=flat" alt="seaborn"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/pgmpy-4B8BBE?style=flat" alt="pgmpy"/>
  <img src="https://img.shields.io/badge/SHAP-FF0051?style=flat" alt="SHAP"/>
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white" alt="HTML"/>
</p>

## Repository Structure 🌲

```
.
├── .gitattributes
├── README.md
├── Sepsis_Organized.ipynb
├── XGBoost_Model.ipynb
└── sepsis_dashboard.html
```

- `Sepsis_Organized.ipynb`: Bayesian Network model
- `XGBoost_Model.ipynb`: XGBoost model
- `sepsis_dashboard.html`: front-end prototype (open in a browser)

## Method 🧪

| Stage | What we did |
|---|---|
| **Cleaning** | Removed duplicates, reset implausible vitals to missing, forward-filled within each patient (no back-fill, to avoid future leakage), dropped `EtCO2` |
| **EDA** | Class imbalance, descriptive statistics, standardised boxplots, correlation heatmap, mean by class |
| **Feature engineering** | Hour-to-hour change (`_delta`) and 6-hour rolling mean (`_roll6`), using past data only |
| **Preprocessing** | Training-set medians for remaining gaps; quartile binning (4 bins) for the Bayesian Network |
| **Bayesian Network** | Structure learned with Hill-Climb search (BIC), parameters fitted with a BDeu prior, inference by variable elimination |
| **XGBoost** | Class reweighting with `scale_pos_weight`, feature subsampling and early stopping on the validation set |
| **Thresholding** | Bayesian Network threshold chosen on the validation set, not the test set, to avoid leakage |
| **Explainability** | SHAP to explain individual predictions |

## Results 📊

| Model | Test ROC-AUC | Test PR-AUC | Precision | Recall |
|---|---|---|---|---|
| **Bayesian Network** | 0.646 | 0.037 | 0.044 | 0.286 |
| **XGBoost** | 0.804 | 0.107 | 0.123 | 0.353 |

Lead time on the 269 septic patients in the test set:

| Model | Septic patients flagged early | Mean warning before clinical onset |
|---|---|---|
| **Bayesian Network** | 126 (46.8%) | ~63.6 hours |
| **XGBoost** | 112 (41.6%) | ~57.9 hours |

- **XGBoost performed better** on ROC-AUC, PR-AUC, precision and recall.
- The Bayesian Network flagged slightly more patients early, but with many more false alarms.
- At the default 0.5 threshold the Bayesian Network predicted no sepsis cases at all, because sepsis is rare and its probabilities stay low. A lower threshold chosen on the validation set was needed.

## Main Findings 🔍

- **Sepsis is rare**: about 2% of hourly records are sepsis-positive, so accuracy alone is misleading.
- **Threshold choice matters**: choosing the threshold on the test set is a form of leakage, so it was fixed using the validation set.
- **Patient-level splitting** was essential to stop hours from the same patient appearing in both train and test sets.
- **Interpretability vs performance**: the Bayesian Network is easier to inspect, while XGBoost is more accurate.
- **Trend features** (change and rolling mean) give the models a sense of whether a patient is deteriorating.

## Recommendations for Improvements 📈

- **Hyperparameter tuning** and **cross-validation** for both models.
- **Continuous variables in the Bayesian Network** instead of discretising into 4 bins, which loses information.
- **Use training set B** or external data to check generalisation across hospitals.
- **Connect the dashboard** to the trained model so it shows live predictions.

## Reflection 🪞

This project showed how important careful evaluation is in healthcare, where rare events, data leakage and threshold choice can make results look much better or worse than they really are. Comparing an interpretable model with a more powerful one highlighted the trade-off between transparency and performance, and building the dashboard showed how model output could be made usable for clinicians. Next, we want to explore model tuning, richer network structures and connecting the front end to the live model.
