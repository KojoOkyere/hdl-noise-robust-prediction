# Robust Prediction of HDL Cholesterol Under Noise-Perturbed Outcomes
### NHANES Application — Prediction Track (Graduate)

[![R](https://img.shields.io/badge/Made%20with-R-blue.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducible](https://img.shields.io/badge/Reproducible-Yes-success.svg)](#reproducibility)

---

## Overview

This repository contains the complete analytical pipeline for the competition submission:

**"Robust Prediction of HDL Cholesterol Under Noise-Perturbed Outcomes: An NHANES Application"**

The project evaluates the robustness, stability, and predictive accuracy of multiple statistical and machine learning models under controlled outcome perturbation.

Our analysis goes beyond leaderboard optimization by emphasizing:

- Predictive robustness
- Feature stability
- Interpretability
- Reproducibility

under realistic measurement noise.

---

## Competition Track

**Track:** Prediction Track (Graduate)  
**Task:** Predict `LBDHDD_outcome` for the test dataset  
**Primary Metric:** Root Mean Squared Error (RMSE)

Secondary evaluation emphasizes report quality, clarity, and rigor.

---

## Data Description

- Source: NHANES (CDC Public Use Files)
- Outcome: `LBDHDD_outcome` (HDL cholesterol, mg/dL)
- Sample Size: ~1,200
- Predictors: 97 demographic, dietary, and anthropometric variables

---

## Preprocessing Pipeline

1. Missing values:
   - Continuous → Median imputation
   - Categorical → Mode imputation

2. Feature encoding:
   - One-hot encoding for categorical variables

3. Scaling:
   - Standardization (mean 0, variance 1)

4. Feature engineering:
   - Removal of near-zero variance predictors
   - Consistent encoding across train/test sets

---

## Outcome Noise Design

To evaluate robustness under realistic measurement error and privacy-style perturbations, synthetic Gaussian noise was added to the outcome variable.

For each noise level σ:
Y(σ) = Y + ε, where ε ~ N(0, σ²)

Noise levels considered:
σ ∈ {0, 0.5, 1, 2, 3, 5}


This design enables systematic analysis of performance degradation under increasing outcome distortion.

---

## Models Evaluated

The following models were implemented and compared:

| Category | Models |
|----------|---------|
| Linear | OLS |
| Regularized | Ridge, Lasso, Elastic Net |
| Ensemble | Random Forest, XGBoost |

Key libraries:
- `glmnet`
- `randomForest`
- `xgboost`
- `tidymodels`

---

## Validation & Tuning Strategy

- Repeated K-fold cross-validation
- Nested tuning at baseline (σ = 0)
- Fixed hyperparameters across noise levels
- Out-of-fold (OOF) evaluation

### Performance Metrics

| Metric | Purpose |
|--------|----------|
| RMSE | Primary ranking metric |
| MAE | Robust error measure |
| R² | Explained variance |

---

## Feature Stability Analysis

Elastic Net models were evaluated using bootstrap resampling.

Stability metrics include:

- Selection frequency
- Sign consistency
- Coefficient variability
- Aggregated stability score

This identifies predictors with consistent associations under perturbation.

---

## Final Model

Based on predictive accuracy and robustness:

**Selected Model:** XGBoost

Rationale:
- Lowest baseline RMSE
- Most stable degradation
- Strong nonlinear modeling capacity

---

## Repository Structure
hdl-noise-robust-prediction/
│
├── analysis/
│ └── main_pipeline.R
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── output/
│ └── pred.csv
│
├── figures/
│ └── Fig1.png
│
├── tables/
│ └── performance_metrics.csv
│
└── README.md


---

## How to Reproduce Results
### Clone Repository

```bash
git clone https://github.com/KojoOkyere/hdl-noise-robust-prediction.git
cd hdl-noise-robust-prediction

## Install Dependencies
In R:
install.packages(c(
  "tidyverse", "glmnet", "randomForest",
  "xgboost", "caret", "rsample", "yardstick"
))

## Run Main Pipeline
From project root:
```r
source("main_pipeline.R")
```
This script performs:
- Data preprocessing
- Cross-validation
- Noise perturbation
- Model training Evaluation
- Test prediction generation

## Outputs
Results are saved in:
output/   → pred.csv
figures/  → figures
tables/   → performance tables

## Submission File
The final submission file:
output/pred.csv

Format:
Exactly one column: pred
Row order matches test set

## Reproducibility Statement
All analyses were conducted using a fully reproducible workflow with fixed random seeds and documented dependencies. The complete data preprocessing, model training, validation, and prediction pipeline is publicly available in this repository, enabling independent verification and replication of results.

## Citation
If you use this work, please cite:
Citation
Okyere, F. (2026). Robust Prediction of HDL Cholesterol Under Noise-Perturbed Outcomes: An NHANES Application.
Competition Submission.

## Author
Francis Okyere
M.S. Statistics (Applied Statistics)
Florida State University
Email: fokyere@fsu.edu

## License
This project is licensed under the MIT License.
