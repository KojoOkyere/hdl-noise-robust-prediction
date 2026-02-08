## Code Repository
Full reproducible code and outputs are available here:
https://github.com/KojoOkyere/hdl-noise-robust-prediction

# Robust Prediction of HDL Cholesterol Under Noise-Perturbed Outcomes

This repository contains the full code and reproducible pipeline for the graduate-track
submission to the ASASF/NHANES Prediction Challenge.

The project evaluates the robustness of statistical and machine learning models
for predicting HDL cholesterol under realistic outcome perturbations.

## Project Overview

High-density lipoprotein cholesterol (HDL-C) is an important biomarker for
cardiovascular risk. In population surveys such as NHANES, outcome measurements
may be affected by random noise arising from laboratory variability, reporting
errors, and privacy-preserving mechanisms.

This project investigates how different predictive models degrade under
increasing levels of synthetic outcome noise and identifies methods that remain
stable and reliable.

## Data

- Source: NHANES-derived ASASF dataset
- Training size: 1,000 observations
- Test size: 200 observations
- Predictors: Anthropometric, demographic, and dietary variables
- Outcome: `LBDHDD_outcome` (HDL cholesterol, mg/dL)

Note: Raw data are not redistributed due to competition policies.

## Methods

### Preprocessing
- Median imputation for continuous variables
- Mode imputation for categorical variables
- One-hot encoding
- Z-score normalization

### Models
- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- Elastic Net (α = 0.9)
- Random Forest
- XGBoost

### Noise Design
Synthetic Gaussian noise is added to the outcome:
```r
\[
Y^{(\sigma)} = Y + \epsilon, \quad \epsilon \sim N(0, \sigma^2)
\]

with σ ∈ {0, 0.5, 1, 2, 3, 5}.
```

### Validation
- Fixed 10-fold cross-validation
- Baseline tuning at σ = 0
- Out-of-fold evaluation under noise

### Metrics
- RMSE (primary competition metric)
- MAE
- R²

## Feature Stability

Elastic Net models are evaluated using bootstrap resampling to assess:

- Selection frequency
- Sign consistency
- Coefficient variability

A composite stability score is used to rank predictors.

## Final Model

XGBoost was selected as the final model based on:

- Best baseline performance
- Strong robustness under noise
- Consistent generalization

This model was trained on the full training set and used to generate test
predictions.

## Reproducibility and Usage

This repository contains the complete pipeline for data preprocessing,
model training, validation, and prediction generation.

### Requirements
- R (version ≥ 4.2)
- Required packages:
  tidyverse, glmnet, randomForest, xgboost, caret, Matrix

Install dependencies in R:

```r
install.packages(c(
  "tidyverse", "glmnet", "randomForest",
  "xgboost", "caret", "Matrix"
))
---

2. Run the main script:
```r
source("main_pipeline.R")
```

3. Outputs will be saved in:
- output/
- figures/
- tables/

## Submission File
The file pred.csv satisfies competition requirements:
- One column: pred
- Correct ordering
- Matches test set length
- 
## Reproducibility
- Fixed random seed: 2026
- Fixed cross-validation folds
- Fully scripted pipeline
- All results reproducible

## Citation
If you use this work, please cite:
Okyere, F. (2026). Robust Prediction of HDL Cholesterol Under Noise-Perturbed Outcomes.

## Author
Francis Okyere
Department of Statistics
Florida State University
Email: [kokyere.gh@gmail.com]

## License
This project is licensed under the MIT License.
