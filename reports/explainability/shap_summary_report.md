# SHAP Feature Importance Summary

## Purpose

This report explains which features most influenced the champion XGBoost model's fraud predictions, based on mean absolute SHAP values calculated on a sample of validation data.

## Important Limitation

Because `V1` to `V28` are anonymized PCA-transformed features, SHAP values explain model behavior in terms of these transformed components. They do not map directly to real-world transaction attributes such as merchant, location, or card type.

## Top 20 Features by SHAP Importance

| Rank | Feature | Mean Absolute SHAP Value |
|---:|---|---:|
| 1 | V4 | 1.914230 |
| 2 | V14 | 1.808449 |
| 3 | V12 | 0.840523 |
| 4 | V10 | 0.603617 |
| 5 | V3 | 0.506720 |
| 6 | V11 | 0.377626 |
| 7 | V26 | 0.320547 |
| 8 | V16 | 0.318310 |
| 9 | Amount | 0.313929 |
| 10 | V8 | 0.304865 |
| 11 | Time | 0.291748 |
| 12 | V15 | 0.291153 |
| 13 | V7 | 0.287668 |
| 14 | V28 | 0.279879 |
| 15 | V21 | 0.279400 |
| 16 | V24 | 0.267114 |
| 17 | V25 | 0.258173 |
| 18 | V18 | 0.254940 |
| 19 | V19 | 0.249909 |
| 20 | V20 | 0.210927 |

## How to Read This Report

- A higher mean absolute SHAP value means the feature has, on average, a larger impact on the model's predicted fraud probability across the explained sample.
- This ranking reflects model behavior only. It does not imply causation or a verified real-world explanation.

## Scope Notes

- SHAP values were calculated on a sample of validation data only, never on the test set used for final evaluation.
- SHAP was used strictly for explanation. It was not used to tune the model, select features, or change preprocessing.
