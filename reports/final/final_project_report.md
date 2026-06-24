# Final Project Report: Credit Card Fraud Detection

## Executive Summary

This project is an end-to-end fraud detection ML pipeline for a highly imbalanced transaction dataset. It covers data validation, EDA, leakage-safe preprocessing, baseline models, XGBoost, validation-only threshold tuning, SHAP explainability, and one locked final test-set evaluation.

## Locked Model And Threshold

- Champion model: `xgboost_baseline`
- Threshold source: Day 6 validation recall-target selection
- Locked business threshold: `0.53`

## Final Test Evaluation

| Metric | Value |
|---|---:|
| PR-AUC | 0.8287848539773868 |
| ROC-AUC | 0.9613432016710013 |
| Precision | 0.6966292134831461 |
| Recall | 0.8378378378378378 |
| F1-score | 0.7607361963190185 |
| True positives | 62 |
| False positives | 27 |
| False negatives | 12 |
| True negatives | 42621 |

## Integrity Notes

- The model was selected using validation metrics only.
- The operating threshold was selected using validation data only.
- The test split was used once for final locked evaluation.
- SHAP was used for explanation only, not tuning or feature selection.

## Key Outputs

- `reports/explainability/shap_summary_report.md`
- `reports/final/final_evaluation_report.md`
- `reports/final/final_model_evaluation.json`
- `reports/final/project_audit_checklist.md`
