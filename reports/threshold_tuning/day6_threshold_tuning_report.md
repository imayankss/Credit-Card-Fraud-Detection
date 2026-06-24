# Day 6 Threshold Tuning Report: Credit Card Fraud Detection

**Generated:** 2026-06-24 16:52:40
**Model:** xgboost_baseline

## Validation Dataset

- Validation rows: 42721
- Validation fraud cases: 74

## Why Threshold Tuning Matters

The default classification threshold of 0.50 is rarely optimal for highly imbalanced fraud detection problems. Lowering the threshold tends to catch more fraud (higher recall) at the cost of more false alerts (lower precision). Raising the threshold reduces false alerts but risks missing more fraud. Accuracy is not used to choose a threshold here because the fraud class is rare; precision, recall, F1-score, and the confusion matrix are far more informative.

## Default Threshold (0.50)

### Default Threshold Performance

- Threshold: 0.50
- Precision: 0.6061
- Recall: 0.8108
- F1-score: 0.6936
- True Positives (fraud caught): 60
- False Positives (false alerts): 39
- False Negatives (fraud missed): 14
- True Negatives: 42608

## Best F1 Threshold

### Best F1 Threshold Performance

- Threshold: 0.98
- Precision: 0.9138
- Recall: 0.7162
- F1-score: 0.8030
- True Positives (fraud caught): 53.0
- False Positives (false alerts): 5.0
- False Negatives (fraud missed): 21.0
- True Negatives: 42642.0

## Recall-Target Threshold (recall >= 0.80)

### Recall-Target Threshold Performance

- Threshold: 0.53
- Precision: 0.6250
- Recall: 0.8108
- F1-score: 0.7059
- True Positives (fraud caught): 60.0
- False Positives (false alerts): 36.0
- False Negatives (fraud missed): 14.0
- True Negatives: 42611.0

## Recommended Operating Threshold

The recommended operating threshold is **0.53** (the recall-target threshold). In fraud detection, missing a fraudulent transaction (a false negative) is typically far more costly than reviewing a legitimate transaction that was flagged (a false positive). This threshold prioritizes catching at least 80% of fraud while keeping precision as high as possible at that recall level. The final choice should still be confirmed against business cost analysis.

## Generated Artifacts

- reports/threshold_tuning/threshold_metrics.csv
- reports/threshold_tuning/selected_thresholds.json
- reports/figures/precision_recall_curve.png
- reports/figures/roc_curve.png
- reports/figures/confusion_matrix_default_threshold.png
- reports/figures/confusion_matrix_best_f1_threshold.png
- reports/figures/confusion_matrix_recall_target_threshold.png

## Note on Data Usage

Only the validation set was used for this analysis. The held-out test set was not loaded, referenced, or evaluated at any point in this Day 6 pipeline. The test set remains untouched for the final evaluation stage.
