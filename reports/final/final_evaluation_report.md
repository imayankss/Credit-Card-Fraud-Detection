# Final Model Evaluation Report: Credit Card Fraud Detection

## Report Metadata

| Field | Value |
|---|---|
| Generated | 2026-06-24T11:47:35.859632+00:00 |
| Champion model | `xgboost_baseline` |
| Evaluated split | `test` |
| Locked threshold | `0.53` |
| Threshold source | Day 6 recall-target (validation-only selection) |

---

## Integrity Note — Threshold Selection

The operating threshold was selected exclusively on the validation set during Day 6.  Test-set results were not used in any threshold or model decision.  This evaluation is therefore an honest, unbiased estimate of real-world performance.

This ensures the final test-set evaluation is a truthful, unbiased estimate of production performance.

---

## Dataset Summary

| Field | Value |
|---|---|
| Split evaluated | `test` |
| Total samples | 42,722 |
| Fraud cases | 74 |
| Legitimate cases | 42,648 |

---

## Final Evaluation Metrics

| Metric | Value |
|---|---|
| **PR-AUC** | **0.8288** |
| ROC-AUC | 0.9613 |
| Precision | 0.6966 |
| Recall | 0.8378 |
| F1-score | 0.7607 |
| Specificity | 0.9994 |
| False Positive Rate | 0.0006 |
| False Negative Rate | 0.1622 |

---

## Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actual Legitimate** | TN = 42,621 | FP = 27 |
| **Actual Fraud**      | FN = 12 | TP = 62 |

### Interpretation

- **Fraud caught (TP):** 62 — fraudulent transactions correctly blocked.
- **Fraud missed (FN):** 12 — fraudulent transactions that slipped through.
- **False alerts (FP):** 27 — legitimate transactions incorrectly flagged.
- **True negatives (TN):** 42,621 — legitimate transactions correctly approved.

> **Business context:** In fraud detection, false negatives (missed fraud) typically carry higher cost than false positives (false alerts). The recall-target threshold (0.53) was chosen to catch as much fraud as possible while keeping false alerts at an acceptable level.

---

## Why PR-AUC Is the Primary Metric

The dataset contains roughly **0.17 % fraud** — an extreme class imbalance.
Under these conditions:

- **Accuracy** is misleading.  A model that always predicts 'legitimate' achieves ~99.8 % accuracy while catching zero fraud.
- **ROC-AUC** is influenced heavily by the large number of true negatives and can appear strong even when fraud detection is poor.
- **PR-AUC** (Average Precision) measures the quality of the precision–recall trade-off for the fraud class only.  It is the most meaningful single-number summary for this problem.

---

## Project Result Summary

| Stage | Result |
|---|---|
| Champion model | XGBoost (scale_pos_weight balanced) |
| Validation PR-AUC | 0.8129 |
| Validation ROC-AUC | 0.9851 |
| Operating threshold | 0.53 (recall-target, Day 6 validation) |
| **Final test PR-AUC** | **0.8288** |
| Final test Recall | 0.8378 |
| Final test Precision | 0.6966 |
| Final test F1-score | 0.7607 |

---

*Report generated automatically by `src/evaluation/final_evaluation.py`.*