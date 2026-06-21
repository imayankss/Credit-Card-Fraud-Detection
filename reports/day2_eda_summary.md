# Day 2 EDA Summary: Credit Card Fraud Detection

_Report generated: 2026-06-21 13:03:16_

## Project Context

This report covers Day 2 of the Credit Card Fraud Detection & Risk Scoring System project. Day 2 focuses exclusively on exploratory data analysis, dataset validation, and class imbalance explanation. No preprocessing, scaling, model training, or threshold tuning has been performed at this stage.

## Dataset Overview

- Rows: 284807
- Columns: 31
- Numeric columns: 31
- Non-numeric columns: 0
- Missing values (total): 0
- Duplicate rows: 1081
- Memory usage: 67.3602 MB

## Column Overview

`Time`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`, `V7`, `V8`, `V9`, `V10`, `V11`, `V12`, `V13`, `V14`, `V15`, `V16`, `V17`, `V18`, `V19`, `V20`, `V21`, `V22`, `V23`, `V24`, `V25`, `V26`, `V27`, `V28`, `Amount`, `Class`

## Missing Values Summary

No missing values were found in the dataset.

## Duplicate Rows

The dataset contains 1081 duplicate row(s).

## Basic Numeric Summary

| index | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time | 284807.0000 | 94813.8596 | 47488.1460 | 0.0000 | 54201.5000 | 84692.0000 | 139320.5000 | 172792.0000 |
| V1 | 284807.0000 | 0.0000 | 1.9587 | -56.4075 | -0.9204 | 0.0181 | 1.3156 | 2.4549 |
| V2 | 284807.0000 | 0.0000 | 1.6513 | -72.7157 | -0.5985 | 0.0655 | 0.8037 | 22.0577 |
| V3 | 284807.0000 | -0.0000 | 1.5163 | -48.3256 | -0.8904 | 0.1798 | 1.0272 | 9.3826 |
| V4 | 284807.0000 | 0.0000 | 1.4159 | -5.6832 | -0.8486 | -0.0198 | 0.7433 | 16.8753 |
| V5 | 284807.0000 | 0.0000 | 1.3802 | -113.7433 | -0.6916 | -0.0543 | 0.6119 | 34.8017 |
| V6 | 284807.0000 | 0.0000 | 1.3323 | -26.1605 | -0.7683 | -0.2742 | 0.3986 | 73.3016 |
| V7 | 284807.0000 | -0.0000 | 1.2371 | -43.5572 | -0.5541 | 0.0401 | 0.5704 | 120.5895 |
| V8 | 284807.0000 | 0.0000 | 1.1944 | -73.2167 | -0.2086 | 0.0224 | 0.3273 | 20.0072 |
| V9 | 284807.0000 | -0.0000 | 1.0986 | -13.4341 | -0.6431 | -0.0514 | 0.5971 | 15.5950 |
| V10 | 284807.0000 | 0.0000 | 1.0888 | -24.5883 | -0.5354 | -0.0929 | 0.4539 | 23.7451 |
| V11 | 284807.0000 | 0.0000 | 1.0207 | -4.7975 | -0.7625 | -0.0328 | 0.7396 | 12.0189 |
| V12 | 284807.0000 | -0.0000 | 0.9992 | -18.6837 | -0.4056 | 0.1400 | 0.6182 | 7.8484 |
| V13 | 284807.0000 | 0.0000 | 0.9953 | -5.7919 | -0.6485 | -0.0136 | 0.6625 | 7.1269 |
| V14 | 284807.0000 | 0.0000 | 0.9586 | -19.2143 | -0.4256 | 0.0506 | 0.4931 | 10.5268 |
| V15 | 284807.0000 | 0.0000 | 0.9153 | -4.4989 | -0.5829 | 0.0481 | 0.6488 | 8.8777 |
| V16 | 284807.0000 | 0.0000 | 0.8763 | -14.1299 | -0.4680 | 0.0664 | 0.5233 | 17.3151 |
| V17 | 284807.0000 | -0.0000 | 0.8493 | -25.1628 | -0.4837 | -0.0657 | 0.3997 | 9.2535 |
| V18 | 284807.0000 | 0.0000 | 0.8382 | -9.4987 | -0.4988 | -0.0036 | 0.5008 | 5.0411 |
| V19 | 284807.0000 | 0.0000 | 0.8140 | -7.2135 | -0.4563 | 0.0037 | 0.4589 | 5.5920 |
| V20 | 284807.0000 | 0.0000 | 0.7709 | -54.4977 | -0.2117 | -0.0625 | 0.1330 | 39.4209 |
| V21 | 284807.0000 | 0.0000 | 0.7345 | -34.8304 | -0.2284 | -0.0295 | 0.1864 | 27.2028 |
| V22 | 284807.0000 | -0.0000 | 0.7257 | -10.9331 | -0.5424 | 0.0068 | 0.5286 | 10.5031 |
| V23 | 284807.0000 | 0.0000 | 0.6245 | -44.8077 | -0.1618 | -0.0112 | 0.1476 | 22.5284 |
| V24 | 284807.0000 | 0.0000 | 0.6056 | -2.8366 | -0.3546 | 0.0410 | 0.4395 | 4.5845 |
| V25 | 284807.0000 | 0.0000 | 0.5213 | -10.2954 | -0.3171 | 0.0166 | 0.3507 | 7.5196 |
| V26 | 284807.0000 | 0.0000 | 0.4822 | -2.6046 | -0.3270 | -0.0521 | 0.2410 | 3.5173 |
| V27 | 284807.0000 | -0.0000 | 0.4036 | -22.5657 | -0.0708 | 0.0013 | 0.0910 | 31.6122 |
| V28 | 284807.0000 | -0.0000 | 0.3301 | -15.4301 | -0.0530 | 0.0112 | 0.0783 | 33.8478 |
| Amount | 284807.0000 | 88.3496 | 250.1201 | 0.0000 | 5.6000 | 22.0000 | 77.1650 | 25691.1600 |
| Class | 284807.0000 | 0.0017 | 0.0415 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## Class Imbalance Analysis

| Metric | Value |
| --- | --- |
| Total transactions | 284807 |
| Legitimate transactions | 284315 (99.8273%) |
| Fraudulent transactions | 492 (0.1727%) |
| Imbalance ratio (legitimate:fraud) | 577.88:1 |
| Majority-class accuracy baseline | 99.8273% |

### Why Accuracy Is Misleading

This dataset is highly imbalanced: only 0.1727% of transactions are fraudulent, while 99.8273% are legitimate. Because legitimate transactions dominate the data, a naive model that always predicts 'legitimate' would achieve about 99.8273% accuracy without identifying a single fraudulent transaction. This makes plain accuracy a dangerous and misleading metric for this project, since a model can score very high while completely failing at its core purpose of catching fraud. Instead, this project should be evaluated using metrics that are sensitive to the minority class, such as precision, recall, F1-score, PR-AUC, ROC-AUC, and the confusion matrix. Recall and precision are especially important for fraud detection, since recall measures how many actual frauds are caught and precision measures how trustworthy a fraud alert is.

Because legitimate transactions dominate this dataset, a model can achieve a very high accuracy score while still failing to detect any fraud at all. Accuracy treats every correct prediction equally, but in fraud detection the rare positive class (fraud) is exactly what matters most to the business.

### Recommended Evaluation Metrics

- Precision
- Recall
- F1-score
- PR-AUC
- ROC-AUC
- Confusion Matrix

Precision and recall are especially important for fraud detection: recall measures how many actual fraudulent transactions are correctly identified, while precision measures how trustworthy a fraud alert is. F1-score balances the two, and PR-AUC and ROC-AUC summarize performance across all classification thresholds. The confusion matrix makes the tradeoff between false positives and false negatives explicit.

## Generated Figures

- **Class Distribution:** `reports/figures/class_distribution.png`
- **Amount Distribution by Class:** `reports/figures/amount_distribution_by_class.png`
- **Time Distribution by Class:** `reports/figures/time_distribution_by_class.png`
- **Correlation Heatmap:** `reports/figures/correlation_heatmap.png`

## Why Accuracy Is Misleading

This dataset is highly imbalanced: only 0.1727% of transactions are fraudulent, while 99.8273% are legitimate. Because legitimate transactions dominate the data, a naive model that always predicts 'legitimate' would achieve about 99.8273% accuracy without identifying a single fraudulent transaction. This makes plain accuracy a dangerous and misleading metric for this project, since a model can score very high while completely failing at its core purpose of catching fraud. Instead, this project should be evaluated using metrics that are sensitive to the minority class, such as precision, recall, F1-score, PR-AUC, ROC-AUC, and the confusion matrix. Recall and precision are especially important for fraud detection, since recall measures how many actual frauds are caught and precision measures how trustworthy a fraud alert is.

## Day 2 Conclusions

- The dataset was loaded and validated successfully.
- The target distribution (`Class`) is highly imbalanced.
- Fraudulent transactions are extremely rare compared to legitimate transactions.
- Accuracy alone is not a reliable metric for evaluating fraud detection performance.
- Future evaluation must rely on precision, recall, F1-score, PR-AUC, ROC-AUC, and the confusion matrix instead of accuracy.

## Day 3 Next Steps

- Build a leakage-safe preprocessing pipeline.
- Create a stratified train/validation/test split.
- Apply scaling where appropriate, such as for the `Amount` and `Time` columns.
- Prepare the dataset for baseline model training.
