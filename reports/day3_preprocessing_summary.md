# Day 3 Preprocessing Summary: Credit Card Fraud Detection

## Report Metadata

- Generated at: `2026-06-21T17:36:23.164504+00:00`

## Project Context

This project detects fraudulent credit card transactions in a highly imbalanced dataset. Day 2 confirmed that fraud cases make up only a small fraction of all transactions, which means accuracy alone is not a reliable evaluation metric. Day 3 builds on that understanding by preparing a leakage-safe foundation for modeling: separating features from the `Class` target, creating a stratified train/validation/test split, and fitting a preprocessing pipeline strictly on training data.

## Split Strategy

- Train: 70%
- Validation: 15%
- Test: 15%
- Stratification was used on `Class` to preserve the rare fraud class ratio across all three splits.
- This prevents any split from accidentally over- or under-representing fraud cases.

## Split Summary

- Train rows: 199364 (69.9997%)
- Validation rows: 42721 (15.0%)
- Test rows: 42722 (15.0003%)
- Total rows: 284807
- Feature count: 30

## Class Distribution by Split

| Split | Rows | Legitimate Count | Fraud Count | Legitimate % | Fraud % |
|---|---|---|---|---|---|
| Train | 199364 | 199020 | 344 | 99.8275% | 0.1725% |
| Validation | 42721 | 42647 | 74 | 99.8268% | 0.1732% |
| Test | 42722 | 42648 | 74 | 99.8268% | 0.1732% |

## Preprocessing Strategy

- `Time` and `Amount` are scaled using `StandardScaler`.
- `V1`-`V28` are passed through unchanged because they are already PCA-transformed.
- `Class` is never included in the preprocessing pipeline.
- The preprocessor is fitted only on training data (`X_train`) and reused to transform validation and test data.
- Original feature count: 30
- Processed feature count: 30
- Scaled features: ['Time', 'Amount']
- Passthrough features: ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

## Data Leakage Prevention

- The train/validation/test split is created **before** fitting any preprocessing.
- `StandardScaler` is fitted only on `X_train`.
- Validation and test sets are only **transformed** using the already-fitted preprocessor, never used to fit it.
- The test set remains untouched until final model evaluation.
- Leakage note: The preprocessor (StandardScaler on Time and Amount, passthrough on V1-V28) was fit only on X_train. Validation and test features were transformed using this already-fitted preprocessor and were never used to fit any scaling statistics.

## Generated Interim Files

- `X_train`: `data/interim/X_train.parquet`
- `X_val`: `data/interim/X_val.parquet`
- `X_test`: `data/interim/X_test.parquet`
- `y_train`: `data/interim/y_train.parquet`
- `y_val`: `data/interim/y_val.parquet`
- `y_test`: `data/interim/y_test.parquet`

## Generated Processed Files

- `X_train_processed`: `data/processed/X_train_processed.parquet`
- `X_val_processed`: `data/processed/X_val_processed.parquet`
- `X_test_processed`: `data/processed/X_test_processed.parquet`
- `y_train`: `data/processed/y_train.parquet`
- `y_val`: `data/processed/y_val.parquet`
- `y_test`: `data/processed/y_test.parquet`

## Generated Artifacts

- `preprocessor`: `artifacts/preprocessing/preprocessor.joblib`
- `split_metadata`: `artifacts/preprocessing/split_metadata.json`

## Day 3 Conclusions

- Stratified train/validation/test splits were created successfully.
- Stratification preserved the fraud ratio across all splits.
- The preprocessing pipeline is leakage-safe: it was fitted only on training data.
- Processed datasets are ready to be used for model training.

## Day 4 Next Steps

- Train a baseline Logistic Regression model.
- Train a Random Forest model.
- Evaluate both models using precision, recall, F1-score, PR-AUC, ROC-AUC, and the confusion matrix.
- Avoid relying on accuracy alone, since it is misleading for this highly imbalanced dataset.
