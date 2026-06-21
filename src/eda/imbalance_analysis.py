from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "Class"
LEGITIMATE_LABEL = 0
FRAUD_LABEL = 1


def validate_target_column(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> None:
    """Validate that the target column is present and well-formed.

    Args:
        df: The DataFrame to validate.
        target_col: Name of the binary target column.

    Raises:
        ValueError: If the DataFrame is empty, the target column is
            missing, contains missing values, contains labels other than
            0 and 1, or is missing one of the two classes.
    """
    if df.empty:
        raise ValueError("Dataset is empty. Expected at least one row of data.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the dataset.")

    if df[target_col].isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing values.")

    unique_labels = set(df[target_col].dropna().unique().tolist())
    if not unique_labels.issubset({LEGITIMATE_LABEL, FRAUD_LABEL}):
        raise ValueError(
            f"Target column '{target_col}' must only contain "
            f"{LEGITIMATE_LABEL} and {FRAUD_LABEL}. Found values: {sorted(unique_labels)}."
        )

    if unique_labels != {LEGITIMATE_LABEL, FRAUD_LABEL}:
        raise ValueError(
            f"Target column '{target_col}' must contain both classes "
            f"({LEGITIMATE_LABEL} and {FRAUD_LABEL}). Found values: {sorted(unique_labels)}."
        )


def calculate_class_counts(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> dict[str, int]:
    """Count total, legitimate, and fraudulent transactions.

    Args:
        df: The DataFrame containing the target column.
        target_col: Name of the binary target column.

    Returns:
        A dictionary with total_transactions, legitimate_count, and fraud_count.
    """
    validate_target_column(df, target_col)

    legitimate_count = int((df[target_col] == LEGITIMATE_LABEL).sum())
    fraud_count = int((df[target_col] == FRAUD_LABEL).sum())
    total_transactions = int(df.shape[0])

    return {
        "total_transactions": total_transactions,
        "legitimate_count": legitimate_count,
        "fraud_count": fraud_count,
    }


def calculate_class_percentages(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> dict[str, float]:
    """Calculate the percentage share of each class.

    Args:
        df: The DataFrame containing the target column.
        target_col: Name of the binary target column.

    Returns:
        A dictionary with legitimate_percentage and fraud_percentage,
        each rounded to 4 decimal places.
    """
    counts = calculate_class_counts(df, target_col)
    total = counts["total_transactions"]

    legitimate_percentage = round((counts["legitimate_count"] / total) * 100, 4)
    fraud_percentage = round((counts["fraud_count"] / total) * 100, 4)

    return {
        "legitimate_percentage": legitimate_percentage,
        "fraud_percentage": fraud_percentage,
    }


def calculate_imbalance_ratio(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> float:
    """Calculate the ratio of legitimate to fraudulent transactions.

    Args:
        df: The DataFrame containing the target column.
        target_col: Name of the binary target column.

    Returns:
        The legitimate-to-fraud ratio, rounded to 2 decimal places.

    Raises:
        ValueError: If there are zero fraudulent transactions.
    """
    counts = calculate_class_counts(df, target_col)

    if counts["fraud_count"] == 0:
        raise ValueError("Cannot calculate imbalance ratio: fraud_count is 0.")

    ratio = counts["legitimate_count"] / counts["fraud_count"]
    return round(ratio, 2)


def get_majority_class_accuracy_baseline(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> float:
    """Calculate the accuracy of a naive majority-class classifier.

    A naive classifier that always predicts the majority class would
    achieve this accuracy while detecting zero instances of the minority
    class.

    Args:
        df: The DataFrame containing the target column.
        target_col: Name of the binary target column.

    Returns:
        The majority-class baseline accuracy as a percentage, rounded to
        4 decimal places.
    """
    counts = calculate_class_counts(df, target_col)
    total = counts["total_transactions"]

    majority_count = max(counts["legitimate_count"], counts["fraud_count"])
    baseline_accuracy = round((majority_count / total) * 100, 4)

    return baseline_accuracy


def explain_accuracy_problem(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> str:
    """Explain why plain accuracy is a misleading metric for this dataset.

    Args:
        df: The DataFrame containing the target column.
        target_col: Name of the binary target column.

    Returns:
        A paragraph of text explaining the class imbalance problem using
        the actual fraud percentage and majority-class baseline accuracy
        calculated from the DataFrame.
    """
    percentages = calculate_class_percentages(df, target_col)
    baseline_accuracy = get_majority_class_accuracy_baseline(df, target_col)
    fraud_percentage = percentages["fraud_percentage"]
    legitimate_percentage = percentages["legitimate_percentage"]

    explanation = (
        f"This dataset is highly imbalanced: only {fraud_percentage}% of transactions "
        f"are fraudulent, while {legitimate_percentage}% are legitimate. Because legitimate "
        f"transactions dominate the data, a naive model that always predicts 'legitimate' "
        f"would achieve about {baseline_accuracy}% accuracy without identifying a single "
        f"fraudulent transaction. This makes plain accuracy a dangerous and misleading metric "
        f"for this project, since a model can score very high while completely failing at its "
        f"core purpose of catching fraud. Instead, this project should be evaluated using "
        f"metrics that are sensitive to the minority class, such as precision, recall, "
        f"F1-score, PR-AUC, ROC-AUC, and the confusion matrix. Recall and precision are "
        f"especially important for fraud detection, since recall measures how many actual "
        f"frauds are caught and precision measures how trustworthy a fraud alert is."
    )

    return explanation


def generate_imbalance_summary(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> dict[str, object]:
    """Generate a complete class imbalance summary for the dataset.

    Args:
        df: The DataFrame containing the target column.
        target_col: Name of the binary target column.

    Returns:
        A dictionary containing transaction counts, class percentages,
        imbalance ratio, majority-class accuracy baseline, an accuracy
        warning, and a list of recommended evaluation metrics.
    """
    counts = calculate_class_counts(df, target_col)
    percentages = calculate_class_percentages(df, target_col)
    imbalance_ratio = calculate_imbalance_ratio(df, target_col)
    baseline_accuracy = get_majority_class_accuracy_baseline(df, target_col)
    accuracy_warning = explain_accuracy_problem(df, target_col)

    recommended_metrics = [
        "Precision",
        "Recall",
        "F1-score",
        "PR-AUC",
        "ROC-AUC",
        "Confusion Matrix",
    ]

    summary: dict[str, object] = {
        "total_transactions": counts["total_transactions"],
        "legitimate_count": counts["legitimate_count"],
        "fraud_count": counts["fraud_count"],
        "legitimate_percentage": percentages["legitimate_percentage"],
        "fraud_percentage": percentages["fraud_percentage"],
        "imbalance_ratio": imbalance_ratio,
        "majority_class_accuracy_baseline": baseline_accuracy,
        "accuracy_warning": accuracy_warning,
        "recommended_metrics": recommended_metrics,
    }

    return summary


def format_imbalance_summary_for_console(summary: dict[str, object]) -> str:
    """Format an imbalance summary dictionary as a readable console string.

    Args:
        summary: The dictionary returned by generate_imbalance_summary().

    Returns:
        A multi-line, human-readable string suitable for printing in
        scripts or notebooks.
    """
    recommended_metrics = summary.get("recommended_metrics", [])
    metrics_lines = "\n".join(f"  - {metric}" for metric in recommended_metrics)

    lines = [
        "Class Imbalance Summary",
        "=" * 40,
        f"Total transactions: {summary.get('total_transactions')}",
        f"Legitimate transactions: {summary.get('legitimate_count')} "
        f"({summary.get('legitimate_percentage')}%)",
        f"Fraudulent transactions: {summary.get('fraud_count')} "
        f"({summary.get('fraud_percentage')}%)",
        f"Imbalance ratio (legitimate:fraud): {summary.get('imbalance_ratio')}:1",
        f"Majority-class accuracy baseline: {summary.get('majority_class_accuracy_baseline')}%",
        "",
        "Why accuracy is misleading:",
        f"{summary.get('accuracy_warning')}",
        "",
        "Recommended evaluation metrics:",
        metrics_lines,
    ]

    return "\n".join(lines)
