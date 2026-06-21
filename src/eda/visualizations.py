from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_FIGURES_DIR = Path("reports/figures")
TARGET_COLUMN = "Class"
LEGITIMATE_LABEL = 0
FRAUD_LABEL = 1


def ensure_output_dir(output_dir: str | Path = DEFAULT_FIGURES_DIR) -> Path:
    """Ensure the figures output directory exists.

    Args:
        output_dir: Directory where figures should be saved.

    Returns:
        The resolved Path to the output directory.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that a DataFrame is non-empty and contains required columns.

    Args:
        df: The DataFrame to validate.
        required_columns: Column names that must be present.

    Raises:
        ValueError: If the DataFrame is empty or any required column is missing.
    """
    if df.empty:
        raise ValueError("Dataset is empty. Expected at least one row of data.")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}.")


def save_current_figure(output_path: str | Path) -> Path:
    """Save the current matplotlib figure to disk and close it.

    Args:
        output_path: Destination file path for the saved figure.

    Returns:
        The Path where the figure was saved.
    """
    path = Path(output_path)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path


def plot_class_distribution(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    target_col: str = TARGET_COLUMN,
) -> Path:
    """Plot and save a bar chart of legitimate vs fraud transaction counts.

    Args:
        df: The DataFrame containing the target column.
        output_dir: Directory where the figure should be saved.
        target_col: Name of the binary target column.

    Returns:
        The Path to the saved figure.
    """
    validate_required_columns(df, [target_col])
    output_path = ensure_output_dir(output_dir)

    class_counts = df[target_col].value_counts().sort_index()
    labels = ["Legitimate" if label == LEGITIMATE_LABEL else "Fraud" for label in class_counts.index]

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=labels, y=class_counts.values, hue=labels, palette="Set2", legend=False)
    ax.set_title("Class Distribution: Legitimate vs Fraud Transactions")
    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("Number of Transactions")

    for bar, count in zip(ax.patches, class_counts.values):
        ax.annotate(
            f"{int(count):,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    return save_current_figure(output_path / "class_distribution.png")


def plot_amount_distribution_by_class(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    amount_col: str = "Amount",
    target_col: str = TARGET_COLUMN,
) -> Path:
    """Plot and save the transaction amount distribution grouped by class.

    Amount is log1p-transformed for visualization purposes only; the input
    DataFrame is never modified.

    Args:
        df: The DataFrame containing amount and target columns.
        output_dir: Directory where the figure should be saved.
        amount_col: Name of the transaction amount column.
        target_col: Name of the binary target column.

    Returns:
        The Path to the saved figure.
    """
    validate_required_columns(df, [amount_col, target_col])
    output_path = ensure_output_dir(output_dir)

    plot_df = df[[amount_col, target_col]].copy()
    plot_df["log_amount"] = np.log1p(plot_df[amount_col].clip(lower=0))
    plot_df["class_label"] = plot_df[target_col].map(
        {LEGITIMATE_LABEL: "Legitimate", FRAUD_LABEL: "Fraud"}
    )

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=plot_df,
        x="log_amount",
        hue="class_label",
        stat="count",
        bins=50,
        common_norm=False,
        element="step",
    )
    ax.set_title("Transaction Amount Distribution by Class")
    ax.set_xlabel("log1p(Amount)")
    ax.set_ylabel("Transaction Count")

    return save_current_figure(output_path / "amount_distribution_by_class.png")


def plot_time_distribution_by_class(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    time_col: str = "Time",
    target_col: str = TARGET_COLUMN,
) -> Path:
    """Plot and save the transaction time distribution grouped by class.

    Args:
        df: The DataFrame containing time and target columns.
        output_dir: Directory where the figure should be saved.
        time_col: Name of the transaction time column.
        target_col: Name of the binary target column.

    Returns:
        The Path to the saved figure.
    """
    validate_required_columns(df, [time_col, target_col])
    output_path = ensure_output_dir(output_dir)

    plot_df = df[[time_col, target_col]].copy()
    plot_df["class_label"] = plot_df[target_col].map(
        {LEGITIMATE_LABEL: "Legitimate", FRAUD_LABEL: "Fraud"}
    )

    plt.figure(figsize=(8, 5))
    ax = sns.histplot(
        data=plot_df,
        x=time_col,
        hue="class_label",
        stat="density",
        bins=50,
        common_norm=False,
        element="step",
    )
    ax.set_title("Transaction Time Distribution by Class")
    ax.set_xlabel("Time (seconds since first transaction)")
    ax.set_ylabel("Density")

    return save_current_figure(output_path / "time_distribution_by_class.png")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    target_col: str = TARGET_COLUMN,
) -> Path:
    """Plot and save a correlation heatmap of all numeric columns.

    Args:
        df: The DataFrame containing the target column and numeric features.
        output_dir: Directory where the figure should be saved.
        target_col: Name of the binary target column.

    Returns:
        The Path to the saved figure.
    """
    validate_required_columns(df, [target_col])
    output_path = ensure_output_dir(output_dir)

    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(20, 16))
    ax = sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature Correlation Heatmap")

    return save_current_figure(output_path / "correlation_heatmap.png")


def generate_all_eda_figures(
    df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
) -> dict[str, Path]:
    """Generate and save all Day 2 EDA figures.

    Args:
        df: The DataFrame containing the full credit card transaction dataset.
        output_dir: Directory where figures should be saved.

    Returns:
        A dictionary mapping figure names to their saved Path locations.
    """
    figures: dict[str, Path] = {
        "class_distribution": plot_class_distribution(df, output_dir),
        "amount_distribution_by_class": plot_amount_distribution_by_class(df, output_dir),
        "time_distribution_by_class": plot_time_distribution_by_class(df, output_dir),
        "correlation_heatmap": plot_correlation_heatmap(df, output_dir),
    }

    return figures
