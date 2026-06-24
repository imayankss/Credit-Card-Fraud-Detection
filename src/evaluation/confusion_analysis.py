"""Validation-only confusion matrix utilities for selected thresholds.

This module converts validation fraud probabilities into confusion-matrix
based summaries and confusion matrix plots at specific, business-relevant
thresholds (e.g. default, best-F1, recall-target).

This module does not train models and does not use test data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

try:
    from src.evaluation.threshold_tuning import validate_binary_inputs
except ImportError:  # pragma: no cover - fallback if import path differs
    def validate_binary_inputs(y_true: Any, y_proba: Any) -> None:
        """Fallback validation helper used if the project import is unavailable.

        Args:
            y_true: Ground-truth binary labels (must contain only 0 and 1).
            y_proba: Predicted probabilities for the positive class.

        Raises:
            ValueError: If either input is empty, lengths mismatch, y_true
                contains values outside {0, 1}, or y_proba contains values
                outside the [0, 1] range.
        """
        y_true_arr = np.asarray(y_true)
        y_proba_arr = np.asarray(y_proba)

        if y_true_arr.size == 0 or y_proba_arr.size == 0:
            raise ValueError(
                "y_true and y_proba must not be empty. "
                f"Got y_true size={y_true_arr.size}, y_proba size={y_proba_arr.size}."
            )

        if y_true_arr.shape[0] != y_proba_arr.shape[0]:
            raise ValueError(
                "y_true and y_proba must have the same length. "
                f"Got y_true length={y_true_arr.shape[0]}, "
                f"y_proba length={y_proba_arr.shape[0]}."
            )

        unique_labels = set(np.unique(y_true_arr).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                "y_true must contain only 0 and 1 (legitimate=0, fraud=1). "
                f"Found unexpected values: {sorted(unique_labels - {0, 1})}."
            )

        if np.isnan(y_proba_arr.astype(float)).any():
            raise ValueError("y_proba must not contain NaN values.")

        if (y_proba_arr < 0).any() or (y_proba_arr > 1).any():
            raise ValueError(
                "y_proba must contain values between 0 and 1 inclusive. "
                f"Found min={float(np.min(y_proba_arr))}, max={float(np.max(y_proba_arr))}."
            )


ArrayLike = Union[pd.Series, np.ndarray, Sequence[float]]


def _validate_threshold(threshold: float) -> None:
    """Validate that a single threshold value is within [0, 1].

    Args:
        threshold: Classification threshold to validate.

    Raises:
        ValueError: If threshold is not between 0 and 1 inclusive.
    """
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"threshold must be between 0 and 1. Got {threshold}.")


def calculate_confusion_summary(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    threshold: float,
) -> Dict[str, Any]:
    """Calculate a confusion-matrix based summary at a single threshold.

    Args:
        y_true: Ground-truth binary labels (0=legitimate, 1=fraud).
        y_proba: Predicted probabilities for the positive (fraud) class.
        threshold: Probability cutoff used to convert probabilities into
            binary predictions (prediction = 1 if y_proba >= threshold).

    Returns:
        Dictionary containing threshold, tp, fp, fn, tn, precision,
        recall, f1, specificity, false_positive_rate,
        false_negative_rate, fraud_caught, fraud_missed, false_alerts,
        predicted_frauds, and predicted_legitimate.
    """
    validate_binary_inputs(y_true, y_proba)
    _validate_threshold(threshold)

    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)

    y_pred = (y_proba_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true_arr, y_pred, zero_division=0)
    recall = recall_score(y_true_arr, y_pred, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred, zero_division=0)

    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = float(fn) / float(fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "fraud_caught": int(tp),
        "fraud_missed": int(fn),
        "false_alerts": int(fp),
        "predicted_frauds": int(tp + fp),
        "predicted_legitimate": int(tn + fn),
    }


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    threshold: float,
    output_path: Union[str, Path],
    title: str = None,
) -> Path:
    """Plot and save a confusion matrix for a single threshold.

    Args:
        y_true: Ground-truth binary labels (0=legitimate, 1=fraud).
        y_proba: Predicted probabilities for the positive (fraud) class.
        threshold: Probability cutoff used to convert probabilities into
            binary predictions (prediction = 1 if y_proba >= threshold).
        output_path: File path where the PNG figure will be saved. Parent
            directories are created if they do not already exist.
        title: Optional plot title. Defaults to a title that includes the
            threshold value.

    Returns:
        Path to the saved confusion matrix PNG file.
    """
    validate_binary_inputs(y_true, y_proba)
    _validate_threshold(threshold)

    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)

    y_pred = (y_proba_arr >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])

    plot_title = title if title is not None else f"Confusion Matrix (threshold={threshold:.2f})"
    class_labels = ["Legitimate (0)", "Fraud (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(plot_title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    max_value = cm.max() if cm.size > 0 else 0
    text_threshold = max_value / 2.0 if max_value > 0 else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            cell_value = cm[row, col]
            text_color = "white" if cell_value > text_threshold else "black"
            ax.text(
                col,
                row,
                format(cell_value, "d"),
                ha="center",
                va="center",
                color=text_color,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)

    return output_file


def build_confusion_summaries_for_thresholds(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    selected_thresholds: Dict[str, Union[float, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Build confusion-matrix summaries for a set of named thresholds.

    Args:
        y_true: Ground-truth binary labels (0=legitimate, 1=fraud).
        y_proba: Predicted probabilities for the positive (fraud) class.
        selected_thresholds: Mapping of threshold name (e.g. "default",
            "best_f1", "recall_target") to either a raw threshold value
            (float) or a metrics dictionary containing a "threshold" key.

    Returns:
        Dictionary mapping each threshold name to its confusion summary
        dictionary, as produced by calculate_confusion_summary.

    Raises:
        ValueError: If selected_thresholds is empty.
    """
    validate_binary_inputs(y_true, y_proba)

    if not selected_thresholds:
        raise ValueError("selected_thresholds must not be empty.")

    summaries: Dict[str, Dict[str, Any]] = {}
    for name, threshold_value in selected_thresholds.items():
        if isinstance(threshold_value, dict):
            if "threshold" not in threshold_value:
                raise ValueError(
                    f"selected_thresholds['{name}'] is a dict but has no 'threshold' key."
                )
            threshold = threshold_value["threshold"]
        else:
            threshold = threshold_value

        summaries[name] = calculate_confusion_summary(y_true, y_proba, threshold)

    return summaries


if __name__ == "__main__":
    print(
        "This module is intended to be used by "
        "scripts/run_day6_threshold_tuning.py"
    )
