"""
src/evaluation/curves.py
------------------------
Precision-recall and ROC curve plotting utilities for the
Credit Card Fraud Detection & Risk Scoring System.

Scope: Day 6 — validation-only curve analysis.
Do NOT use test data here.  Do NOT train models here.

Public API
----------
plot_precision_recall_curve(y_true, y_proba, output_path)
plot_roc_curve(y_true, y_proba, output_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared input validator
# ---------------------------------------------------------------------------

def _validate_binary_inputs(
    y_true: "np.ndarray | list",
    y_proba: "np.ndarray | list",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary classification inputs, returning clean numpy arrays.

    Tries to delegate to ``src.evaluation.threshold_tuning.validate_binary_inputs``
    when that module is available; otherwise performs the same checks inline so
    this file remains self-contained.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = legitimate, 1 = fraud).
    y_proba:
        Predicted fraud probabilities in [0, 1].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (y_true_arr, y_proba_arr) as 1-D float64 arrays.

    Raises
    ------
    ValueError
        On empty input, length mismatch, invalid labels, or out-of-range
        probabilities.
    """
    # --- prefer the shared validator from threshold_tuning --------------------
    try:
        from src.evaluation.threshold_tuning import validate_binary_inputs  # type: ignore
        validate_binary_inputs(y_true, y_proba)
        return np.asarray(y_true, dtype=np.float64), np.asarray(y_proba, dtype=np.float64)
    except ImportError:
        pass  # fall through to inline validation

    # --- inline validation (mirrors threshold_tuning contract) ----------------
    y_true_arr = np.asarray(y_true, dtype=np.float64).ravel()
    y_proba_arr = np.asarray(y_proba, dtype=np.float64).ravel()

    if y_true_arr.size == 0:
        raise ValueError("y_true must not be empty.")
    if y_proba_arr.size == 0:
        raise ValueError("y_proba must not be empty.")
    if y_true_arr.size != y_proba_arr.size:
        raise ValueError(
            f"Length mismatch: y_true has {y_true_arr.size} elements "
            f"but y_proba has {y_proba_arr.size}."
        )

    unique_labels = set(np.unique(y_true_arr))
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError(
            f"y_true must contain only 0 and 1; found labels: {unique_labels}."
        )
    if len(unique_labels) < 2:
        raise ValueError(
            "y_true must contain both classes (0 and 1) to plot curves."
        )

    if np.any(y_proba_arr < 0.0) or np.any(y_proba_arr > 1.0):
        raise ValueError("y_proba values must be in [0, 1].")

    return y_true_arr, y_proba_arr


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir(output_path: Path) -> None:
    """Create parent directories for *output_path* if they do not exist."""
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _close_figure(fig: "plt.Figure") -> None:
    """Close a matplotlib figure to free memory."""
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(
    y_true: "np.ndarray | list",
    y_proba: "np.ndarray | list",
    output_path: Union[str, Path],
) -> Path:
    """Plot and save the precision-recall curve for fraud detection.

    The curve is computed on the **validation set only**.  The area under
    the curve (PR-AUC / average precision) is shown in the legend.

    A random baseline at ``precision = fraud_prevalence`` is drawn for
    reference, illustrating how much better the model is than chance.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = legitimate, 1 = fraud).
    y_proba:
        Predicted fraud probabilities in [0, 1].
    output_path:
        Destination file path (PNG).  Parent directories are created
        automatically.

    Returns
    -------
    Path
        Resolved path of the saved PNG file.

    Raises
    ------
    ValueError
        If inputs are invalid (see ``_validate_binary_inputs``).
    """
    y_true_arr, y_proba_arr = _validate_binary_inputs(y_true, y_proba)
    output_path = Path(output_path).resolve()
    _ensure_output_dir(output_path)

    pr_auc = average_precision_score(y_true_arr, y_proba_arr)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true_arr, y_proba_arr)

    fraud_prevalence = float(y_true_arr.mean())

    fig, ax = plt.subplots(figsize=(8, 6))

    # Model curve
    ax.plot(
        recall_vals,
        precision_vals,
        color="#2563eb",  # blue
        linewidth=2,
        label=f"Model (PR-AUC = {pr_auc:.4f})",
    )

    # Random baseline (horizontal line at fraud prevalence)
    ax.axhline(
        y=fraud_prevalence,
        color="#dc2626",  # red
        linewidth=1.5,
        linestyle="--",
        label=f"Random baseline (prevalence = {fraud_prevalence:.4f})",
    )

    ax.set_xlabel("Recall (Fraud Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (Fraud PPV)", fontsize=12)
    ax.set_title("Precision-Recall Curve — Fraud Detection (Validation Set)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _close_figure(fig)

    logger.info("Precision-recall curve saved to: %s  (PR-AUC=%.4f)", output_path, pr_auc)
    return output_path


def plot_roc_curve(
    y_true: "np.ndarray | list",
    y_proba: "np.ndarray | list",
    output_path: Union[str, Path],
) -> Path:
    """Plot and save the ROC curve for fraud detection.

    The curve is computed on the **validation set only**.  The area under
    the curve (ROC-AUC) is shown in the legend.

    The diagonal random-classifier baseline (AUC = 0.5) is drawn for
    reference.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = legitimate, 1 = fraud).
    y_proba:
        Predicted fraud probabilities in [0, 1].
    output_path:
        Destination file path (PNG).  Parent directories are created
        automatically.

    Returns
    -------
    Path
        Resolved path of the saved PNG file.

    Raises
    ------
    ValueError
        If inputs are invalid (see ``_validate_binary_inputs``).
    """
    y_true_arr, y_proba_arr = _validate_binary_inputs(y_true, y_proba)
    output_path = Path(output_path).resolve()
    _ensure_output_dir(output_path)

    roc_auc = roc_auc_score(y_true_arr, y_proba_arr)
    fpr_vals, tpr_vals, _ = roc_curve(y_true_arr, y_proba_arr)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Model curve
    ax.plot(
        fpr_vals,
        tpr_vals,
        color="#16a34a",  # green
        linewidth=2,
        label=f"Model (ROC-AUC = {roc_auc:.4f})",
    )

    # Random baseline diagonal
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#9ca3af",  # grey
        linewidth=1.5,
        linestyle="--",
        label="Random classifier (AUC = 0.5000)",
    )

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve — Fraud Detection (Validation Set)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _close_figure(fig)

    logger.info("ROC curve saved to: %s  (ROC-AUC=%.4f)", output_path, roc_auc)
    return output_path


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        "curves.py is a utility module.\n"
        "It is intended to be called from scripts/run_day6_threshold_tuning.py.\n"
        "Example usage:\n\n"
        "  from src.evaluation.curves import plot_precision_recall_curve, plot_roc_curve\n"
        "  plot_precision_recall_curve(y_val, y_val_proba, 'reports/figures/pr_curve.png')\n"
        "  plot_roc_curve(y_val, y_val_proba, 'reports/figures/roc_curve.png')"
    )
