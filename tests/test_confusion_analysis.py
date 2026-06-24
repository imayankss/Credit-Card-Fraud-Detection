"""Tests for src/evaluation/confusion_analysis.py.

These tests use small, deterministic synthetic arrays and temporary
directories only. They do not require the real Kaggle dataset, the real
trained model artifact, or any test-set data.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.confusion_analysis import (  # noqa: E402
    build_confusion_summaries_for_thresholds,
    calculate_confusion_summary,
    plot_confusion_matrix,
)

REQUIRED_SUMMARY_KEYS = {
    "threshold",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "fraud_caught",
    "fraud_missed",
    "false_alerts",
    "predicted_frauds",
    "predicted_legitimate",
}


@pytest.fixture
def synthetic_labels_and_probabilities() -> Tuple[List[int], List[float]]:
    """Provide a small, deterministic synthetic dataset for confusion analysis.

    At threshold 0.5, predictions are [0, 1, 0, 1]:
        sample 1 (true=0, pred=0) -> TN
        sample 2 (true=0, pred=1) -> FP
        sample 3 (true=1, pred=0) -> FN
        sample 4 (true=1, pred=1) -> TP

    Returns:
        Tuple of (y_true, y_proba) with both classes present.
    """
    y_true = [0, 0, 1, 1]
    y_proba = [0.1, 0.6, 0.4, 0.9]
    return y_true, y_proba


# ---------------------------------------------------------------------------
# calculate_confusion_summary
# ---------------------------------------------------------------------------


def test_calculate_confusion_summary_returns_required_keys(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """The confusion summary should contain all required keys."""
    y_true, y_proba = synthetic_labels_and_probabilities

    summary = calculate_confusion_summary(y_true, y_proba, threshold=0.5)

    assert REQUIRED_SUMMARY_KEYS.issubset(summary.keys())


def test_calculate_confusion_summary_confusion_values_are_correct(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """TP, FP, FN, and TN should match the expected confusion outcome."""
    y_true, y_proba = synthetic_labels_and_probabilities

    summary = calculate_confusion_summary(y_true, y_proba, threshold=0.5)

    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["tn"] == 1


def test_calculate_confusion_summary_specificity_is_correct(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """Specificity should equal tn / (tn + fp)."""
    y_true, y_proba = synthetic_labels_and_probabilities

    summary = calculate_confusion_summary(y_true, y_proba, threshold=0.5)

    # tn=1, fp=1 -> specificity = 1 / (1 + 1) = 0.5
    assert summary["specificity"] == pytest.approx(0.5)


def test_calculate_confusion_summary_false_positive_rate_is_correct(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """False positive rate should equal fp / (fp + tn)."""
    y_true, y_proba = synthetic_labels_and_probabilities

    summary = calculate_confusion_summary(y_true, y_proba, threshold=0.5)

    # fp=1, tn=1 -> fpr = 1 / (1 + 1) = 0.5
    assert summary["false_positive_rate"] == pytest.approx(0.5)


def test_calculate_confusion_summary_false_negative_rate_is_correct(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """False negative rate should equal fn / (fn + tp)."""
    y_true, y_proba = synthetic_labels_and_probabilities

    summary = calculate_confusion_summary(y_true, y_proba, threshold=0.5)

    # fn=1, tp=1 -> fnr = 1 / (1 + 1) = 0.5
    assert summary["false_negative_rate"] == pytest.approx(0.5)


def test_calculate_confusion_summary_predicted_frauds_and_legitimate_are_correct(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """predicted_frauds and predicted_legitimate should match tp+fp and tn+fn."""
    y_true, y_proba = synthetic_labels_and_probabilities

    summary = calculate_confusion_summary(y_true, y_proba, threshold=0.5)

    assert summary["predicted_frauds"] == 2  # tp(1) + fp(1)
    assert summary["predicted_legitimate"] == 2  # tn(1) + fn(1)
    assert summary["fraud_caught"] == summary["tp"]
    assert summary["fraud_missed"] == summary["fn"]
    assert summary["false_alerts"] == summary["fp"]


def test_calculate_confusion_summary_with_no_false_positives_or_negatives() -> None:
    """Perfectly separated data should yield zero FPR/FNR and full specificity."""
    y_true = [0, 0, 1, 1]
    y_proba = [0.05, 0.10, 0.90, 0.95]

    summary = calculate_confusion_summary(y_true, y_proba, threshold=0.5)

    assert summary["fp"] == 0
    assert summary["fn"] == 0
    assert summary["specificity"] == pytest.approx(1.0)
    assert summary["false_positive_rate"] == pytest.approx(0.0)
    assert summary["false_negative_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------


def test_plot_confusion_matrix_creates_non_empty_png_file(
    tmp_path: Path,
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """plot_confusion_matrix should save a non-empty PNG file."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "confusion_matrix.png"

    saved_path = plot_confusion_matrix(y_true, y_proba, threshold=0.5, output_path=output_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_plot_confusion_matrix_creates_parent_directories(
    tmp_path: Path,
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """Missing parent directories should be created automatically."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "nested" / "figures" / "confusion_matrix.png"

    saved_path = plot_confusion_matrix(y_true, y_proba, threshold=0.5, output_path=output_path)

    assert saved_path.exists()


def test_plot_confusion_matrix_accepts_custom_title(
    tmp_path: Path,
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """A custom title should not prevent the PNG file from being created."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "confusion_matrix_custom_title.png"

    saved_path = plot_confusion_matrix(
        y_true, y_proba, threshold=0.5, output_path=output_path, title="Custom Title"
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# build_confusion_summaries_for_thresholds
# ---------------------------------------------------------------------------


def test_build_confusion_summaries_for_thresholds_returns_summary_per_name(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """A summary should be returned for each named threshold."""
    y_true, y_proba = synthetic_labels_and_probabilities
    selected_thresholds = {"default": 0.5, "high": 0.8}

    summaries = build_confusion_summaries_for_thresholds(y_true, y_proba, selected_thresholds)

    assert set(summaries.keys()) == {"default", "high"}
    assert summaries["default"]["threshold"] == pytest.approx(0.5)
    assert summaries["high"]["threshold"] == pytest.approx(0.8)
    for summary in summaries.values():
        assert REQUIRED_SUMMARY_KEYS.issubset(summary.keys())


def test_build_confusion_summaries_for_thresholds_accepts_metrics_dict_values(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """Threshold values provided as metrics dictionaries should be supported."""
    y_true, y_proba = synthetic_labels_and_probabilities
    selected_thresholds = {
        "best_f1": {"threshold": 0.6, "f1": 0.8},
        "recall_target": {"threshold": 0.3, "recall": 0.9},
    }

    summaries = build_confusion_summaries_for_thresholds(y_true, y_proba, selected_thresholds)

    assert summaries["best_f1"]["threshold"] == pytest.approx(0.6)
    assert summaries["recall_target"]["threshold"] == pytest.approx(0.3)


def test_build_confusion_summaries_for_thresholds_raises_on_empty_input(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """An empty selected_thresholds mapping should raise a clear ValueError."""
    y_true, y_proba = synthetic_labels_and_probabilities

    with pytest.raises(ValueError):
        build_confusion_summaries_for_thresholds(y_true, y_proba, {})


def test_build_confusion_summaries_for_thresholds_raises_on_missing_threshold_key(
    synthetic_labels_and_probabilities: Tuple[List[int], List[float]],
) -> None:
    """A dict value without a 'threshold' key should raise a clear ValueError."""
    y_true, y_proba = synthetic_labels_and_probabilities
    selected_thresholds = {"broken": {"f1": 0.8}}

    with pytest.raises(ValueError):
        build_confusion_summaries_for_thresholds(y_true, y_proba, selected_thresholds)
