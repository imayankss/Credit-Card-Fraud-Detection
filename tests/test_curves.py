"""Tests for src/evaluation/curves.py.

These tests use small, deterministic synthetic arrays and temporary
directories only. They do not require the real Kaggle dataset, the real
trained model artifact, or any test-set data.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.curves import plot_precision_recall_curve, plot_roc_curve  # noqa: E402


@pytest.fixture
def synthetic_labels_and_probabilities() -> tuple[list[int], list[float]]:
    """Provide a small, deterministic synthetic dataset for curve plotting.

    Returns:
        Tuple of (y_true, y_proba) with both classes present.
    """
    y_true = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    y_proba = [0.05, 0.20, 0.40, 0.95, 0.85, 0.60, 0.30, 0.70, 0.10, 0.55]
    return y_true, y_proba


# ---------------------------------------------------------------------------
# plot_precision_recall_curve
# ---------------------------------------------------------------------------


def test_plot_precision_recall_curve_creates_png_file(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """Calling plot_precision_recall_curve should create a PNG file."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "precision_recall_curve.png"

    plot_precision_recall_curve(y_true, y_proba, output_path)

    assert output_path.exists()


def test_plot_precision_recall_curve_png_is_non_empty(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """The saved precision-recall curve PNG file should not be empty."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "precision_recall_curve.png"

    plot_precision_recall_curve(y_true, y_proba, output_path)

    assert output_path.stat().st_size > 0


def test_plot_precision_recall_curve_rejects_invalid_probabilities(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """Probabilities outside [0, 1] should raise a clear ValueError."""
    y_true, _ = synthetic_labels_and_probabilities
    invalid_proba = [0.5] * (len(y_true) - 1) + [1.5]
    output_path = tmp_path / "precision_recall_curve.png"

    with pytest.raises(ValueError):
        plot_precision_recall_curve(y_true, invalid_proba, output_path)


def test_plot_precision_recall_curve_rejects_invalid_labels(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """Labels outside {0, 1} should raise a clear ValueError."""
    _, y_proba = synthetic_labels_and_probabilities
    invalid_labels = [0] * (len(y_proba) - 1) + [2]
    output_path = tmp_path / "precision_recall_curve.png"

    with pytest.raises(ValueError):
        plot_precision_recall_curve(invalid_labels, y_proba, output_path)


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------


def test_plot_roc_curve_creates_png_file(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """Calling plot_roc_curve should create a PNG file."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "roc_curve.png"

    plot_roc_curve(y_true, y_proba, output_path)

    assert output_path.exists()


def test_plot_roc_curve_png_is_non_empty(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """The saved ROC curve PNG file should not be empty."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "roc_curve.png"

    plot_roc_curve(y_true, y_proba, output_path)

    assert output_path.stat().st_size > 0


def test_plot_roc_curve_rejects_invalid_probabilities(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """Probabilities outside [0, 1] should raise a clear ValueError."""
    y_true, _ = synthetic_labels_and_probabilities
    invalid_proba = [-0.2] + [0.5] * (len(y_true) - 1)
    output_path = tmp_path / "roc_curve.png"

    with pytest.raises(ValueError):
        plot_roc_curve(y_true, invalid_proba, output_path)


def test_plot_roc_curve_rejects_invalid_labels(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """Labels outside {0, 1} should raise a clear ValueError."""
    _, y_proba = synthetic_labels_and_probabilities
    invalid_labels = [3] + [0] * (len(y_proba) - 1)
    output_path = tmp_path / "roc_curve.png"

    with pytest.raises(ValueError):
        plot_roc_curve(invalid_labels, y_proba, output_path)


# ---------------------------------------------------------------------------
# Output directory creation
# ---------------------------------------------------------------------------


def test_plot_precision_recall_curve_creates_parent_directories(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """Missing parent directories should be created automatically."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "nested" / "figures" / "precision_recall_curve.png"

    plot_precision_recall_curve(y_true, y_proba, output_path)

    assert output_path.exists()


def test_plot_roc_curve_creates_parent_directories(
    tmp_path: Path,
    synthetic_labels_and_probabilities: tuple[list[int], list[float]],
) -> None:
    """Missing parent directories should be created automatically."""
    y_true, y_proba = synthetic_labels_and_probabilities
    output_path = tmp_path / "nested" / "figures" / "roc_curve.png"

    plot_roc_curve(y_true, y_proba, output_path)

    assert output_path.exists()
