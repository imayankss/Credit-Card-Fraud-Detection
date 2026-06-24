"""
Tests for src.models.advanced_models (Day 5).

These tests use only small, synthetic, in-memory datasets. They never load
the real Kaggle dataset and never touch Day 3's train/validation/test split
files on disk. They are intentionally lightweight so they run quickly as
part of a normal pytest invocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is importable when tests are run from the
# repo root (e.g. `pytest`) or from within the tests/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.advanced_models import (  # noqa: E402
    build_xgboost_classifier,
    calculate_scale_pos_weight,
    save_model,
    train_xgboost_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_imbalanced_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a tiny, deterministic, imbalanced synthetic training dataset.

    Returns:
        Tuple of (X_train, y_train) with 100 rows, 5 numeric features, and
        roughly 10% positive (fraud) examples. Both classes are present.
    """
    rng = np.random.default_rng(seed=42)

    n_rows = 100
    n_fraud = 10
    n_legit = n_rows - n_fraud

    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 5))
    fraud_features = rng.normal(loc=3.0, scale=1.0, size=(n_fraud, 5))

    X = pd.DataFrame(
        np.vstack([legit_features, fraud_features]),
        columns=[f"feature_{i}" for i in range(5)],
    )
    y = pd.Series([0] * n_legit + [1] * n_fraud, name="Class")

    return X, y


@pytest.fixture
def balanced_labels() -> pd.Series:
    """A label series with an equal number of 0s and 1s for a known ratio."""
    return pd.Series([0, 0, 0, 0, 1, 1])  # negative_count=4, positive_count=2


@pytest.fixture
def no_fraud_labels() -> pd.Series:
    """A label series containing only the negative class."""
    return pd.Series([0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# calculate_scale_pos_weight
# ---------------------------------------------------------------------------


def test_calculate_scale_pos_weight_returns_negative_over_positive_ratio(
    balanced_labels: pd.Series,
) -> None:
    """scale_pos_weight should equal negative_count / positive_count."""
    result = calculate_scale_pos_weight(balanced_labels)
    expected = 4 / 2
    assert result == pytest.approx(expected)


def test_calculate_scale_pos_weight_matches_manual_calculation(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """scale_pos_weight should match a manual negative/positive calculation."""
    _, y_train = synthetic_imbalanced_data
    expected = (y_train == 0).sum() / (y_train == 1).sum()

    result = calculate_scale_pos_weight(y_train)

    assert result == pytest.approx(expected)


def test_calculate_scale_pos_weight_raises_on_no_fraud_cases(
    no_fraud_labels: pd.Series,
) -> None:
    """A clear error should be raised when y_train has no fraud examples."""
    with pytest.raises(ValueError, match="no positive"):
        calculate_scale_pos_weight(no_fraud_labels)


def test_calculate_scale_pos_weight_raises_on_empty_series() -> None:
    """An empty y_train should raise a clear, immediate error."""
    with pytest.raises(ValueError, match="empty"):
        calculate_scale_pos_weight(pd.Series([], dtype=int))


def test_calculate_scale_pos_weight_raises_on_invalid_labels() -> None:
    """Labels outside {0, 1} should raise a clear error."""
    with pytest.raises(ValueError, match="only 0 and 1"):
        calculate_scale_pos_weight(pd.Series([0, 1, 2]))


# ---------------------------------------------------------------------------
# build_xgboost_classifier
# ---------------------------------------------------------------------------


def test_build_xgboost_classifier_returns_object_with_fit_and_predict_proba() -> None:
    """The builder should return an unfitted classifier with the expected API."""
    model = build_xgboost_classifier(scale_pos_weight=5.0, random_state=42)

    assert hasattr(model, "fit")
    assert callable(model.fit)
    assert hasattr(model, "predict_proba")
    assert callable(model.predict_proba)


def test_build_xgboost_classifier_uses_expected_scale_pos_weight() -> None:
    """The configured scale_pos_weight should be stored on the model."""
    model = build_xgboost_classifier(scale_pos_weight=7.5, random_state=42)
    assert model.get_params()["scale_pos_weight"] == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# train_xgboost_model
# ---------------------------------------------------------------------------


def test_train_xgboost_model_fits_on_tiny_synthetic_dataset(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """The model should train successfully and be usable for prediction."""
    X_train, y_train = synthetic_imbalanced_data

    model = train_xgboost_model(X_train, y_train, random_state=42)

    predictions = model.predict(X_train)
    assert len(predictions) == len(y_train)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_train_xgboost_model_predict_proba_values_are_between_zero_and_one(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """predict_proba output should be valid probabilities in [0, 1]."""
    X_train, y_train = synthetic_imbalanced_data

    model = train_xgboost_model(X_train, y_train, random_state=42)
    probabilities = model.predict_proba(X_train)

    assert probabilities.shape == (len(X_train), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    # Each row of class probabilities should sum to 1.
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_train_xgboost_model_is_deterministic_with_fixed_random_state(
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Training twice with the same random_state should give the same results."""
    X_train, y_train = synthetic_imbalanced_data

    model_a = train_xgboost_model(X_train, y_train, random_state=42)
    model_b = train_xgboost_model(X_train, y_train, random_state=42)

    proba_a = model_a.predict_proba(X_train)
    proba_b = model_b.predict_proba(X_train)

    assert np.allclose(proba_a, proba_b)


# ---------------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------------


def test_save_model_writes_joblib_artifact_to_disk(
    tmp_path: Path,
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """save_model should write a loadable joblib file to the given path."""
    import joblib

    X_train, y_train = synthetic_imbalanced_data
    model = train_xgboost_model(X_train, y_train, random_state=42)

    output_path = tmp_path / "models" / "xgboost_baseline.joblib"
    saved_path = save_model(model, output_path)

    assert saved_path == output_path
    assert output_path.exists()

    loaded_model = joblib.load(output_path)
    original_predictions = model.predict(X_train)
    loaded_predictions = loaded_model.predict(X_train)
    assert np.array_equal(original_predictions, loaded_predictions)


def test_save_model_creates_missing_parent_directories(
    tmp_path: Path,
    synthetic_imbalanced_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """save_model should create any missing parent directories automatically."""
    X_train, y_train = synthetic_imbalanced_data
    model = train_xgboost_model(X_train, y_train, random_state=42)

    nested_path = tmp_path / "deeply" / "nested" / "model.joblib"
    assert not nested_path.parent.exists()

    save_model(model, nested_path)

    assert nested_path.exists()
