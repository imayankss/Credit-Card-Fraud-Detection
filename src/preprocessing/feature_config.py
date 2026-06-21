"""
Feature configuration for the Credit Card Fraud Detection & Risk Scoring System.

This module centralizes feature group definitions, the target column name,
and train/validation/test split constants used throughout the preprocessing
pipeline. Other modules should import from here instead of hardcoding column
names or split ratios.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

TARGET_COLUMN: str = "Class"
TIME_COLUMN: str = "Time"
AMOUNT_COLUMN: str = "Amount"

PCA_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]

SCALE_FEATURES: List[str] = [TIME_COLUMN, AMOUNT_COLUMN]
PASSTHROUGH_FEATURES: List[str] = PCA_FEATURES

ALL_FEATURES: List[str] = [TIME_COLUMN] + PCA_FEATURES + [AMOUNT_COLUMN]
REQUIRED_COLUMNS: List[str] = ALL_FEATURES + [TARGET_COLUMN]

# ---------------------------------------------------------------------------
# Split / reproducibility constants
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TRAIN_SIZE: float = 0.70
VALIDATION_SIZE: float = 0.15
TEST_SIZE: float = 0.15


def get_pca_features() -> List[str]:
    """Return the list of PCA-transformed feature column names (V1-V28)."""
    return list(PCA_FEATURES)


def get_scale_features() -> List[str]:
    """Return the list of feature columns that should be scaled."""
    return list(SCALE_FEATURES)


def get_passthrough_features() -> List[str]:
    """Return the list of feature columns that should pass through unscaled."""
    return list(PASSTHROUGH_FEATURES)


def get_all_features() -> List[str]:
    """Return the full ordered list of model input feature columns."""
    return list(ALL_FEATURES)


def get_required_columns() -> List[str]:
    """Return the full list of columns required in the raw dataset."""
    return list(REQUIRED_COLUMNS)


def validate_feature_config() -> None:
    """
    Validate that the feature configuration is internally consistent.

    Checks performed:
        - PCA features are exactly V1 through V28, in order.
        - Time and Amount are included in ALL_FEATURES.
        - The target column is NOT included in ALL_FEATURES.
        - The target column IS included in REQUIRED_COLUMNS.
        - No duplicate columns exist in ALL_FEATURES or REQUIRED_COLUMNS.
        - Train/validation/test split sizes sum to 1.0.

    Raises:
        ValueError: If any of the above checks fail.
    """
    expected_pca_features = [f"V{i}" for i in range(1, 29)]
    if PCA_FEATURES != expected_pca_features:
        raise ValueError(
            "PCA_FEATURES must be exactly V1 through V28, in order. "
            f"Got: {PCA_FEATURES}"
        )

    if TIME_COLUMN not in ALL_FEATURES:
        raise ValueError(
            f"'{TIME_COLUMN}' must be included in ALL_FEATURES."
        )

    if AMOUNT_COLUMN not in ALL_FEATURES:
        raise ValueError(
            f"'{AMOUNT_COLUMN}' must be included in ALL_FEATURES."
        )

    if TARGET_COLUMN in ALL_FEATURES:
        raise ValueError(
            f"'{TARGET_COLUMN}' must NOT be included in ALL_FEATURES. "
            "The target column must never be used as a model feature."
        )

    if TARGET_COLUMN not in REQUIRED_COLUMNS:
        raise ValueError(
            f"'{TARGET_COLUMN}' must be included in REQUIRED_COLUMNS."
        )

    if len(ALL_FEATURES) != len(set(ALL_FEATURES)):
        raise ValueError("ALL_FEATURES contains duplicate column names.")

    if len(REQUIRED_COLUMNS) != len(set(REQUIRED_COLUMNS)):
        raise ValueError("REQUIRED_COLUMNS contains duplicate column names.")

    split_total = TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE
    if abs(split_total - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_SIZE, VALIDATION_SIZE, and TEST_SIZE must sum to 1.0. "
            f"Got: {split_total}"
        )


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return a dictionary summarizing all feature groups.

    Returns:
        A dictionary with keys: "scale_features", "passthrough_features",
        "all_features", and "required_columns".
    """
    return {
        "scale_features": get_scale_features(),
        "passthrough_features": get_passthrough_features(),
        "all_features": get_all_features(),
        "required_columns": get_required_columns(),
    }


if __name__ == "__main__":
    validate_feature_config()

    groups = get_feature_groups()

    print("Feature configuration validated successfully.\n")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Scale features: {groups['scale_features']}")
    print(
        f"Passthrough features ({len(groups['passthrough_features'])}): "
        f"{groups['passthrough_features'][0]} ... "
        f"{groups['passthrough_features'][-1]}"
    )
    print(f"All features ({len(groups['all_features'])} total): {groups['all_features']}")
    print(f"Required columns ({len(groups['required_columns'])} total): {groups['required_columns']}")
    print(f"Random state: {RANDOM_STATE}")
    print(
        f"Split sizes -> train: {TRAIN_SIZE}, "
        f"validation: {VALIDATION_SIZE}, test: {TEST_SIZE}"
    )
