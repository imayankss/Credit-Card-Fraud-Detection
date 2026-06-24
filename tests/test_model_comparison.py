"""
Tests for src.models.model_comparison (Day 5).

These tests use only small, fake, in-memory metric dictionaries written to
temporary directories. They never load the real Kaggle dataset, never read
Day 3/4 artifacts from the real project folders, and never use test-set
metrics — only validation-style metric dictionaries are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

# Make sure the project root is importable when tests are run from the
# repo root (e.g. `pytest`) or from within the tests/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_comparison import (  # noqa: E402
    REQUIRED_COLUMNS,
    build_model_comparison_table,
    save_model_comparison_outputs,
    select_champion_model,
    write_markdown_comparison_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_metrics_list() -> List[Dict[str, Any]]:
    """
    Fake validation-only metrics for four models, with distinct PR-AUC
    values so the highest one is unambiguous.

    "xgboost_baseline" intentionally has the highest pr_auc so tests can
    assert it is selected as the champion and sorted to the top.
    """
    return [
        {
            "model_name": "dummy_baseline",
            "pr_auc": 0.01,
            "roc_auc": 0.50,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "validation_rows": 42721,
            "validation_frauds": 74,
        },
        {
            "model_name": "logistic_regression",
            "pr_auc": 0.62,
            "roc_auc": 0.95,
            "precision": 0.71,
            "recall": 0.80,
            "f1": 0.75,
            "validation_rows": 42721,
            "validation_frauds": 74,
        },
        {
            "model_name": "random_forest",
            "pr_auc": 0.78,
            "roc_auc": 0.97,
            "precision": 0.85,
            "recall": 0.81,
            "f1": 0.83,
            "validation_rows": 42721,
            "validation_frauds": 74,
        },
        {
            "model_name": "xgboost_baseline",
            "pr_auc": 0.86,
            "roc_auc": 0.98,
            "precision": 0.88,
            "recall": 0.84,
            "f1": 0.86,
            "validation_rows": 42721,
            "validation_frauds": 74,
        },
    ]


@pytest.fixture
def fake_metrics_missing_column() -> List[Dict[str, Any]]:
    """Fake metrics where every row is missing the 'recall' column entirely."""
    return [
        {
            "model_name": "dummy_baseline",
            "pr_auc": 0.01,
            "roc_auc": 0.50,
            "precision": 0.0,
            "f1": 0.0,
            "validation_rows": 42721,
            "validation_frauds": 74,
        },
        {
            "model_name": "logistic_regression",
            "pr_auc": 0.62,
            "roc_auc": 0.95,
            "precision": 0.71,
            "f1": 0.75,
            "validation_rows": 42721,
            "validation_frauds": 74,
        },
    ]


@pytest.fixture
def fake_comparison_df(fake_metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """A built comparison DataFrame derived from fake_metrics_list."""
    return build_model_comparison_table(fake_metrics_list)


# ---------------------------------------------------------------------------
# build_model_comparison_table
# ---------------------------------------------------------------------------


def test_build_model_comparison_table_creates_dataframe_with_required_columns(
    fake_metrics_list: List[Dict[str, Any]],
) -> None:
    """The resulting DataFrame should contain exactly the required columns."""
    comparison_df = build_model_comparison_table(fake_metrics_list)

    assert isinstance(comparison_df, pd.DataFrame)
    assert list(comparison_df.columns) == REQUIRED_COLUMNS
    assert len(comparison_df) == len(fake_metrics_list)


def test_build_model_comparison_table_is_sorted_by_pr_auc_descending(
    fake_metrics_list: List[Dict[str, Any]],
) -> None:
    """Rows should be ordered from highest to lowest pr_auc."""
    comparison_df = build_model_comparison_table(fake_metrics_list)

    pr_auc_values = comparison_df["pr_auc"].tolist()
    assert pr_auc_values == sorted(pr_auc_values, reverse=True)
    assert comparison_df.iloc[0]["model_name"] == "xgboost_baseline"
    assert comparison_df.iloc[-1]["model_name"] == "dummy_baseline"


def test_build_model_comparison_table_raises_on_empty_metrics_list() -> None:
    """An empty metrics_list should raise a clear, immediate error."""
    with pytest.raises(ValueError, match="empty"):
        build_model_comparison_table([])


def test_build_model_comparison_table_raises_on_missing_required_column(
    fake_metrics_missing_column: List[Dict[str, Any]],
) -> None:
    """A metrics_list missing a required column should raise a clear error."""
    with pytest.raises(ValueError, match="recall"):
        build_model_comparison_table(fake_metrics_missing_column)


# ---------------------------------------------------------------------------
# select_champion_model
# ---------------------------------------------------------------------------


def test_select_champion_model_returns_highest_pr_auc_model(
    fake_comparison_df: pd.DataFrame,
) -> None:
    """The champion should be the model with the highest pr_auc."""
    champion = select_champion_model(fake_comparison_df, primary_metric="pr_auc")
    assert champion == "xgboost_baseline"


def test_select_champion_model_raises_on_empty_dataframe() -> None:
    """An empty comparison_df should raise a clear error."""
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="empty"):
        select_champion_model(empty_df, primary_metric="pr_auc")


def test_select_champion_model_raises_on_missing_metric_column(
    fake_comparison_df: pd.DataFrame,
) -> None:
    """An unknown primary_metric column should raise a clear error."""
    with pytest.raises(ValueError, match="not found"):
        select_champion_model(fake_comparison_df, primary_metric="not_a_real_metric")


def test_select_champion_model_ignores_null_pr_auc_values(
    fake_metrics_list: List[Dict[str, Any]],
) -> None:
    """Rows with a null pr_auc should not be selected as champion."""
    metrics_with_null = [dict(row) for row in fake_metrics_list]
    metrics_with_null[-1]["pr_auc"] = None  # previously the highest, now null
    comparison_df = build_model_comparison_table(metrics_with_null)

    champion = select_champion_model(comparison_df, primary_metric="pr_auc")
    assert champion == "random_forest"


# ---------------------------------------------------------------------------
# save_model_comparison_outputs
# ---------------------------------------------------------------------------


def test_save_model_comparison_outputs_creates_csv_and_json(
    tmp_path: Path,
    fake_comparison_df: pd.DataFrame,
) -> None:
    """CSV and JSON comparison files should be written to output_dir."""
    output_dir = tmp_path / "model_comparison"

    saved_paths = save_model_comparison_outputs(fake_comparison_df, output_dir)

    assert "csv" in saved_paths
    assert "json" in saved_paths
    assert saved_paths["csv"].exists()
    assert saved_paths["json"].exists()

    reloaded_csv = pd.read_csv(saved_paths["csv"])
    assert len(reloaded_csv) == len(fake_comparison_df)
    assert "model_name" in reloaded_csv.columns

    with open(saved_paths["json"], "r", encoding="utf-8") as handle:
        reloaded_json = json.load(handle)
    assert len(reloaded_json) == len(fake_comparison_df)
    assert reloaded_json[0]["model_name"] == fake_comparison_df.iloc[0]["model_name"]


def test_save_model_comparison_outputs_creates_missing_directory(
    tmp_path: Path,
    fake_comparison_df: pd.DataFrame,
) -> None:
    """The output directory should be created automatically if missing."""
    output_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not output_dir.exists()

    save_model_comparison_outputs(fake_comparison_df, output_dir)

    assert output_dir.exists()


# ---------------------------------------------------------------------------
# write_markdown_comparison_report
# ---------------------------------------------------------------------------


def test_write_markdown_comparison_report_creates_file_with_expected_content(
    tmp_path: Path,
    fake_comparison_df: pd.DataFrame,
) -> None:
    """The Markdown report should mention model names, the champion, and PR-AUC."""
    champion = select_champion_model(fake_comparison_df, primary_metric="pr_auc")
    report_path = tmp_path / "reports" / "day5_model_comparison.md"

    saved_path = write_markdown_comparison_report(fake_comparison_df, champion, report_path)

    assert saved_path == report_path
    assert report_path.exists()

    content = report_path.read_text(encoding="utf-8")

    # Model names are present.
    for model_name in fake_comparison_df["model_name"]:
        assert model_name in content

    # Champion model is called out explicitly.
    assert champion in content
    assert "champion" in content.lower()

    # PR-AUC is explained, not just listed in the table.
    assert "PR-AUC" in content
    assert "imbalanced" in content.lower() or "rare" in content.lower()


def test_write_markdown_comparison_report_creates_missing_parent_directory(
    tmp_path: Path,
    fake_comparison_df: pd.DataFrame,
) -> None:
    """Missing parent directories for the report path should be created."""
    champion = select_champion_model(fake_comparison_df, primary_metric="pr_auc")
    report_path = tmp_path / "nested" / "reports" / "comparison.md"
    assert not report_path.parent.exists()

    write_markdown_comparison_report(fake_comparison_df, champion, report_path)

    assert report_path.exists()
