"""Synthetic tests for Day 7 SHAP explainability helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import (
    build_shap_feature_importance,
    plot_shap_summary_bar,
    sample_explanation_data,
    save_shap_outputs,
    write_shap_markdown_report,
)


def test_sample_explanation_data_caps_sample_size() -> None:
    X = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    sample = sample_explanation_data(X, sample_size=5, random_state=42)

    assert len(sample) == 5
    assert list(sample.columns) == ["a", "b"]


def test_sample_explanation_data_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError):
        sample_explanation_data(pd.DataFrame())


def test_build_shap_feature_importance_sorts_mean_abs_values() -> None:
    shap_values = np.array([[1.0, -3.0, 0.5], [-1.0, 1.0, -0.5]])
    df = build_shap_feature_importance(shap_values, ["f1", "f2", "f3"])

    assert list(df["feature"]) == ["f2", "f1", "f3"]
    assert df.iloc[0]["mean_abs_shap_value"] == pytest.approx(2.0)


def test_build_shap_feature_importance_rejects_feature_mismatch() -> None:
    with pytest.raises(ValueError):
        build_shap_feature_importance(np.ones((3, 2)), ["only_one"])


def test_save_shap_outputs_plot_and_report(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature": ["V1", "V2"],
            "mean_abs_shap_value": [0.25, 0.10],
        }
    )

    outputs = save_shap_outputs(df, tmp_path / "explainability")
    plot_path = plot_shap_summary_bar(df, tmp_path / "figures" / "shap.png")
    report_path = write_shap_markdown_report(df, tmp_path / "report.md")

    assert outputs["csv"].exists()
    assert outputs["json"].exists()
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert report_path.exists()
    assert "SHAP" in report_path.read_text(encoding="utf-8")
