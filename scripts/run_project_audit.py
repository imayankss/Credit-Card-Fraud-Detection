"""Generate the final Day 7 project audit checklist and summary report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FINAL_DIR = Path("reports/final")
DEFAULT_CHECKLIST_PATH = DEFAULT_FINAL_DIR / "project_audit_checklist.md"
DEFAULT_PROJECT_REPORT_PATH = DEFAULT_FINAL_DIR / "final_project_report.md"

REQUIRED_OUTPUTS = {
    "README": Path("README.md"),
    "Day 2 EDA report": Path("reports/day2_eda_summary.md"),
    "Day 3 preprocessing report": Path("reports/day3_preprocessing_summary.md"),
    "Day 4 baseline report": Path("reports/day4_baseline_model_summary.md"),
    "Day 5 model comparison report": Path("reports/model_comparison/day5_model_comparison.md"),
    "Day 6 threshold report": Path("reports/threshold_tuning/day6_threshold_tuning_report.md"),
    "Day 7 SHAP report": Path("reports/explainability/shap_summary_report.md"),
    "Final evaluation JSON": Path("reports/final/final_model_evaluation.json"),
    "Final evaluation report": Path("reports/final/final_evaluation_report.md"),
    "XGBoost model artifact": Path("artifacts/models/xgboost_baseline.joblib"),
    "SHAP feature importance CSV": Path("reports/explainability/shap_feature_importance.csv"),
    "SHAP top features JSON": Path("reports/explainability/shap_top_features.json"),
}

CORE_SCRIPTS = [
    Path("scripts/run_day3_preprocessing.py"),
    Path("scripts/run_day4_baseline_models.py"),
    Path("scripts/run_day5_advanced_models.py"),
    Path("scripts/run_day6_threshold_tuning.py"),
    Path("scripts/run_day7_explainability.py"),
    Path("scripts/run_final_evaluation.py"),
]

CORE_TESTS = [
    Path("tests/test_day3_preprocessing.py"),
    Path("tests/test_day4_baseline_models.py"),
    Path("tests/test_advanced_models.py"),
    Path("tests/test_threshold_tuning.py"),
    Path("tests/test_shap_explainer.py"),
    Path("tests/test_final_evaluation.py"),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run final project audit.")
    parser.add_argument("--checklist-path", type=Path, default=DEFAULT_CHECKLIST_PATH)
    parser.add_argument("--project-report-path", type=Path, default=DEFAULT_PROJECT_REPORT_PATH)
    return parser.parse_args()


def _exists(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size >= 0)


def build_audit_rows() -> list[tuple[str, Path, bool]]:
    """Build audit rows for required outputs, scripts, and tests."""
    rows: list[tuple[str, Path, bool]] = []
    for label, path in REQUIRED_OUTPUTS.items():
        rows.append((label, path, _exists(path)))
    for path in CORE_SCRIPTS:
        rows.append((f"Core script: {path.name}", path, _exists(path)))
    for path in CORE_TESTS:
        rows.append((f"Core test: {path.name}", path, _exists(path)))
    return rows


def write_project_audit_checklist(rows: list[tuple[str, Path, bool]], output_path: Path) -> Path:
    """Write a Markdown checklist for the project audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Project Audit Checklist",
        "",
        f"Generated: {generated}",
        "",
        "| Status | Item | Path |",
        "|---|---|---|",
    ]
    for label, path, ok in rows:
        status = "PASS" if ok else "MISSING"
        lines.append(f"| {status} | {label} | `{path}` |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def write_final_project_report(output_path: Path) -> Path:
    """Write a concise final project report for GitHub/reviewer context."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_eval = _load_json(Path("reports/final/final_model_evaluation.json"))
    selected = _load_json(Path("reports/threshold_tuning/selected_thresholds.json"))
    recall_target = selected.get("recall_target", {})

    lines = [
        "# Final Project Report: Credit Card Fraud Detection",
        "",
        "## Executive Summary",
        "",
        "This project is an end-to-end fraud detection ML pipeline for a highly "
        "imbalanced transaction dataset. It covers data validation, EDA, "
        "leakage-safe preprocessing, baseline models, XGBoost, validation-only "
        "threshold tuning, SHAP explainability, and one locked final test-set "
        "evaluation.",
        "",
        "## Locked Model And Threshold",
        "",
        "- Champion model: `xgboost_baseline`",
        "- Threshold source: Day 6 validation recall-target selection",
        f"- Locked business threshold: `{recall_target.get('threshold', 0.53)}`",
        "",
        "## Final Test Evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| PR-AUC | {final_eval.get('pr_auc', 'N/A')} |",
        f"| ROC-AUC | {final_eval.get('roc_auc', 'N/A')} |",
        f"| Precision | {final_eval.get('precision', 'N/A')} |",
        f"| Recall | {final_eval.get('recall', 'N/A')} |",
        f"| F1-score | {final_eval.get('f1_score', 'N/A')} |",
        f"| True positives | {final_eval.get('true_positives', 'N/A')} |",
        f"| False positives | {final_eval.get('false_positives', 'N/A')} |",
        f"| False negatives | {final_eval.get('false_negatives', 'N/A')} |",
        f"| True negatives | {final_eval.get('true_negatives', 'N/A')} |",
        "",
        "## Integrity Notes",
        "",
        "- The model was selected using validation metrics only.",
        "- The operating threshold was selected using validation data only.",
        "- The test split was used once for final locked evaluation.",
        "- SHAP was used for explanation only, not tuning or feature selection.",
        "",
        "## Key Outputs",
        "",
        "- `reports/explainability/shap_summary_report.md`",
        "- `reports/final/final_evaluation_report.md`",
        "- `reports/final/final_model_evaluation.json`",
        "- `reports/final/project_audit_checklist.md`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_project_audit(
    checklist_path: Path = DEFAULT_CHECKLIST_PATH,
    project_report_path: Path = DEFAULT_PROJECT_REPORT_PATH,
) -> dict[str, Any]:
    """Run the audit and write final reports."""
    rows = build_audit_rows()
    checklist = write_project_audit_checklist(rows, checklist_path)
    project_report = write_final_project_report(project_report_path)
    missing = [(label, str(path)) for label, path, ok in rows if not ok]

    print("Project audit completed")
    print(f"Checklist: {checklist}")
    print(f"Final project report: {project_report}")
    if missing:
        print("Missing items:")
        for label, path in missing:
            print(f"- {label}: {path}")
    else:
        print("All required audit items are present.")

    return {
        "checklist_path": checklist,
        "project_report_path": project_report,
        "missing": missing,
    }


def main() -> None:
    args = parse_args()
    result = run_project_audit(args.checklist_path, args.project_report_path)
    if result["missing"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
