"""Tests for model-comparison artifact generation."""

import json
from pathlib import Path

import pytest

from src.evaluation.evaluate_models import (
    LABEL_ORDER,
    MetricsValidationError,
    build_overall_table,
    build_per_class_table,
    build_summary,
    evaluate_models,
    load_metrics,
    validate_metrics,
)


def _metrics(model: str, score: float) -> dict:
    per_class = {
        label: {
            "precision": score,
            "recall": score,
            "f1": score,
            "support": 10,
        }
        for label in LABEL_ORDER
    }
    split_metrics = {
        "accuracy": score,
        "macro_precision": score,
        "macro_recall": score,
        "macro_f1": score,
        "weighted_f1": score,
        "per_class": per_class,
        "confusion_matrix": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        "loss": 0.5,
    }
    metrics = {
        "model_name": model,
        "label_order": LABEL_ORDER,
        "splits": {"test": split_metrics},
        "training_time_seconds": 5.0,
        "model_size_mb": 2.0,
        "inference_time_ms_per_sample": 1.0,
    }
    if model == "dl":
        metrics["history"] = [
            {
                "epoch": 1,
                "train": {"loss": 0.7, "macro_f1": 0.5},
                "validation": {"loss": 0.6, "macro_f1": score},
            }
        ]
    return metrics


def test_load_metrics_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(MetricsValidationError):
        load_metrics(path)


def test_validate_metrics_rejects_label_mismatch() -> None:
    ml = _metrics("ml", 0.5)
    dl = _metrics("dl", 0.7)
    dl["label_order"] = list(reversed(LABEL_ORDER))

    with pytest.raises(MetricsValidationError, match="label orders"):
        validate_metrics(ml, dl)


def test_comparison_tables_have_expected_shapes() -> None:
    ml = _metrics("ml", 0.5)
    dl = _metrics("dl", 0.7)

    overall = build_overall_table(ml, dl)
    per_class = build_per_class_table(ml, dl)

    assert overall.shape == (2, 6)
    assert per_class.shape == (6, 6)
    assert set(per_class["label"]) == set(LABEL_ORDER)


def test_summary_calculates_f1_gain() -> None:
    summary = build_summary(_metrics("ml", 0.5), _metrics("dl", 0.7))

    assert summary["absolute_macro_f1_gain"] == pytest.approx(0.2)
    assert summary["relative_macro_f1_gain_percent"] == pytest.approx(40.0)
    assert summary["recommended_model"] == "DistilBERT"


def test_evaluate_models_writes_outputs_and_charts(tmp_path: Path) -> None:
    ml_path = tmp_path / "ml.json"
    dl_path = tmp_path / "dl.json"
    output_dir = tmp_path / "evaluation"
    ml_path.write_text(json.dumps(_metrics("ml", 0.5)), encoding="utf-8")
    dl_path.write_text(json.dumps(_metrics("dl", 0.7)), encoding="utf-8")

    result = evaluate_models(ml_path, dl_path, output_dir)

    assert (output_dir / "model_comparison.json").exists()
    assert (output_dir / "overall_metrics.csv").exists()
    assert (output_dir / "per_class_metrics.csv").exists()
    assert (output_dir / "operational_metrics.csv").exists()
    assert len(result["charts"]) == 9
    assert all(Path(path).exists() for path in result["charts"])
