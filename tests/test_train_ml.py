"""Tests for classical ML training utilities."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from src.models.inference import (
    ModelNotAvailableError,
    PredictionResult,
    load_ml_model,
    predict_ml_sentiment,
)
from src.models.train_ml import (
    LABEL_ORDER,
    build_param_grid,
    evaluate_predictions,
    train_ml_model,
)


def test_build_param_grid_maps_config_to_pipeline_names() -> None:
    config = {
        "model": {
            "vectorizer": {
                "ngram_range": [[1, 1], [1, 2]],
                "max_features": [1000],
                "min_df": [1],
            },
            "classifier": {
                "type": "logistic_regression",
                "C": [1.0],
                "class_weight": ["balanced"],
            },
        }
    }

    grid = build_param_grid(config)

    assert grid["tfidf__ngram_range"] == [(1, 1), (1, 2)]
    assert grid["tfidf__max_features"] == [1000]
    assert grid["classifier__C"] == [1.0]
    assert grid["classifier__class_weight"] == ["balanced"]


def test_evaluate_predictions_returns_required_metric_keys() -> None:
    y_true = pd.Series(["positive", "negative", "neutral"])
    y_pred = pd.Series(["positive", "neutral", "neutral"])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert set(metrics["per_class"]) == set(LABEL_ORDER)
    assert len(metrics["confusion_matrix"]) == 3


def _write_tiny_split(path: Path) -> None:
    rows = [
        ("i love this excellent product", "positive"),
        ("wonderful happy useful item", "positive"),
        ("this is terrible and broken", "negative"),
        ("awful bad disappointing item", "negative"),
        ("it arrived yesterday as described", "neutral"),
        ("standard package with basic details", "neutral"),
    ]
    df = pd.DataFrame(
        {
            "text_clean_ml": [row[0] for row in rows],
            "sentiment": [row[1] for row in rows],
        }
    )
    df.to_csv(path, index=False)


def test_train_ml_model_on_tiny_fixture(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    _write_tiny_split(train_path)
    _write_tiny_split(validation_path)
    _write_tiny_split(test_path)

    config = {
        "project": {"seed": 42},
        "paths": {
            "train_data": str(train_path),
            "validation_data": str(validation_path),
            "test_data": str(test_path),
            "artifact_dir": str(tmp_path),
            "model_path": str(tmp_path / "model.joblib"),
            "metrics_path": str(tmp_path / "metrics.json"),
        },
        "model": {
            "vectorizer": {
                "ngram_range": [[1, 1]],
                "max_features": [100],
                "min_df": [1],
            },
            "classifier": {
                "type": "logistic_regression",
                "C": [1.0],
                "class_weight": ["balanced"],
            },
            "grid_search": {
                "cv": 2,
                "scoring": "f1_macro",
                "n_jobs": 1,
            },
        },
    }
    config_path = tmp_path / "ml.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    metrics = train_ml_model(config_path)

    assert Path(config["paths"]["model_path"]).exists()
    assert Path(config["paths"]["metrics_path"]).exists()
    saved_metrics = json.loads(Path(config["paths"]["metrics_path"]).read_text())
    assert saved_metrics["model_name"] == "tfidf_logistic_regression"
    assert "test" in metrics["splits"]
    assert metrics["grid_search"]["candidate_count"] == 1


def test_load_ml_model_and_predict_wrapper(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    _write_tiny_split(train_path)
    model_path = tmp_path / "model.joblib"

    from src.models.train_ml import build_pipeline

    model = build_pipeline(seed=42)
    df = pd.read_csv(train_path)
    model.fit(df["text_clean_ml"], df["sentiment"])
    joblib.dump(model, model_path)

    loaded_model = load_ml_model(model_path)
    assert loaded_model.predict(["excellent useful item"])[0] in LABEL_ORDER

    result = predict_ml_sentiment("This product is excellent and useful", model_path)
    assert isinstance(result, PredictionResult)
    assert result.label in LABEL_ORDER
    assert result.confidence is not None
    assert result.inference_time_ms >= 0


def test_load_ml_model_missing_path_raises() -> None:
    with pytest.raises(ModelNotAvailableError):
        load_ml_model("missing-model.joblib")
