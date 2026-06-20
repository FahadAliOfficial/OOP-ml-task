"""Tests for FastAPI prediction endpoints and SQLite logging."""

from pathlib import Path

import httpx
import pytest
import yaml

import src.api.main as api_main
from src.api.database import (
    fetch_prediction_logs,
    initialize_database,
    log_prediction,
    validate_table_name,
)
from src.models.inference import ModelNotAvailableError, PredictionResult


def _write_api_config(tmp_path: Path) -> tuple[Path, Path]:
    ml_path = tmp_path / "model.joblib"
    ml_path.write_bytes(b"model")
    dl_dir = tmp_path / "dl-model"
    dl_dir.mkdir()
    (dl_dir / "config.json").write_text("{}", encoding="utf-8")
    sqlite_path = tmp_path / "predictions.db"
    config = {
        "api": {"title": "Test Sentiment API", "version": "test"},
        "paths": {
            "ml_model_path": str(ml_path),
            "dl_model_dir": str(dl_dir),
            "sqlite_path": str(sqlite_path),
        },
        "logging": {"table_name": "prediction_logs"},
    }
    config_path = tmp_path / "api.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, sqlite_path


def _prediction(label: str, model_name: str) -> PredictionResult:
    return PredictionResult(
        label=label,
        confidence=0.8,
        inference_time_ms=1.25,
        model_name=model_name,
    )


def test_database_initialization_and_logging(tmp_path: Path) -> None:
    database = tmp_path / "logs.db"
    initialize_database(database)
    row_id = log_prediction(
        database,
        input_text="Great product",
        model_used="test-model",
        prediction="positive",
        confidence=0.9,
        inference_time_ms=2.0,
    )

    rows = fetch_prediction_logs(database)

    assert row_id == 1
    assert len(rows) == 1
    assert rows[0]["input_text"] == "Great product"
    assert rows[0]["prediction"] == "positive"


def test_database_rejects_unsafe_table_name() -> None:
    with pytest.raises(ValueError):
        validate_table_name("logs; DROP TABLE logs")


@pytest.mark.asyncio
async def test_healthcheck_reports_available_dependencies(tmp_path: Path) -> None:
    config_path, _ = _write_api_config(tmp_path)
    app = api_main.create_app(config_path)

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/healthcheck")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "ml_model_available": True,
        "dl_model_available": True,
        "database_available": True,
    }


@pytest.mark.asyncio
async def test_prediction_endpoints_return_and_log_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, sqlite_path = _write_api_config(tmp_path)
    monkeypatch.setattr(
        api_main,
        "predict_ml_sentiment",
        lambda text, path: _prediction("positive", "tfidf_logistic_regression"),
    )
    monkeypatch.setattr(
        api_main,
        "predict_dl_sentiment",
        lambda text, path: _prediction("negative", "distilbert_sentiment"),
    )
    app = api_main.create_app(config_path)

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            ml_response = await client.post(
                "/predict-ml", json={"text": "  Great product  "}
            )
            dl_response = await client.post(
                "/predict-dl", json={"text": "Terrible product"}
            )

    assert ml_response.status_code == 200
    assert ml_response.json()["prediction"] == "positive"
    assert dl_response.status_code == 200
    assert dl_response.json()["prediction"] == "negative"
    logs = fetch_prediction_logs(sqlite_path)
    assert len(logs) == 2
    assert logs[0]["input_text"] == "Great product"
    assert {row["model_used"] for row in logs} == {
        "tfidf_logistic_regression",
        "distilbert_sentiment",
    }


@pytest.mark.asyncio
async def test_blank_text_is_rejected(tmp_path: Path) -> None:
    config_path, _ = _write_api_config(tmp_path)
    app = api_main.create_app(config_path)

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict-ml", json={"text": "   "})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_model_returns_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _write_api_config(tmp_path)

    def missing_model(text: str, path: str | Path) -> PredictionResult:
        raise ModelNotAvailableError(f"Missing: {path}")

    monkeypatch.setattr(api_main, "predict_ml_sentiment", missing_model)
    app = api_main.create_app(config_path)

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict-ml", json={"text": "test input"})

    assert response.status_code == 503
    assert "Missing" in response.json()["detail"]
