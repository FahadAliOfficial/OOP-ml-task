"""FastAPI service for classical and transformer sentiment prediction."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException

from src.api.database import initialize_database, log_prediction
from src.api.schemas import (
    HealthCheckResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.models.inference import (
    ModelNotAvailableError,
    PredictionResult,
    predict_dl_sentiment,
    predict_ml_sentiment,
)
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


logger = get_logger(__name__)


def _to_response(result: PredictionResult) -> PredictionResponse:
    return PredictionResponse(
        prediction=result.label,
        model_used=result.model_name,
        confidence=result.confidence,
        inference_time_ms=result.inference_time_ms,
    )


def _run_prediction(
    *,
    text: str,
    predictor: Callable[[str, str | Path], PredictionResult],
    model_path: str | Path,
    sqlite_path: str | Path,
    table_name: str,
) -> PredictionResponse:
    try:
        result = predictor(text, model_path)
        log_prediction(
            sqlite_path,
            input_text=text,
            model_used=result.model_name,
            prediction=result.label,
            confidence=result.confidence,
            inference_time_ms=result.inference_time_ms,
            table_name=table_name,
        )
        return _to_response(result)
    except ModelNotAvailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Prediction request failed.")
        raise HTTPException(status_code=500, detail="Prediction failed") from error


def create_app(config_path: str | Path = "src/config/api.yaml") -> FastAPI:
    """Create a configured sentiment API application."""
    config = load_yaml_config(config_path)
    api_config = config["api"]
    paths = config["paths"]
    table_name = config.get("logging", {}).get("table_name", "prediction_logs")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        initialize_database(paths["sqlite_path"], table_name)
        application.state.database_initialized = True
        logger.info("Prediction database initialized at %s", paths["sqlite_path"])
        yield

    application = FastAPI(
        title=api_config["title"],
        version=str(api_config["version"]),
        lifespan=lifespan,
    )
    application.state.api_config = config
    application.state.database_initialized = False

    @application.get("/healthcheck", response_model=HealthCheckResponse)
    def healthcheck() -> HealthCheckResponse:
        ml_available = Path(paths["ml_model_path"]).is_file()
        dl_available = (Path(paths["dl_model_dir"]) / "config.json").is_file()
        database_available = Path(paths["sqlite_path"]).is_file()
        status = "ok" if ml_available and dl_available and database_available else "degraded"
        return HealthCheckResponse(
            status=status,
            ml_model_available=ml_available,
            dl_model_available=dl_available,
            database_available=database_available,
        )

    @application.post("/predict-ml", response_model=PredictionResponse)
    def predict_ml(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(
            text=request.text,
            predictor=predict_ml_sentiment,
            model_path=paths["ml_model_path"],
            sqlite_path=paths["sqlite_path"],
            table_name=table_name,
        )

    @application.post("/predict-dl", response_model=PredictionResponse)
    def predict_dl(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(
            text=request.text,
            predictor=predict_dl_sentiment,
            model_path=paths["dl_model_dir"],
            sqlite_path=paths["sqlite_path"],
            table_name=table_name,
        )

    return application


app = create_app()
