"""Pydantic schemas for the FastAPI service."""

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Request body for sentiment prediction endpoints."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text to classify.",
    )

    @field_validator("text")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        """Reject whitespace-only input and normalize its outer whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class PredictionResponse(BaseModel):
    """Response body returned by prediction endpoints."""

    prediction: str
    model_used: str
    inference_time_ms: float
    confidence: float | None = None


class HealthCheckResponse(BaseModel):
    """Health and artifact availability returned by the service."""

    status: str
    ml_model_available: bool
    dl_model_available: bool
    database_available: bool
