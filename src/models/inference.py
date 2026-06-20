"""Shared inference wrappers for trained sentiment models."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

import joblib

from src.preprocessing.clean_text import clean_for_dl, clean_for_ml


@dataclass(frozen=True)
class PredictionResult:
    """Standard prediction response used by API and evaluation code."""

    label: str
    confidence: float | None
    inference_time_ms: float
    model_name: str


class ModelNotAvailableError(RuntimeError):
    """Raised when a requested trained model artifact does not exist yet."""


def load_ml_model(model_path: str | Path = "artifacts/ml/tfidf_sentiment_pipeline.joblib") -> Any:
    """Load the saved classical ML sentiment model."""
    path = Path(model_path)
    if not path.exists():
        raise ModelNotAvailableError(f"ML model artifact not found: {path}")
    return joblib.load(path)


def predict_ml_sentiment(
    text: str,
    model_path: str | Path = "artifacts/ml/tfidf_sentiment_pipeline.joblib",
) -> PredictionResult:
    """Predict sentiment with the saved classical ML model."""
    model = load_ml_model(model_path)
    cleaned_text = clean_for_ml(text)
    start = time.perf_counter()
    prediction = model.predict([cleaned_text])[0]
    inference_time_ms = (time.perf_counter() - start) * 1000

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(max(probabilities))

    return PredictionResult(
        label=str(prediction),
        confidence=confidence,
        inference_time_ms=float(inference_time_ms),
        model_name="tfidf_logistic_regression",
    )


@lru_cache(maxsize=2)
def load_dl_model(
    model_dir: str | Path = "artifacts/dl/distilbert_sentiment",
) -> tuple[Any, Any, Any]:
    """Load the saved transformer model, tokenizer, and active device."""
    path = Path(model_dir)
    if not path.exists() or not (path / "config.json").exists():
        raise ModelNotAvailableError(f"DL model artifact not found: {path}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    model.eval()
    return model, tokenizer, device


def predict_dl_sentiment(
    text: str,
    model_dir: str | Path = "artifacts/dl/distilbert_sentiment",
) -> PredictionResult:
    """Predict sentiment with the saved transformer model."""
    import torch

    model, tokenizer, device = load_dl_model(model_dir)
    cleaned_text = clean_for_dl(text)
    start = time.perf_counter()
    encoded = tokenizer(
        cleaned_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time_ms = (time.perf_counter() - start) * 1000
    prediction_id = int(probabilities.argmax().item())
    id2label = model.config.id2label
    label = id2label.get(prediction_id, id2label.get(str(prediction_id), str(prediction_id)))

    return PredictionResult(
        label=str(label).lower(),
        confidence=float(probabilities[prediction_id].item()),
        inference_time_ms=float(inference_time_ms),
        model_name="distilbert_sentiment",
    )
