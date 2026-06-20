"""Tests for transformer training and inference utilities."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.models.inference as inference_module
from src.models.inference import ModelNotAvailableError, PredictionResult
from src.models.train_dl import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    EarlyStopping,
    SentimentDataset,
    compute_metrics,
    directory_size_mb,
)


class FakeTokenizer:
    """Small tokenizer double that avoids model downloads."""

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        max_length = int(kwargs.get("max_length", 8))
        token_count = min(max(len(text.split()), 1), max_length)
        input_ids = [1] * token_count + [0] * (max_length - token_count)
        attention_mask = [1] * token_count + [0] * (max_length - token_count)
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label=ID_TO_LABEL)

    def __call__(self, **_: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=torch.tensor([[0.1, 0.2, 1.2]]))


def test_label_mapping_is_stable() -> None:
    assert LABEL_TO_ID == {"negative": 0, "neutral": 1, "positive": 2}
    assert ID_TO_LABEL == {0: "negative", 1: "neutral", 2: "positive"}


def test_sentiment_dataset_tokenizes_and_encodes_label() -> None:
    dataset = SentimentDataset(
        ["A useful product"],
        ["positive"],
        FakeTokenizer(),
        max_length=8,
    )

    item = dataset[0]

    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["labels"].item() == 2


def test_compute_metrics_returns_stable_confusion_matrix() -> None:
    metrics = compute_metrics([0, 1, 2], [0, 2, 2])

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert "macro_f1" in metrics
    assert set(metrics["per_class"]) == set(LABEL_TO_ID)
    assert len(metrics["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in metrics["confusion_matrix"])


def test_early_stopping_tracks_improvement_and_patience() -> None:
    stopping = EarlyStopping(patience=2)

    assert stopping.step(0.4) == (True, False)
    assert stopping.step(0.39) == (False, False)
    assert stopping.step(0.38) == (False, True)
    assert stopping.step(0.5) == (True, False)


def test_directory_size_mb(tmp_path: Path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"x" * 1024)

    assert directory_size_mb(tmp_path) == pytest.approx(1024 / (1024 * 1024))


def test_load_dl_model_missing_path_raises() -> None:
    with pytest.raises(ModelNotAvailableError):
        inference_module.load_dl_model("missing-dl-model")


def test_predict_dl_sentiment_with_mocked_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inference_module,
        "load_dl_model",
        lambda model_dir: (FakeModel(), FakeTokenizer(), torch.device("cpu")),
    )

    result = inference_module.predict_dl_sentiment("I really liked this", "mock-model")

    assert isinstance(result, PredictionResult)
    assert result.label == "positive"
    assert result.confidence is not None
    assert result.confidence > 0.5
    assert result.inference_time_ms >= 0
