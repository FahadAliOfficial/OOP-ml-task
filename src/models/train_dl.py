"""DistilBERT sentiment fine-tuning with a custom PyTorch loop."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


logger = get_logger(__name__)
LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
TEXT_COLUMN = "text_clean_dl"
TARGET_COLUMN = "sentiment"


class SentimentDataset(Dataset):
    """Tokenized sentiment examples for transformer training."""

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        invalid_labels = sorted(set(labels) - set(LABEL_TO_ID))
        if invalid_labels:
            raise ValueError(f"Unsupported sentiment labels: {invalid_labels}")
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(LABEL_TO_ID[self.labels[index]], dtype=torch.long)
        return item


class EarlyStopping:
    """Track validation improvements and signal when training should stop."""

    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.best_score = float("-inf")
        self.bad_epochs = 0

    def step(self, score: float) -> tuple[bool, bool]:
        """Return `(improved, should_stop)` for the new validation score."""
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, Torch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(path: str | Path) -> tuple[list[str], list[str]]:
    """Load transformer-ready text and labels from a processed split."""
    df = pd.read_csv(path)
    missing = {TEXT_COLUMN, TARGET_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()
    labels = df[TARGET_COLUMN].astype(str).tolist()
    invalid_labels = sorted(set(labels) - set(LABEL_TO_ID))
    if invalid_labels:
        raise ValueError(f"Unsupported labels in {path}: {invalid_labels}")
    return texts, labels


def compute_metrics(label_ids: list[int], prediction_ids: list[int]) -> dict[str, Any]:
    """Compute classification metrics in stable sentiment-label order."""
    precision, recall, f1, support = precision_recall_fscore_support(
        label_ids,
        prediction_ids,
        labels=list(ID_TO_LABEL),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        label_ids,
        prediction_ids,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        label_ids,
        prediction_ids,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(label_ids, prediction_ids)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            ID_TO_LABEL[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in ID_TO_LABEL
        },
        "confusion_matrix": confusion_matrix(
            label_ids,
            prediction_ids,
            labels=list(ID_TO_LABEL),
        ).tolist(),
    }


def create_data_loader(
    texts: list[str],
    labels: list[str],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    use_cuda: bool,
) -> DataLoader:
    """Create a deterministic DataLoader for one dataset split."""
    dataset = SentimentDataset(texts, labels, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=use_cuda,
        num_workers=0,
    )


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_one_epoch(
    model: Any,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> dict[str, Any]:
    """Train for one epoch and return loss plus classification metrics."""
    model.train()
    total_loss = 0.0
    prediction_ids: list[int] = []
    label_ids: list[int] = []

    for batch in data_loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += float(loss.detach().item())
        prediction_ids.extend(outputs.logits.detach().argmax(dim=1).cpu().tolist())
        label_ids.extend(batch["labels"].detach().cpu().tolist())

    metrics = compute_metrics(label_ids, prediction_ids)
    metrics["loss"] = total_loss / max(len(data_loader), 1)
    return metrics


@torch.no_grad()
def evaluate_model(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate the model without gradient tracking."""
    model.eval()
    total_loss = 0.0
    prediction_ids: list[int] = []
    label_ids: list[int] = []

    for batch in data_loader:
        batch = _move_batch(batch, device)
        outputs = model(**batch)
        total_loss += float(outputs.loss.detach().item())
        prediction_ids.extend(outputs.logits.detach().argmax(dim=1).cpu().tolist())
        label_ids.extend(batch["labels"].detach().cpu().tolist())

    metrics = compute_metrics(label_ids, prediction_ids)
    metrics["loss"] = total_loss / max(len(data_loader), 1)
    return metrics


def directory_size_mb(path: str | Path) -> float:
    """Return recursive directory size in megabytes."""
    directory = Path(path)
    return sum(file.stat().st_size for file in directory.rglob("*") if file.is_file()) / (
        1024 * 1024
    )


@torch.no_grad()
def measure_inference_speed(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    device: torch.device,
) -> float:
    """Measure batch-size-1 inference latency in milliseconds per sample."""
    model.eval()
    samples = texts[: min(30, len(texts))]
    if not samples:
        return 0.0

    warmup = tokenizer(
        samples[0],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    warmup = {key: value.to(device) for key, value in warmup.items()}
    model(**warmup)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for text in samples:
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model(**encoded)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return ((time.perf_counter() - start) / len(samples)) * 1000


def train_dl_model(config_path: str | Path = "src/config/dl.yaml") -> dict[str, Any]:
    """Fine-tune DistilBERT, save the best checkpoint, and evaluate it."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    config = load_yaml_config(config_path)
    seed = int(config["project"]["seed"])
    set_reproducible_seed(seed)
    paths = config["paths"]
    model_config = config["model"]
    training_config = config["training"]
    model_dir = Path(paths["model_dir"])
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info("Training transformer on device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["pretrained_name"],
        num_labels=int(model_config["num_labels"]),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    split_values = {
        "train": load_split(paths["train_data"]),
        "validation": load_split(paths["validation_data"]),
        "test": load_split(paths["test_data"]),
    }
    max_length = int(model_config["max_length"])
    batch_size = int(training_config["batch_size"])
    loaders = {
        name: create_data_loader(
            texts,
            labels,
            tokenizer,
            max_length,
            batch_size,
            shuffle=name == "train",
            use_cuda=use_amp,
        )
        for name, (texts, labels) in split_values.items()
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    total_steps = len(loaders["train"]) * epochs
    warmup_steps = int(total_steps * float(training_config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stopping = EarlyStopping(int(training_config["early_stopping_patience"]))

    history: list[dict[str, Any]] = []
    best_epoch = 0
    stopped_early = False
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scheduler,
            device,
            scaler,
            use_amp,
        )
        validation_metrics = evaluate_model(model, loaders["validation"], device)
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
        logger.info(
            "Epoch %d/%d | train F1 %.4f | validation F1 %.4f",
            epoch,
            epochs,
            train_metrics["macro_f1"],
            validation_metrics["macro_f1"],
        )

        improved, should_stop = early_stopping.step(validation_metrics["macro_f1"])
        if improved:
            best_epoch = epoch
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        if should_stop:
            stopped_early = True
            logger.info("Early stopping triggered after epoch %d.", epoch)
            break

    training_time_seconds = time.perf_counter() - training_start
    best_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    final_metrics = {
        name: evaluate_model(best_model, loader, device)
        for name, loader in loaders.items()
    }
    inference_time_ms = measure_inference_speed(
        best_model,
        tokenizer,
        split_values["test"][0],
        max_length,
        device,
    )

    metrics = {
        "model_name": "distilbert_sentiment",
        "pretrained_name": model_config["pretrained_name"],
        "device": str(device),
        "label_order": LABEL_ORDER,
        "label_to_id": LABEL_TO_ID,
        "text_column": TEXT_COLUMN,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(early_stopping.best_score),
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "history": history,
        "splits": final_metrics,
        "training_time_seconds": float(training_time_seconds),
        "model_size_mb": float(directory_size_mb(model_dir)),
        "inference_time_ms_per_sample": float(inference_time_ms),
        "training_config": training_config,
        "model_config": model_config,
    }
    metrics_path = Path(paths["metrics_path"])
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved best transformer checkpoint to %s", model_dir)
    logger.info("Test macro F1: %.4f", final_metrics["test"]["macro_f1"])
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT sentiment model.")
    parser.add_argument(
        "--config",
        default="src/config/dl.yaml",
        help="Path to the DL YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dl_model(args.config)
