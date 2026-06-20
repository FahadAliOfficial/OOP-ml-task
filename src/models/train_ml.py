"""Classical TF-IDF sentiment model training."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


logger = get_logger(__name__)
LABEL_ORDER = ["negative", "neutral", "positive"]
TEXT_COLUMN = "text_clean_ml"
TARGET_COLUMN = "sentiment"


def load_split(path: str | Path) -> tuple[pd.Series, pd.Series]:
    """Load a processed split and return ML text with labels."""
    df = pd.read_csv(path)
    missing = {TEXT_COLUMN, TARGET_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df[TEXT_COLUMN].fillna("").astype(str), df[TARGET_COLUMN].astype(str)


def build_pipeline(seed: int) -> Pipeline:
    """Build the TF-IDF plus Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(strip_accents="unicode")),
            (
                "classifier",
                LogisticRegression(
                    random_state=seed,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_param_grid(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Convert ml.yaml settings into sklearn pipeline grid parameters."""
    model_config = config["model"]
    vectorizer = model_config["vectorizer"]
    classifier = model_config["classifier"]

    if classifier.get("type") != "logistic_regression":
        raise ValueError("Phase 4 supports classifier.type=logistic_regression only.")

    return {
        "tfidf__ngram_range": [tuple(value) for value in vectorizer["ngram_range"]],
        "tfidf__max_features": vectorizer["max_features"],
        "tfidf__min_df": vectorizer["min_df"],
        "classifier__C": classifier["C"],
        "classifier__class_weight": classifier["class_weight"],
    }


def evaluate_predictions(y_true: pd.Series, y_pred: list[str] | pd.Series) -> dict[str, Any]:
    """Compute classification metrics with stable label order."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABEL_ORDER)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
        ).tolist(),
    }


def measure_inference_speed(model: Pipeline, texts: pd.Series) -> float:
    """Return average model inference time in milliseconds per sample."""
    if texts.empty:
        return 0.0
    start = time.perf_counter()
    model.predict(texts)
    elapsed = time.perf_counter() - start
    return float((elapsed / len(texts)) * 1000)


def train_ml_model(config_path: str | Path = "src/config/ml.yaml") -> dict[str, Any]:
    """Train, evaluate, and save the TF-IDF sentiment classifier."""
    config = load_yaml_config(config_path)
    seed = int(config["project"]["seed"])
    paths = config["paths"]
    artifact_dir = Path(paths["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(paths["train_data"])
    x_validation, y_validation = load_split(paths["validation_data"])
    x_test, y_test = load_split(paths["test_data"])

    pipeline = build_pipeline(seed)
    param_grid = build_param_grid(config)
    grid_config = config["model"]["grid_search"]
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=grid_config["scoring"],
        cv=int(grid_config["cv"]),
        n_jobs=int(grid_config["n_jobs"]),
        refit=True,
        return_train_score=True,
    )

    logger.info("Starting GridSearchCV for classical ML model.")
    train_start = time.perf_counter()
    search.fit(x_train, y_train)
    training_time_seconds = time.perf_counter() - train_start

    best_model = search.best_estimator_
    model_path = Path(paths["model_path"])
    joblib.dump(best_model, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    split_data = {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
    }
    metrics_by_split = {}
    for split_name, (x_values, y_values) in split_data.items():
        predictions = best_model.predict(x_values)
        metrics_by_split[split_name] = evaluate_predictions(y_values, predictions)

    inference_time_ms = measure_inference_speed(best_model, x_test)
    metrics = {
        "model_name": "tfidf_logistic_regression",
        "label_order": LABEL_ORDER,
        "text_column": TEXT_COLUMN,
        "target_column": TARGET_COLUMN,
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "grid_search": {
            "cv": int(grid_config["cv"]),
            "scoring": grid_config["scoring"],
            "n_jobs": int(grid_config["n_jobs"]),
            "candidate_count": len(search.cv_results_["params"]),
        },
        "training_time_seconds": float(training_time_seconds),
        "model_size_mb": float(model_size_mb),
        "inference_time_ms_per_sample": inference_time_ms,
        "splits": metrics_by_split,
    }

    metrics_path = Path(paths["metrics_path"])
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Saved ML model to %s", model_path)
    logger.info("Saved ML metrics to %s", metrics_path)
    logger.info("Validation macro F1: %.4f", metrics_by_split["validation"]["macro_f1"])
    logger.info("Test macro F1: %.4f", metrics_by_split["test"]["macro_f1"])
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the classical ML model.")
    parser.add_argument(
        "--config",
        default="src/config/ml.yaml",
        help="Path to the ML YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ml_model(args.config)
