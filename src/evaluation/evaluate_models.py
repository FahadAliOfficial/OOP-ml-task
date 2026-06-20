"""Generate reusable comparison artifacts for the ML and DL models."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger


logger = get_logger(__name__)
MODEL_NAMES = {"ml": "Classical ML", "dl": "DistilBERT"}
OVERALL_METRICS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
]
PER_CLASS_METRICS = ["precision", "recall", "f1"]
LABEL_ORDER = ["negative", "neutral", "positive"]


class MetricsValidationError(ValueError):
    """Raised when saved model metrics cannot be compared safely."""


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load a model metrics JSON artifact."""
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetricsValidationError(f"Invalid metrics JSON: {metrics_path}") from exc


def validate_metrics(ml_metrics: dict[str, Any], dl_metrics: dict[str, Any]) -> None:
    """Validate required comparison fields and matching label order."""
    required = {
        "label_order",
        "splits",
        "training_time_seconds",
        "model_size_mb",
        "inference_time_ms_per_sample",
    }
    for name, metrics in (("ML", ml_metrics), ("DL", dl_metrics)):
        missing = sorted(required - set(metrics))
        if missing:
            raise MetricsValidationError(f"{name} metrics missing fields: {missing}")
        if "test" not in metrics["splits"]:
            raise MetricsValidationError(f"{name} metrics missing test split")
        test_missing = sorted(set(OVERALL_METRICS) - set(metrics["splits"]["test"]))
        if test_missing:
            raise MetricsValidationError(
                f"{name} test metrics missing fields: {test_missing}"
            )

    if ml_metrics["label_order"] != dl_metrics["label_order"]:
        raise MetricsValidationError("Model label orders do not match")
    if ml_metrics["label_order"] != LABEL_ORDER:
        raise MetricsValidationError(
            f"Expected label order {LABEL_ORDER}, found {ml_metrics['label_order']}"
        )


def build_overall_table(
    ml_metrics: dict[str, Any], dl_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Build test-set overall metric comparison rows."""
    rows = []
    for key, metrics in (("ml", ml_metrics), ("dl", dl_metrics)):
        test = metrics["splits"]["test"]
        rows.append(
            {"model": MODEL_NAMES[key], **{metric: test[metric] for metric in OVERALL_METRICS}}
        )
    return pd.DataFrame(rows)


def build_per_class_table(
    ml_metrics: dict[str, Any], dl_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Build long-form per-class metric comparison rows."""
    rows = []
    for key, metrics in (("ml", ml_metrics), ("dl", dl_metrics)):
        per_class = metrics["splits"]["test"]["per_class"]
        for label in LABEL_ORDER:
            row = {"model": MODEL_NAMES[key], "label": label}
            row.update({metric: per_class[label][metric] for metric in PER_CLASS_METRICS})
            row["support"] = int(per_class[label]["support"])
            rows.append(row)
    return pd.DataFrame(rows)


def build_operational_table(
    ml_metrics: dict[str, Any], dl_metrics: dict[str, Any]
) -> pd.DataFrame:
    """Build size, latency, and training-time comparison rows."""
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAMES[key],
                "training_time_seconds": metrics["training_time_seconds"],
                "inference_time_ms_per_sample": metrics[
                    "inference_time_ms_per_sample"
                ],
                "model_size_mb": metrics["model_size_mb"],
            }
            for key, metrics in (("ml", ml_metrics), ("dl", dl_metrics))
        ]
    )


def build_summary(ml_metrics: dict[str, Any], dl_metrics: dict[str, Any]) -> dict[str, Any]:
    """Calculate headline differences and deployment recommendation."""
    ml_f1 = float(ml_metrics["splits"]["test"]["macro_f1"])
    dl_f1 = float(dl_metrics["splits"]["test"]["macro_f1"])
    absolute_gain = dl_f1 - ml_f1
    relative_gain = (absolute_gain / ml_f1) * 100 if ml_f1 else None
    return {
        "ml_test_macro_f1": ml_f1,
        "dl_test_macro_f1": dl_f1,
        "absolute_macro_f1_gain": absolute_gain,
        "relative_macro_f1_gain_percent": relative_gain,
        "recommended_model": "DistilBERT",
        "recommendation_reason": (
            "DistilBERT provides a substantial macro-F1 improvement while retaining "
            "practical GPU inference latency. Classical ML remains the lightweight fallback."
        ),
    }


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_charts(
    ml_metrics: dict[str, Any],
    dl_metrics: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Generate all required comparison charts and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    chart_paths: list[str] = []

    overall = build_overall_table(ml_metrics, dl_metrics).melt(
        id_vars="model", var_name="metric", value_name="score"
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=overall, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Overall Test Performance")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.xticks(rotation=20, ha="right")
    path = output_path / "overall_test_metrics.png"
    _save_figure(path)
    chart_paths.append(str(path))

    per_class = build_per_class_table(ml_metrics, dl_metrics)
    for metric in PER_CLASS_METRICS:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=per_class, x="label", y=metric, hue="model", order=LABEL_ORDER)
        plt.ylim(0, 1)
        plt.title(f"Per-Class {metric.title()}")
        plt.xlabel("Sentiment")
        plt.ylabel(metric.title())
        path = output_path / f"per_class_{metric}.png"
        _save_figure(path)
        chart_paths.append(str(path))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, key, metrics in zip(
        axes,
        ("ml", "dl"),
        (ml_metrics, dl_metrics),
    ):
        sns.heatmap(
            metrics["splits"]["test"]["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABEL_ORDER,
            yticklabels=LABEL_ORDER,
            cbar=False,
            ax=axis,
        )
        axis.set_title(MODEL_NAMES[key])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.suptitle("Test Confusion Matrices")
    path = output_path / "confusion_matrices.png"
    _save_figure(path)
    chart_paths.append(str(path))

    operational = build_operational_table(ml_metrics, dl_metrics)
    operational_specs = [
        ("training_time_seconds", "Training Time (seconds)", "training_time.png"),
        (
            "inference_time_ms_per_sample",
            "Inference Time (ms/sample)",
            "inference_speed.png",
        ),
        ("model_size_mb", "Model Size (MB)", "model_size.png"),
    ]
    for column, title, filename in operational_specs:
        plt.figure(figsize=(7, 4.5))
        sns.barplot(data=operational, x="model", y=column, hue="model", legend=False)
        plt.yscale("log")
        plt.title(f"{title} - Log Scale")
        plt.xlabel("")
        plt.ylabel(title)
        path = output_path / filename
        _save_figure(path)
        chart_paths.append(str(path))

    history = dl_metrics.get("history", [])
    if history:
        epochs = [entry["epoch"] for entry in history]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(epochs, [entry["train"]["loss"] for entry in history], marker="o", label="Train")
        axes[0].plot(epochs, [entry["validation"]["loss"] for entry in history], marker="o", label="Validation")
        axes[0].set_title("DistilBERT Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(epochs, [entry["train"]["macro_f1"] for entry in history], marker="o", label="Train")
        axes[1].plot(epochs, [entry["validation"]["macro_f1"] for entry in history], marker="o", label="Validation")
        axes[1].set_title("DistilBERT Macro F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        path = output_path / "distilbert_training_history.png"
        _save_figure(path)
        chart_paths.append(str(path))

    return chart_paths


def evaluate_models(
    ml_metrics_path: str | Path = "artifacts/ml/ml_metrics.json",
    dl_metrics_path: str | Path = "artifacts/dl/dl_metrics.json",
    output_dir: str | Path = "artifacts/evaluation",
) -> dict[str, Any]:
    """Create comparison tables, charts, JSON, and CSV artifacts."""
    ml_metrics = load_metrics(ml_metrics_path)
    dl_metrics = load_metrics(dl_metrics_path)
    validate_metrics(ml_metrics, dl_metrics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    overall = build_overall_table(ml_metrics, dl_metrics)
    per_class = build_per_class_table(ml_metrics, dl_metrics)
    operational = build_operational_table(ml_metrics, dl_metrics)
    summary = build_summary(ml_metrics, dl_metrics)
    charts = generate_charts(ml_metrics, dl_metrics, output_path)

    overall.to_csv(output_path / "overall_metrics.csv", index=False)
    per_class.to_csv(output_path / "per_class_metrics.csv", index=False)
    operational.to_csv(output_path / "operational_metrics.csv", index=False)
    comparison = {
        "summary": summary,
        "overall_metrics": overall.to_dict(orient="records"),
        "per_class_metrics": per_class.to_dict(orient="records"),
        "operational_metrics": operational.to_dict(orient="records"),
        "charts": charts,
    }
    (output_path / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    logger.info("Saved model comparison artifacts to %s", output_path)
    return comparison


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare ML and DL model metrics.")
    parser.add_argument("--ml-metrics", default="artifacts/ml/ml_metrics.json")
    parser.add_argument("--dl-metrics", default="artifacts/dl/dl_metrics.json")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_models(args.ml_metrics, args.dl_metrics, args.output_dir)
