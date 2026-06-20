"""Preprocessing quality validation for curated sentiment datasets."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.clean_text import clean_for_ml
from src.utils.config import load_yaml_config


VALID_LABELS = {"negative", "neutral", "positive"}
REQUIRED_COLUMNS = {"text", "sentiment", "source", "text_clean_ml", "text_clean_dl"}
MODEL_TEXT_COLUMNS = ("text_clean_ml", "text_clean_dl")
SPLIT_FILENAMES = ("train.csv", "validation.csv", "test.csv")
NOISE_MARKERS = {
    "url": ("http://", "https://", "www."),
    "mention": ("@user",),
    "hashtag_marker": ("#",),
    "unicode_escape": ("\\u", "\\x"),
    "mojibake": ("\u00c3", "\u00c2", "\u00e2"),
    "quote": ('"',),
    "underscore_run": ("__",),
}


class PreprocessingValidationError(ValueError):
    """Raised when dataset preprocessing quality checks fail."""


def normalized_text_key(text: Any) -> str:
    """Return a stable key for split-overlap checks."""
    return clean_for_ml(text)


def validate_dataframe(
    df: pd.DataFrame,
    *,
    name: str,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Validate a curated or split dataframe and return a quality summary."""
    errors: list[str] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        errors.append(f"{name}: missing required columns: {missing_columns}")

    if expected_rows is not None and len(df) != expected_rows:
        errors.append(f"{name}: expected {expected_rows} rows, found {len(df)}")

    if "sentiment" in df.columns:
        invalid_labels = sorted(set(df["sentiment"].dropna()) - VALID_LABELS)
        if invalid_labels:
            errors.append(f"{name}: invalid labels: {invalid_labels}")

    for column in MODEL_TEXT_COLUMNS:
        if column not in df.columns:
            continue

        text = df[column].fillna("").astype(str)
        empty_count = int((text.str.strip() == "").sum())
        if empty_count:
            errors.append(f"{name}: {column} has {empty_count} empty rows")

        token_counts = text.str.findall(r"[A-Za-z0-9]{2,}").str.len()
        low_token_count = int((token_counts < 2).sum())
        if low_token_count:
            errors.append(f"{name}: {column} has {low_token_count} rows below 2 tokens")

        junk_only_count = int(text.str.fullmatch(r"[\s._'-]+", na=False).sum())
        if junk_only_count:
            errors.append(f"{name}: {column} has {junk_only_count} junk-only rows")

        for label, markers in NOISE_MARKERS.items():
            marker_count = int(text.apply(lambda value: any(m in value for m in markers)).sum())
            if marker_count:
                errors.append(f"{name}: {column} has {marker_count} rows with {label}")

    return {
        "name": name,
        "rows": len(df),
        "class_counts": (
            df["sentiment"].value_counts().sort_index().to_dict()
            if "sentiment" in df.columns
            else {}
        ),
        "source_counts": (
            df["source"].value_counts().sort_index().to_dict()
            if "source" in df.columns
            else {}
        ),
        "errors": errors,
    }


def validate_csv(path: str | Path, expected_rows: int | None = None) -> dict[str, Any]:
    """Validate a CSV file and return a summary."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    return validate_dataframe(
        pd.read_csv(csv_path),
        name=str(csv_path),
        expected_rows=expected_rows,
    )


def validate_split_integrity(split_dir: str | Path) -> dict[str, Any]:
    """Confirm train/validation/test splits exist and do not overlap."""
    split_path = Path(split_dir)
    errors: list[str] = []
    split_keys: dict[str, set[str]] = {}
    split_rows: dict[str, int] = {}

    for filename in SPLIT_FILENAMES:
        path = split_path / filename
        if not path.exists():
            errors.append(f"missing split file: {path}")
            continue

        df = pd.read_csv(path)
        split_rows[filename] = len(df)
        keys = set(df["text"].fillna("").astype(str).apply(normalized_text_key))
        if len(keys) != len(df):
            errors.append(f"{filename}: duplicate normalized text keys inside split")
        split_keys[filename] = keys

    filenames = list(split_keys)
    for index, left_name in enumerate(filenames):
        for right_name in filenames[index + 1 :]:
            overlap = split_keys[left_name] & split_keys[right_name]
            if overlap:
                errors.append(
                    f"{left_name} and {right_name}: {len(overlap)} overlapping rows"
                )

    return {"name": str(split_path), "rows": split_rows, "errors": errors}


def _format_dict_table(title: str, values: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", "", "| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return lines


def write_quality_report(
    summaries: list[dict[str, Any]],
    split_summary: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a concise preprocessing quality report."""
    lines = [
        "# Preprocessing Quality Report",
        "",
        "This report validates the model-ready cleaned text columns used by the ML and DL pipelines.",
        "",
    ]

    for summary in summaries:
        lines.extend(
            _format_dict_table(
                f"Dataset: {summary['name']}",
                {
                    "rows": summary["rows"],
                    "class_counts": summary["class_counts"],
                    "source_counts": summary["source_counts"],
                    "errors": len(summary["errors"]),
                },
            )
        )

    lines.extend(
        _format_dict_table(
            "Split Integrity",
            {"rows": split_summary["rows"], "errors": len(split_summary["errors"])},
        )
    )

    all_errors = [
        error
        for summary in summaries
        for error in summary["errors"]
    ] + split_summary["errors"]

    lines.append("## Validation Result")
    lines.append("")
    if all_errors:
        lines.extend(f"- {error}" for error in all_errors)
    else:
        lines.append("All preprocessing validation checks passed.")
    lines.append("")

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(config_path: str | Path = "src/config/data.yaml") -> dict[str, Any]:
    """Validate curated dataset and splits using the data config."""
    config = load_yaml_config(config_path)
    paths = config["paths"]

    dataset_summaries = [
        validate_csv(paths["curated_dataset"], expected_rows=800),
        validate_csv(Path(paths["split_dir"]) / "train.csv"),
        validate_csv(Path(paths["split_dir"]) / "validation.csv"),
        validate_csv(Path(paths["split_dir"]) / "test.csv"),
    ]
    split_summary = validate_split_integrity(paths["split_dir"])
    write_quality_report(
        dataset_summaries,
        split_summary,
        paths["preprocessing_report"],
    )

    errors = [
        error
        for summary in dataset_summaries
        for error in summary["errors"]
    ] + split_summary["errors"]
    if errors:
        raise PreprocessingValidationError("\n".join(errors))

    return {"datasets": dataset_summaries, "splits": split_summary}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate preprocessing quality.")
    parser.add_argument(
        "--config",
        default="src/config/data.yaml",
        help="Path to the data YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_validation(args.config)
    print("Preprocessing validation passed.")
    print(f"Validated {len(result['datasets'])} dataset files.")
