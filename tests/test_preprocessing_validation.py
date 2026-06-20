"""Tests for preprocessing validation."""

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.validate_dataset import (
    run_validation,
    validate_dataframe,
    validate_split_integrity,
)


def _valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["I really liked this product", "This was not good"],
            "sentiment": ["positive", "negative"],
            "source": ["unit", "unit"],
            "text_clean_ml": ["i really liked this product", "this was not good"],
            "text_clean_dl": ["I really liked this product", "This was not good"],
        }
    )


def test_validate_dataframe_accepts_clean_data() -> None:
    summary = validate_dataframe(_valid_dataframe(), name="unit")

    assert summary["errors"] == []
    assert summary["class_counts"] == {"negative": 1, "positive": 1}


def test_validate_dataframe_rejects_noise_and_bad_labels() -> None:
    df = _valid_dataframe()
    df.loc[0, "sentiment"] = "mixed"
    df.loc[1, "text_clean_ml"] = "http://example.com @user #tag"

    summary = validate_dataframe(df, name="bad")

    assert any("invalid labels" in error for error in summary["errors"])
    assert any("url" in error for error in summary["errors"])
    assert any("mention" in error for error in summary["errors"])
    assert any("hashtag_marker" in error for error in summary["errors"])


def test_validate_split_integrity_rejects_overlap(tmp_path: Path) -> None:
    rows = _valid_dataframe()
    for filename in ("train.csv", "validation.csv", "test.csv"):
        rows.to_csv(tmp_path / filename, index=False)

    summary = validate_split_integrity(tmp_path)

    assert summary["errors"]
    assert any("overlapping rows" in error for error in summary["errors"])


def test_run_validation_against_generated_dataset() -> None:
    if not Path("data/processed/curated_sentiment_dataset.csv").exists():
        pytest.skip("Generated dataset is not available.")

    result = run_validation("src/config/data.yaml")

    assert len(result["datasets"]) == 4
    assert result["splits"]["errors"] == []
