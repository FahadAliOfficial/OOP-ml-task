"""Tests for dataset curation helpers."""

import pandas as pd

from src.data.build_dataset import (
    LABEL_ORDER,
    _add_processing_columns,
    _balanced_subset,
    _remove_invalid_rows,
    map_amazon_label,
)


AMAZON_MAPPING = {
    "negative": [0, 1],
    "neutral": [2],
    "positive": [3, 4],
}


def test_map_amazon_label() -> None:
    assert map_amazon_label(0, AMAZON_MAPPING) == "negative"
    assert map_amazon_label(2, AMAZON_MAPPING) == "neutral"
    assert map_amazon_label(4, AMAZON_MAPPING) == "positive"


def test_remove_invalid_rows_removes_short_and_duplicate_text() -> None:
    df = pd.DataFrame(
        {
            "text": ["Great product", "Great product", "bad", "", '""""\n______', "x x"],
            "sentiment": [
                "positive",
                "positive",
                "negative",
                "neutral",
                "neutral",
                "neutral",
            ],
            "source": ["amazon_reviews"] * 6,
            "source_label": [4, 4, 0, 2, 2, 2],
        }
    )

    cleaned = _remove_invalid_rows(df)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["text"] == "Great product"


def test_balanced_subset_respects_target_per_class() -> None:
    rows = []
    for label in LABEL_ORDER:
        for index in range(5):
            rows.append(
                {
                    "text": f"{label} example {index}",
                    "sentiment": label,
                    "source": "tweet_eval",
                    "source_label": index,
                }
            )
    df = pd.DataFrame(rows)

    subset = _balanced_subset(df, "tweet_eval", rows_target=9, seed=42)

    assert len(subset) == 9
    assert subset["sentiment"].value_counts().to_dict() == {
        "negative": 3,
        "neutral": 3,
        "positive": 3,
    }


def test_add_processing_columns() -> None:
    df = pd.DataFrame(
        {
            "text": ["Hello @user visit https://example.com"],
            "sentiment": ["neutral"],
            "source": ["tweet_eval"],
            "source_label": [1],
        }
    )

    processed = _add_processing_columns(df)

    assert "text_clean_ml" in processed.columns
    assert "text_clean_dl" in processed.columns
    assert "text_length" in processed.columns
    assert "word_count" in processed.columns
