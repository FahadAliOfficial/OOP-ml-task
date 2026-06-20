"""Dataset creation and curation entrypoint."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.clean_text import basic_text_cleanup, clean_for_dl, clean_for_ml
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


logger = get_logger(__name__)

TWEET_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_ORDER = ["negative", "neutral", "positive"]


def _load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Import Hugging Face datasets lazily so scaffold imports stay lightweight."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for Phase 2. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return load_dataset(*args, **kwargs)


def map_amazon_label(label: int, mapping: dict[str, list[int]]) -> str:
    """Map SetFit Amazon 0-4 review labels into 3 sentiment classes."""
    label = int(label)
    for sentiment, source_labels in mapping.items():
        if label in source_labels:
            return sentiment
    raise ValueError(f"Unsupported Amazon label: {label}")


def _ensure_directories(config: dict[str, Any]) -> None:
    for key in ("raw_dir", "interim_dir", "processed_dir", "split_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def _text_length_stats(series: pd.Series) -> dict[str, float]:
    lengths = series.fillna("").astype(str).str.len()
    return {
        "min": float(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "max": float(lengths.max()),
    }


def _prepare_tweet_eval(config: dict[str, Any]) -> pd.DataFrame:
    source_config = config["sources"]["tweet_eval"]
    dataset = _load_dataset(
        source_config["dataset_name"],
        source_config["subset"],
        split="train",
    )
    df = dataset.to_pandas()[["text", "label"]]
    df["source"] = "tweet_eval"
    df["source_label"] = df["label"].astype(int)
    df["sentiment"] = df["source_label"].map(TWEET_LABELS)
    return df[["text", "sentiment", "source", "source_label"]]


def _prepare_amazon_reviews(config: dict[str, Any]) -> pd.DataFrame:
    source_config = config["sources"]["amazon_reviews"]
    dataset = _load_dataset(source_config["dataset_name"], split="train")
    df = dataset.to_pandas()
    text_column = "text" if "text" in df.columns else "review_body"
    label_column = "label" if "label" in df.columns else "stars"

    prepared = pd.DataFrame(
        {
            "text": df[text_column],
            "source_label": df[label_column].astype(int),
        }
    )
    prepared["source"] = "amazon_reviews"
    prepared["sentiment"] = prepared["source_label"].apply(
        lambda label: map_amazon_label(label, config["labels"]["amazon_star_mapping"])
    )
    return prepared[["text", "sentiment", "source", "source_label"]]


def _load_or_prepare_sources(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized raw CSVs when present, otherwise download from sources."""
    raw_dir = Path(config["paths"]["raw_dir"])
    tweet_raw_path = raw_dir / "tweet_eval_sentiment_raw.csv"
    amazon_raw_path = raw_dir / "amazon_reviews_raw.csv"

    if tweet_raw_path.exists() and amazon_raw_path.exists():
        logger.info("Loading normalized raw datasets from local CSV cache.")
        return pd.read_csv(tweet_raw_path), pd.read_csv(amazon_raw_path)

    logger.info("Loading TweetEval sentiment dataset.")
    tweet_df = _remove_invalid_rows(_prepare_tweet_eval(config))
    logger.info("Loading Amazon reviews dataset.")
    amazon_df = _remove_invalid_rows(_prepare_amazon_reviews(config))

    tweet_df.to_csv(tweet_raw_path, index=False)
    amazon_df.to_csv(amazon_raw_path, index=False)
    return tweet_df, amazon_df


def _remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    curated = df.copy()
    curated["text"] = curated["text"].apply(basic_text_cleanup)
    curated = curated[curated["text"].str.len() >= 10]
    curated = curated[curated["sentiment"].isin(LABEL_ORDER)]
    quality_text = curated["text"].apply(clean_for_ml)
    meaningful_tokens = quality_text.str.findall(r"[a-z0-9]{2,}").str.len()
    curated = curated[(quality_text.str.len() >= 10) & (meaningful_tokens >= 2)]
    curated["dedupe_key"] = curated["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    curated = curated.drop_duplicates(subset=["dedupe_key"])
    return curated.drop(columns=["dedupe_key"]).reset_index(drop=True)


def _balanced_subset(
    df: pd.DataFrame,
    source_name: str,
    rows_target: int,
    seed: int,
) -> pd.DataFrame:
    per_class = rows_target // len(LABEL_ORDER)
    remainder = rows_target % len(LABEL_ORDER)
    pieces = []

    for index, label in enumerate(LABEL_ORDER):
        target = per_class + (1 if index < remainder else 0)
        class_rows = df[(df["source"] == source_name) & (df["sentiment"] == label)]
        if class_rows.empty:
            raise ValueError(f"No rows found for {source_name}/{label}")
        sample_size = min(target, len(class_rows))
        pieces.append(class_rows.sample(n=sample_size, random_state=seed + index))

    return pd.concat(pieces, ignore_index=True)


def _add_processing_columns(df: pd.DataFrame) -> pd.DataFrame:
    curated = df.copy()
    curated["text_clean_ml"] = curated["text"].apply(clean_for_ml)
    curated["text_clean_dl"] = curated["text"].apply(clean_for_dl)
    curated["text_length"] = curated["text"].str.len()
    curated["word_count"] = curated["text"].str.split().str.len()
    return curated


def _write_splits(df: pd.DataFrame, config: dict[str, Any]) -> None:
    split_config = config["split"]
    split_dir = Path(config["paths"]["split_dir"])
    seed = int(config["project"]["seed"])
    stratify_cols = df["sentiment"] if split_config.get("stratify", True) else None

    train_df, temp_df = train_test_split(
        df,
        train_size=float(split_config["train_size"]),
        random_state=seed,
        stratify=stratify_cols,
    )

    validation_ratio = float(split_config["validation_size"]) / (
        float(split_config["validation_size"]) + float(split_config["test_size"])
    )
    temp_stratify = temp_df["sentiment"] if split_config.get("stratify", True) else None
    validation_df, test_df = train_test_split(
        temp_df,
        train_size=validation_ratio,
        random_state=seed,
        stratify=temp_stratify,
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    validation_df.to_csv(split_dir / "validation.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)


def _write_ambiguous_rows(df: pd.DataFrame, path: str | Path) -> None:
    """Save rows most useful for a short manual review pass."""
    ambiguous = df[
        (df["sentiment"] == "neutral")
        | (df["word_count"] <= 5)
        | (df["text"].str.contains(r"\bbut\b|\bhowever\b|not ", case=False, regex=True))
    ].copy()
    ambiguous = ambiguous.sort_values(["sentiment", "source", "word_count"]).head(80)
    ambiguous["manual_review_note"] = ""
    ambiguous.to_csv(path, index=False)


def _format_counts_table(counts: pd.DataFrame) -> str:
    columns = list(counts.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in counts.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def _write_curation_report(df: pd.DataFrame, config: dict[str, Any]) -> None:
    source_counts = df.groupby(["source", "sentiment"]).size().reset_index(name="rows")
    class_counts = df.groupby("sentiment").size().reset_index(name="rows")
    length_stats = _text_length_stats(df["text"])

    report = f"""# Dataset Curation Report

## Dataset Choices

This project combines two open-source English sentiment datasets:

- `cardiffnlp/tweet_eval`, sentiment subset: short and noisy social media text with mentions, hashtags, informal wording, and a direct negative/neutral/positive label mapping.
- `SetFit/amazon_reviews_multi_en`: product reviews with longer opinion-rich text. Its 0-4 labels are mapped into three sentiment classes: 0-1 negative, 2 neutral, and 3-4 positive.

The combination intentionally tests whether the final models can generalize across both social-media sentiment and product-review sentiment.

## Curation Rules

- Selected a balanced subset from each source using a fixed random seed.
- Removed empty text, very short text, duplicate text, and unsupported labels.
- Preserved both ML-cleaned and transformer-cleaned text columns for later phases.
- Saved ambiguous examples separately for manual review.

## Final Class Distribution

{_format_counts_table(class_counts)}

## Source and Class Distribution

{_format_counts_table(source_counts)}

## Text Length Statistics

- Minimum characters: {length_stats["min"]:.0f}
- Mean characters: {length_stats["mean"]:.2f}
- Median characters: {length_stats["median"]:.2f}
- Maximum characters: {length_stats["max"]:.0f}

## Known Challenges

- Tweets often contain mentions, hashtags, links, abbreviations, emojis, and sarcasm-like phrasing.
- Product reviews can mix positive and negative opinions in the same text.
- Neutral examples are harder to curate because they may contain weak opinions or factual statements.
- Star ratings are not identical to sentiment, so Amazon labels require a documented mapping.

## Manual Review File

Ambiguous rows for human review are saved at `{config["paths"]["ambiguous_rows"]}`.
"""
    Path(config["paths"]["curation_report"]).write_text(report, encoding="utf-8")


def build_dataset(config_path: str | Path = "src/config/data.yaml") -> pd.DataFrame:
    """Build the curated sentiment dataset."""
    config = load_yaml_config(config_path)
    _ensure_directories(config)
    seed = int(config["project"]["seed"])

    tweet_df, amazon_df = _load_or_prepare_sources(config)
    tweet_df = _remove_invalid_rows(tweet_df)
    amazon_df = _remove_invalid_rows(amazon_df)

    tweet_subset = _balanced_subset(
        tweet_df,
        "tweet_eval",
        int(config["sources"]["tweet_eval"]["rows_target"]),
        seed,
    )
    amazon_subset = _balanced_subset(
        amazon_df,
        "amazon_reviews",
        int(config["sources"]["amazon_reviews"]["rows_target"]),
        seed,
    )

    curated = pd.concat([tweet_subset, amazon_subset], ignore_index=True)
    curated = _add_processing_columns(_remove_invalid_rows(curated))
    curated = curated.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    interim_path = Path(config["paths"]["interim_dir"]) / "combined_curated_subset.csv"
    curated.to_csv(interim_path, index=False)
    curated.to_csv(config["paths"]["curated_dataset"], index=False)
    _write_splits(curated, config)
    _write_ambiguous_rows(curated, config["paths"]["ambiguous_rows"])
    _write_curation_report(curated, config)

    logger.info("Curated dataset saved to %s", config["paths"]["curated_dataset"])
    logger.info("Curation report saved to %s", config["paths"]["curation_report"])
    return curated


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the curated sentiment dataset.")
    parser.add_argument(
        "--config",
        default="src/config/data.yaml",
        help="Path to the data YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.config)
