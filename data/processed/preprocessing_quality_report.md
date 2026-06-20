# Preprocessing Quality Report

This report validates the model-ready cleaned text columns used by the ML and DL pipelines.

## Dataset: data\processed\curated_sentiment_dataset.csv

| Key | Value |
| --- | --- |
| rows | 800 |
| class_counts | {'negative': 268, 'neutral': 266, 'positive': 266} |
| source_counts | {'amazon_reviews': 400, 'tweet_eval': 400} |
| errors | 0 |

## Dataset: data\processed\splits\train.csv

| Key | Value |
| --- | --- |
| rows | 560 |
| class_counts | {'negative': 188, 'neutral': 186, 'positive': 186} |
| source_counts | {'amazon_reviews': 279, 'tweet_eval': 281} |
| errors | 0 |

## Dataset: data\processed\splits\validation.csv

| Key | Value |
| --- | --- |
| rows | 120 |
| class_counts | {'negative': 40, 'neutral': 40, 'positive': 40} |
| source_counts | {'amazon_reviews': 61, 'tweet_eval': 59} |
| errors | 0 |

## Dataset: data\processed\splits\test.csv

| Key | Value |
| --- | --- |
| rows | 120 |
| class_counts | {'negative': 40, 'neutral': 40, 'positive': 40} |
| source_counts | {'amazon_reviews': 60, 'tweet_eval': 60} |
| errors | 0 |

## Split Integrity

| Key | Value |
| --- | --- |
| rows | {'train.csv': 560, 'validation.csv': 120, 'test.csv': 120} |
| errors | 0 |

## Validation Result

All preprocessing validation checks passed.
