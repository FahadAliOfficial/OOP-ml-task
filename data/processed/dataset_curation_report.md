# Dataset Curation Report

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

| sentiment | rows |
| --- | --- |
| negative | 268 |
| neutral | 266 |
| positive | 266 |

## Source and Class Distribution

| source | sentiment | rows |
| --- | --- | --- |
| amazon_reviews | negative | 134 |
| amazon_reviews | neutral | 133 |
| amazon_reviews | positive | 133 |
| tweet_eval | negative | 134 |
| tweet_eval | neutral | 133 |
| tweet_eval | positive | 133 |

## Text Length Statistics

- Minimum characters: 20
- Mean characters: 135.17
- Median characters: 113.00
- Maximum characters: 896

## Known Challenges

- Tweets often contain mentions, hashtags, links, abbreviations, emojis, and sarcasm-like phrasing.
- Product reviews can mix positive and negative opinions in the same text.
- Neutral examples are harder to curate because they may contain weak opinions or factual statements.
- Star ratings are not identical to sentiment, so Amazon labels require a documented mapping.

## Manual Review File

Ambiguous rows for human review are saved at `data/interim/ambiguous_rows_for_manual_review.csv`.
