"""Text cleaning functions used by model pipelines."""

import re
import unicodedata
from html import unescape
from typing import Any


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")
ESCAPED_UNICODE_PATTERN = re.compile(r"\\u[0-9a-fA-F]{4}")
ESCAPED_HEX_PATTERN = re.compile(r"\\x[0-9a-fA-F]{2}")
UNDERSCORE_RUN_PATTERN = re.compile(r"_{2,}")
QUOTE_RUN_PATTERN = re.compile(r'"{1,}')
RETWEET_PATTERN = re.compile(r"\brt\b:?", flags=re.IGNORECASE)
STANDALONE_AT_PATTERN = re.compile(r"\s@\s")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,]){2,}")
SPACED_PUNCTUATION_PATTERN = re.compile(r"\s+([,.!?;:])")
UNWANTED_ML_CHARS_PATTERN = re.compile(r"[^a-z0-9\s'!?.,;:%$&/+-]")
UNWANTED_DL_CHARS_PATTERN = re.compile(r"[<>]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim the result."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def decode_escaped_unicode(text: str) -> str:
    """Decode literal unicode escape sequences that appear in CSV text."""
    if not (
        ESCAPED_UNICODE_PATTERN.search(text) or ESCAPED_HEX_PATTERN.search(text)
    ):
        return text

    try:
        return text.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text decoded as Latin-1, such as d\\u00c3\\u00a9co."""
    repaired = text
    for _ in range(2):
        if not any(marker in repaired for marker in ("\u00c3", "\u00c2", "\u00e2")):
            break
        candidate = None
        for encoding in ("latin1", "cp1252"):
            try:
                candidate = repaired.encode(encoding).decode("utf-8")
                break
            except UnicodeError:
                continue
        if candidate is None:
            break
        if candidate.count("\ufffd") > repaired.count("\ufffd"):
            break
        repaired = candidate
    return repaired


def strip_accents(text: str) -> str:
    """Convert accented characters to ASCII equivalents for TF-IDF."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_punctuation(text: str) -> str:
    """Normalize common punctuation variants and repeated punctuation."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = REPEATED_PUNCTUATION_PATTERN.sub(r"\1", text)
    text = SPACED_PUNCTUATION_PATTERN.sub(r"\1", text)
    return text.strip(" \"'")


def basic_text_cleanup(text: Any) -> str:
    """Apply source-agnostic cleanup while preserving sentiment signals."""
    text = unescape("" if text is None else str(text))
    text = decode_escaped_unicode(text)
    text = repair_mojibake(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = CONTROL_CHAR_PATTERN.sub(" ", text)
    text = UNDERSCORE_RUN_PATTERN.sub(" ", text)
    text = QUOTE_RUN_PATTERN.sub(" ", text)
    text = normalize_punctuation(text)
    return normalize_whitespace(text)


def clean_for_ml(text: Any) -> str:
    """Clean text for TF-IDF based classical ML models."""
    text = basic_text_cleanup(text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = STANDALONE_AT_PATTERN.sub(" at ", text)
    text = RETWEET_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = strip_accents(text)
    text = text.lower()
    text = UNWANTED_ML_CHARS_PATTERN.sub(" ", text)
    text = normalize_punctuation(text)
    return normalize_whitespace(text)


def clean_for_dl(text: Any) -> str:
    """Apply light cleaning for transformer models."""
    text = basic_text_cleanup(text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = STANDALONE_AT_PATTERN.sub(" at ", text)
    text = RETWEET_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = UNWANTED_DL_CHARS_PATTERN.sub(" ", text)
    text = normalize_punctuation(text)
    return normalize_whitespace(text)
