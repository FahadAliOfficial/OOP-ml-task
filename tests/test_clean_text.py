"""Tests for text cleaning behavior."""

from src.preprocessing.clean_text import basic_text_cleanup, clean_for_dl, clean_for_ml


def test_basic_cleanup_decodes_literal_unicode_escapes() -> None:
    text = r"it\u2019s not fair\u002c dear..."

    assert basic_text_cleanup(text) == "it's not fair, dear."


def test_ml_cleaning_removes_social_platform_noise() -> None:
    text = "RT @user I LOVE this!!! #BestDay https://example.com <br>"

    assert clean_for_ml(text) == "i love this! bestday"


def test_dl_cleaning_keeps_case_but_removes_handles_and_hashtag_marker() -> None:
    text = "@user Thanks for the update #CustomerCare"

    assert clean_for_dl(text) == "Thanks for the update CustomerCare"


def test_dl_cleaning_normalizes_standalone_at_symbol() -> None:
    text = "Red Sox @ White Sox starts at 7:10 > tonight"

    assert clean_for_dl(text) == "Red Sox at White Sox starts at 7:10 tonight"


def test_ml_cleaning_removes_unsupported_symbols_but_keeps_sentiment_punctuation() -> None:
    text = "Awful 😡!!! would not buy again -- waste of $$$"

    assert clean_for_ml(text) == "awful! would not buy again -- waste of $$$"


def test_cleanup_removes_quote_and_underscore_noise() -> None:
    text = '""""\n______\n"""'

    assert basic_text_cleanup(text) == ""


def test_cleanup_repairs_common_mojibake() -> None:
    text = "d\u00c3\u00a9co looks nice"

    assert basic_text_cleanup(text) == "déco looks nice"
    assert clean_for_ml(text) == "deco looks nice"


def test_cleanup_repairs_double_encoded_mojibake() -> None:
    text = "d\u00c3\u0192\u00c2\u00a9co looks nice"

    assert clean_for_ml(text) == "deco looks nice"
