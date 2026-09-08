"""Test utils."""

from unittest import mock

from llama_index.core.indices.keyword_table.utils import (
    extract_keywords_given_response,
    rake_extract_keywords,
)


def test_expand_tokens_with_subtokens() -> None:
    """Test extract keywords given response."""
    response = "foo bar, baz, Hello hello wOrld bye"
    keywords = extract_keywords_given_response(response)
    assert keywords == {
        "foo bar",
        "foo",
        "bar",
        "baz",
        "hello hello world bye",
        "hello",
        "world",
        "bye",
    }


def test_extract_keywords_with_start_delimiter() -> None:
    """Test extract keywords with start delimiter."""
    response = "KEYWORDS: foo, bar, foobar"
    keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
    assert keywords == {
        "foo",
        "bar",
        "foobar",
    }

    response = "TOKENS: foo, bar, foobar"
    keywords = extract_keywords_given_response(response, start_token="TOKENS:")
    assert keywords == {
        "foo",
        "bar",
        "foobar",
    }


def test_rake_extract_keywords_ensures_nltk_data_dir_is_ready() -> None:
    """
    rake_extract_keywords reads punkt_tab via nltk.tokenize.sent_tokenize,
    which is bundled in _static/nltk_cache the same way stopwords is and is
    subject to the same nltk >= 3.10.3 hardlink rejection. It must go
    through wait_for_nltk_check() (like the stopwords/punkt_tokenizer
    properties already do) instead of relying on nltk's own default,
    unprotected resource lookup.
    """
    with mock.patch(
        "llama_index.core.indices.keyword_table.utils.globals_helper"
    ) as mock_globals_helper:
        rake_extract_keywords("Hello world, this is a test.", max_keywords=5)

    mock_globals_helper.wait_for_nltk_check.assert_called_once()
