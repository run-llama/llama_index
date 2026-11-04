"""Test utils."""

from llama_index.core.indices.keyword_table.utils import (
    extract_keywords_given_response,
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


def test_extract_keywords_skips_empty_entries() -> None:
    """Test that trailing or repeated commas do not produce empty keywords."""
    response = "KEYWORDS: foo, bar,"
    keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
    assert keywords == {"foo", "bar"}

    response = "KEYWORDS: foo,, , bar, "
    keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
    assert keywords == {"foo", "bar"}

    assert extract_keywords_given_response("KEYWORDS: ,", start_token="KEYWORDS:") == (
        set()
    )


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
