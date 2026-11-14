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


def test_extract_keywords_ignores_empty_keywords() -> None:
    """A trailing or repeated comma must not yield an empty keyword."""
    assert extract_keywords_given_response(
        "KEYWORDS: foo, bar,", start_token="KEYWORDS:"
    ) == {"foo", "bar"}
    assert extract_keywords_given_response(
        "KEYWORDS: foo,, bar, ,baz", start_token="KEYWORDS:"
    ) == {"foo", "bar", "baz"}
    assert (
        extract_keywords_given_response("KEYWORDS: , ,", start_token="KEYWORDS:")
        == set()
    )
