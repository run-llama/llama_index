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


def test_extract_keywords_drops_empty_segments() -> None:
    """A trailing or repeated comma must not produce an empty keyword (#23584)."""
    response = "KEYWORDS: foo, bar,"
    keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
    assert keywords == {"foo", "bar"}
    assert "" not in keywords

    response = "KEYWORDS: foo,, bar"
    keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
    assert keywords == {"foo", "bar"}

    # A response that is only separators yields no keywords at all.
    response = "KEYWORDS: , ,"
    keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
    assert keywords == set()


def test_extract_keywords_keeps_multiword_and_case_handling() -> None:
    """The empty-segment guard must not change non-empty keyword handling."""
    response = "KEYWORDS: Foo Bar, Baz,"
    keywords = extract_keywords_given_response(response, start_token="KEYWORDS:")
    assert keywords == {
        "foo bar",
        "foo",
        "bar",
        "baz",
    }
