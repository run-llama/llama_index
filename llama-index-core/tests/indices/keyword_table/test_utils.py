"""Test utils."""

import re
from collections import Counter

import pytest
from llama_index.core.indices.keyword_table.utils import (
    extract_keywords_given_response,
    simple_extract_keywords,
)
from llama_index.core.utils import globals_helper


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


def test_simple_extract_keywords_filters_stopwords() -> None:
    """Test stopwords are removed and tokens are lowercased."""
    text = "The Quick brown fox, and THE lazy dog; of the fox!"
    assert simple_extract_keywords(text) == {"quick", "brown", "fox", "lazy", "dog"}


def test_simple_extract_keywords_keeps_stopwords_when_disabled() -> None:
    """Test stopwords are preserved with filter_stopwords=False."""
    keywords = simple_extract_keywords("The fox and the dog", filter_stopwords=False)
    assert keywords == {"the", "fox", "and", "dog"}


def test_simple_extract_keywords_max_keywords_by_frequency() -> None:
    """Test max_keywords keeps the most frequent non-stopword tokens."""
    text = "alpha beta beta gamma gamma gamma the the the the"
    assert simple_extract_keywords(text, max_keywords=2) == {"beta", "gamma"}
    assert simple_extract_keywords(text, max_keywords=1) == {"gamma"}


def test_simple_extract_keywords_empty_and_all_stopwords() -> None:
    """Test degenerate inputs."""
    assert simple_extract_keywords("") == set()
    assert simple_extract_keywords("the and of to") == set()
    assert simple_extract_keywords("... !!! ???") == set()


@pytest.mark.parametrize("filter_stopwords", [True, False])
@pytest.mark.parametrize("max_keywords", [None, 3, 10])
def test_simple_extract_keywords_matches_reference(
    filter_stopwords: bool, max_keywords: int
) -> None:
    """Test the optimized implementation matches a straightforward reference."""
    text = (
        "The retrieval of keywords from a large chunk of text is a common task, "
        "and the keyword table index does it for every chunk: keywords keywords "
        "table table table index chunk chunk, with Unicode naïve café café too."
    )
    tokens = [t.strip().lower() for t in re.findall(r"\w+", text)]
    if filter_stopwords:
        tokens = [t for t in tokens if t not in globals_helper.stopwords]
    expected = {k for k, _ in Counter(tokens).most_common(max_keywords)}

    actual = simple_extract_keywords(
        text, max_keywords=max_keywords, filter_stopwords=filter_stopwords
    )
    assert actual == expected
