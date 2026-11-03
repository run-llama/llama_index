"""Test indices/utils.py."""

import pytest
from llama_index.core.indices.utils import (
    expand_tokens_with_subtokens,
    get_stopwords_set,
)
from llama_index.core.utils import globals_helper


def test_expand_tokens_with_subtokens() -> None:
    """Test expand tokens."""
    tokens = {"foo bar", "baz", "hello hello world bye"}
    keywords = expand_tokens_with_subtokens(tokens)
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


def test_expand_tokens_with_subtokens_filters_stopwords() -> None:
    """Test that stopword subtokens are dropped but the full token is kept."""
    keywords = expand_tokens_with_subtokens({"the quick fox", "of"})
    assert keywords == {"the quick fox", "quick", "fox", "of"}


def test_get_stopwords_set() -> None:
    """Test the cached stopword set mirrors the stopword list."""
    stopwords = get_stopwords_set()
    assert isinstance(stopwords, frozenset)
    assert stopwords == set(globals_helper.stopwords)
    assert "the" in stopwords
    assert "llama" not in stopwords
    assert get_stopwords_set() is stopwords


def test_get_stopwords_set_tracks_stopword_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the cached set is rebuilt when the underlying stopword list changes."""
    original = get_stopwords_set()
    monkeypatch.setattr(globals_helper, "_stopwords", ["foo", "bar"])
    assert get_stopwords_set() == frozenset({"foo", "bar"})
    assert expand_tokens_with_subtokens({"foo baz bar"}) == {
        "foo baz bar",
        "baz",
    }
    monkeypatch.undo()
    assert get_stopwords_set() == original


parse_choice_test_lines = [
    """ Doc: 2, Relevance: 8 (The document mentions taking a "tasty turn around Barcelona\'s Santa Caterina market" and listening to an episode about Barcelona.)\nDoc: 4, Relevance: 6 (The document mentions Ferramenta in Barcelona and recommends cocktails and pasta dishes that can be tried there.)""",
    "Doc: 2, Relevance: 8\nDoc: 4, Relevance: 6",
    "answer_num: 2, answer_relevance:8\nanswer_num: 4, answer_relevance:6",
]


@pytest.mark.parametrize("answer", parse_choice_test_lines)
def test_default_parse_choice_select_answer_fn(answer):
    from llama_index.core.indices.utils import default_parse_choice_select_answer_fn

    answer_nums, answer_relevances = default_parse_choice_select_answer_fn(answer, 5)
    assert answer_nums == [2, 4]
    assert answer_relevances == [8, 6]
