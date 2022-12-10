"""Test indices/utils.py."""
from gpt_index.indices.utils import expand_tokens_with_subtokens


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
