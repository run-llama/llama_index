"""Test utils."""

from gpt_index.utils import globals_helper


def test_tokenizer() -> None:
    """Make sure tokenizer works.

    NOTE: we use a different tokenizer for python >= 3.9.

    """
    text = "hello world foo bar"
    tokenizer = globals_helper.tokenizer
    assert len(tokenizer(text)) == 4
