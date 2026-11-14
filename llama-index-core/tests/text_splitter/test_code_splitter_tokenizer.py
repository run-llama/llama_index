from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llama_index.core.node_parser.text import code
from llama_index.core.node_parser.text.code import CodeSplitter


class Parser:
    def parse(self, text: bytes) -> SimpleNamespace:
        leaf = SimpleNamespace(
            type="expression_statement", start_byte=0, end_byte=len(text), children=[]
        )
        return SimpleNamespace(root_node=SimpleNamespace(children=[leaf]))


def test_char_splitter_does_not_initialize_default_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_factory = Mock(side_effect=RuntimeError("Tokenizer unavailable offline"))
    monkeypatch.setattr(code, "get_tokenizer", tokenizer_factory)

    splitter = CodeSplitter(language="python", parser=Parser(), max_chars=4)

    assert splitter.split_text("hello") == ["hell", "o"]
    tokenizer_factory.assert_not_called()


def test_switching_to_token_mode_initializes_tokenizer_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_factory = Mock(return_value=list)
    monkeypatch.setattr(code, "get_tokenizer", tokenizer_factory)
    splitter = CodeSplitter(language="python", parser=Parser(), max_tokens=2)
    splitter.count_mode = "token"

    assert splitter.split_text("hello") == ["he", "ll", "o"]
    assert splitter.split_text("world") == ["wo", "rl", "d"]
    tokenizer_factory.assert_called_once_with()


def test_token_splitter_keeps_explicit_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_factory = Mock(side_effect=RuntimeError("Default tokenizer unavailable"))
    monkeypatch.setattr(code, "get_tokenizer", tokenizer_factory)
    splitter = CodeSplitter(
        language="python",
        parser=Parser(),
        count_mode="token",
        max_tokens=2,
        tokenizer=list,
    )

    assert splitter.split_text("hello") == ["he", "ll", "o"]
    tokenizer_factory.assert_not_called()
