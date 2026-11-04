"""Regression tests for from_defaults() forwarding all options the constructor takes."""

from typing import Any

from llama_index.core.node_parser.text.code import CodeSplitter
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import Document


class FakeParser:
    """Stand-in for a tree-sitter parser so CodeSplitter can be built without the dep."""

    def parse(self, data: Any) -> Any:  # pragma: no cover - not exercised here
        raise NotImplementedError


def _my_id_func(i: int, doc: Any) -> str:
    return f"custom-{i}"


def test_code_splitter_from_defaults_accepts_include_metadata() -> None:
    splitter = CodeSplitter.from_defaults(
        language="python", include_metadata=False, parser=FakeParser()
    )
    assert splitter.include_metadata is False


def test_code_splitter_from_defaults_accepts_include_prev_next_rel() -> None:
    splitter = CodeSplitter.from_defaults(
        language="python", include_prev_next_rel=False, parser=FakeParser()
    )
    assert splitter.include_prev_next_rel is False


def test_code_splitter_from_defaults_accepts_id_func() -> None:
    splitter = CodeSplitter.from_defaults(
        language="python", id_func=_my_id_func, parser=FakeParser()
    )
    assert splitter.id_func is _my_id_func


def test_code_splitter_from_defaults_keeps_defaults() -> None:
    splitter = CodeSplitter.from_defaults(language="python", parser=FakeParser())
    assert splitter.include_metadata is True
    assert splitter.include_prev_next_rel is True


def test_sentence_splitter_from_defaults_accepts_id_func() -> None:
    """The id_func must actually be used when nodes are built."""
    splitter = SentenceSplitter.from_defaults(
        id_func=_my_id_func, chunking_tokenizer_fn=lambda text: [text]
    )
    assert splitter.id_func is _my_id_func

    nodes = splitter.get_nodes_from_documents([Document(text="hello world")])
    assert [node.id_ for node in nodes] == ["custom-0"]
