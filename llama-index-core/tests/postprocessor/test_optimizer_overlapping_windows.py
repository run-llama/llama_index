"""Regression tests for overlapping context windows in SentenceEmbeddingOptimizer."""

from typing import List

import pytest

from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.postprocessor.optimizer import SentenceEmbeddingOptimizer
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


def mock_tokenizer_fn(text: str) -> List[str]:
    """Split on commas so each 'sentence' is one word."""
    return text.split(",")


def mock_get_text_embedding(text: str) -> List[float]:
    """One-hot embeddings so similarity is fully deterministic."""
    if text == "hello":
        return [1, 0, 0, 0, 0]
    elif text == "world":
        return [0, 1, 0, 0, 0]
    elif text == "foo":
        return [0, 0, 1, 0, 0]
    elif text == "bar":
        return [0, 0, 0, 1, 0]
    raise ValueError(f"Invalid text for mock embedding: {text!r}")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    return [mock_get_text_embedding(text) for text in texts]


@pytest.fixture(autouse=True)
def _patch_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MockEmbedding, "_get_text_embedding", staticmethod(mock_get_text_embedding)
    )
    monkeypatch.setattr(
        MockEmbedding,
        "_get_text_embeddings",
        staticmethod(mock_get_text_embeddings),
    )


def _optimize(
    text: str,
    query_embedding: List[float],
    context_before: int,
    context_after: int,
) -> str:
    optimizer = SentenceEmbeddingOptimizer(
        embed_model=MockEmbedding(embed_dim=5),
        tokenizer_fn=mock_tokenizer_fn,
        threshold_cutoff=0.3,
        context_before=context_before,
        context_after=context_after,
    )
    query = QueryBundle(query_str="query", embedding=query_embedding)
    node = TextNode(text=text)
    return optimizer.postprocess_nodes([NodeWithScore(node=node)], query)[
        0
    ].node.get_content()


def test_adjacent_top_sentences_are_not_duplicated() -> None:
    """
    Adjacent top sentences have overlapping context windows.

    Sentences 0 ("hello") and 1 ("world") are both selected with a context of
    one sentence on each side, so their windows overlap. The overlapping
    sentences must be emitted once, not once per selected sentence.
    """
    result = _optimize("hello,world,foo,bar", [1, 1, 0, 0, 0], 1, 1)
    assert result == "hello world foo"
    assert result.count("hello") == 1
    assert result.count("world") == 1


def test_non_adjacent_top_sentences_keep_separate_windows() -> None:
    """Selections far apart keep their own windows, each with its context."""
    result = _optimize("hello,world,foo,bar", [1, 0, 0, 1, 0], 0, 0)
    assert result == "hello bar"


def test_top_sentences_keep_original_order() -> None:
    """
    The optimized text must follow the source order, not the similarity ranking.

    "foo" (index 2) scores higher than "hello" (index 0), so it is ranked
    first, but it appears later in the source text.
    """
    result = _optimize("hello,world,foo,bar", [0.5, 0, 1, 0, 0], 0, 0)
    assert result == "hello foo"
