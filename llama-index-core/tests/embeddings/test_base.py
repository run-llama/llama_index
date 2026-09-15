"""Embeddings."""

import asyncio
from typing import Any, List, Set
from unittest.mock import patch
import pytest

from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    SimilarityMode,
    mean_agg,
)
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings.mock_embed_model import MockEmbedding


class FlakyEmbedding(BaseEmbedding):
    """
    Embedding model whose async text-embedding calls can be made to fail for
    specific texts, and which records which texts actually finished. Used to
    verify that a failure in one text embedding doesn't leave sibling
    in-flight embedding calls orphaned (see #22312).
    """

    _fail_texts: Set[str] = PrivateAttr(default_factory=set)
    _completed: List[str] = PrivateAttr(default_factory=list)
    _delay: float = PrivateAttr(default=0.2)

    @classmethod
    def class_name(cls) -> str:
        return "FlakyEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return [0.0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return [0.0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return [0.0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        if text in self._fail_texts:
            await asyncio.sleep(0.01)
            raise ValueError(f"embedding API rejected: {text}")
        await asyncio.sleep(self._delay)
        self._completed.append(text)
        return [0.1, 0.2, 0.3]


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "Hello world.":
        return [1, 0, 0, 0, 0]
    elif text == "This is a test.":
        return [0, 1, 0, 0, 0]
    elif text == "This is another test.":
        return [0, 0, 1, 0, 0]
    elif text == "This is a test v2.":
        return [0, 0, 0, 1, 0]
    elif text == "This is a test v3.":
        return [0, 0, 0, 0, 1]
    elif text == "This is bar test.":
        return [0, 0, 1, 0, 0]
    elif text == "Hello world backup.":
        # this is used when "Hello world." is deleted.
        return [1, 0, 0, 0, 0]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


@patch.object(MockEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding)
@patch.object(
    MockEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_get_text_embeddings(
    _mock_get_text_embeddings: Any, _mock_get_text_embedding: Any
) -> None:
    """Test get queued text embeddings."""
    embed_model = MockEmbedding(embed_dim=8)
    texts_to_embed = []
    for i in range(8):
        texts_to_embed.append("Hello world.")
    for i in range(8):
        texts_to_embed.append("This is a test.")
    for i in range(4):
        texts_to_embed.append("This is another test.")
    for i in range(4):
        texts_to_embed.append("This is a test v2.")

    result_embeddings = embed_model.get_text_embedding_batch(texts_to_embed)
    for i in range(8):
        assert result_embeddings[i] == [1, 0, 0, 0, 0]
    for i in range(8, 16):
        assert result_embeddings[i] == [0, 1, 0, 0, 0]
    for i in range(16, 20):
        assert result_embeddings[i] == [0, 0, 1, 0, 0]
    for i in range(20, 24):
        assert result_embeddings[i] == [0, 0, 0, 1, 0]


def test_embedding_similarity() -> None:
    """Test embedding similarity."""
    embed_model = MockEmbedding(embed_dim=3)
    text_embedding = [3.0, 4.0, 0.0]
    query_embedding = [0.0, 1.0, 0.0]
    cosine = embed_model.similarity(query_embedding, text_embedding)
    assert cosine == 0.8


def test_embedding_similarity_euclidean() -> None:
    embed_model = MockEmbedding(embed_dim=2)
    query_embedding = [1.0, 0.0]
    text1_embedding = [0.0, 1.0]  # further from query_embedding distance=1.414
    text2_embedding = [1.0, 1.0]  # closer to query_embedding distance=1.0
    euclidean_similarity1 = embed_model.similarity(
        query_embedding, text1_embedding, mode=SimilarityMode.EUCLIDEAN
    )
    euclidean_similarity2 = embed_model.similarity(
        query_embedding, text2_embedding, mode=SimilarityMode.EUCLIDEAN
    )
    assert euclidean_similarity1 < euclidean_similarity2


def test_mean_agg() -> None:
    """Test mean aggregation for embeddings."""
    embedding_0 = [3.0, 4.0, 0.0]
    embedding_1 = [0.0, 1.0, 0.0]
    output = mean_agg([embedding_0, embedding_1])
    assert output == [1.5, 2.5, 0.0]


def test_mean_agg_empty_list() -> None:
    """Test mean aggregation raises ValueError for empty list."""
    with pytest.raises(ValueError, match="No embeddings to aggregate"):
        mean_agg([])


@pytest.mark.asyncio
async def test_aget_text_embeddings_no_orphaned_siblings_on_failure() -> None:
    """
    Regression test for #22312 (Level 1: _aget_text_embeddings).

    A single failing text embedding must not leave sibling in-flight
    embedding calls orphaned. By the time the exception propagates, every
    other text in the same sub-batch must have already finished.
    """
    embed_model = FlakyEmbedding()
    embed_model._fail_texts = {"doc chunk 2 with policy-violating content"}
    batch = [
        "doc chunk 1",
        "doc chunk 2 with policy-violating content",
        "doc chunk 3",
    ]

    with pytest.raises(ValueError, match="doc chunk 2"):
        await embed_model._aget_text_embeddings(batch)

    assert sorted(embed_model._completed) == ["doc chunk 1", "doc chunk 3"]


@pytest.mark.asyncio
async def test_aget_text_embedding_batch_no_orphaned_siblings_on_failure() -> None:
    """
    Regression test for #22312 (Level 2: aget_text_embedding_batch, plain
    asyncio.gather branch, i.e. show_progress=False and num_workers<=1).

    A failure in one sub-batch must not leave sibling sub-batches (and the
    in-flight embedding calls within them) orphaned.
    """
    embed_model = FlakyEmbedding(embed_batch_size=2)
    embed_model._fail_texts = {"bad text"}
    texts = ["ok 1", "bad text", "ok 2", "ok 3"]

    with pytest.raises(ValueError, match="bad text"):
        await embed_model.aget_text_embedding_batch(texts, show_progress=False)

    assert sorted(embed_model._completed) == ["ok 1", "ok 2", "ok 3"]


@pytest.mark.asyncio
async def test_aget_text_embedding_batch_show_progress_no_orphaned_siblings() -> None:
    """
    Regression test for #22312 (Level 2: aget_text_embedding_batch,
    show_progress=True branch, which uses tqdm_asyncio.gather or its
    ImportError fallback).
    """
    embed_model = FlakyEmbedding(embed_batch_size=2)
    embed_model._fail_texts = {"bad text"}
    texts = ["ok 1", "bad text", "ok 2", "ok 3"]

    with pytest.raises(ValueError, match="bad text"):
        await embed_model.aget_text_embedding_batch(texts, show_progress=True)

    assert sorted(embed_model._completed) == ["ok 1", "ok 2", "ok 3"]
