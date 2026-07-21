import asyncio
from typing import List, Set

import pytest

from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.sparse_embeddings.mock_sparse_embedding import MockSparseEmbedding


class FlakySparseEmbedding(BaseSparseEmbedding):
    """
    Sparse embedding model whose async text-embedding calls can be made to
    fail for specific texts, and which records which texts actually
    finished. Used to verify that a failure in one text embedding doesn't
    leave sibling in-flight embedding calls orphaned (see #22312).
    """

    _fail_texts: Set[str] = PrivateAttr(default_factory=set)
    _completed: List[str] = PrivateAttr(default_factory=list)
    _delay: float = PrivateAttr(default=0.2)

    @classmethod
    def class_name(cls) -> str:
        return "FlakySparseEmbedding"

    def _get_query_embedding(self, query: str):
        return {0: 0.0}

    async def _aget_query_embedding(self, query: str):
        return {0: 0.0}

    def _get_text_embedding(self, text: str):
        return {0: 0.0}

    async def _aget_text_embedding(self, text: str):
        if text in self._fail_texts:
            await asyncio.sleep(0.01)
            raise ValueError(f"embedding API rejected: {text}")
        await asyncio.sleep(self._delay)
        self._completed.append(text)
        return {0: 0.1}


text_embedding_map = {
    "hello": {0: 0.25},
    "world": {1: 0.5},
    "foo": {2: 0.75},
}


@pytest.fixture()
def mock_sparse_embedding():
    return MockSparseEmbedding(text_to_embedding=text_embedding_map)


def test_embedding_query(mock_sparse_embedding: MockSparseEmbedding):
    query = "hello"
    embedding = mock_sparse_embedding.get_text_embedding(query)
    assert embedding == text_embedding_map[query]


def test_embedding_text(mock_sparse_embedding: MockSparseEmbedding):
    text = "hello"
    embedding = mock_sparse_embedding.get_text_embedding(text)
    assert embedding == text_embedding_map[text]


def test_embedding_texts(mock_sparse_embedding: MockSparseEmbedding):
    texts = ["hello", "world", "foo"]
    embeddings = mock_sparse_embedding.get_text_embedding_batch(texts)
    assert embeddings == [text_embedding_map[text] for text in texts]


def test_embedding_query_not_found(mock_sparse_embedding: MockSparseEmbedding):
    query = "not_found"
    embedding = mock_sparse_embedding.get_text_embedding(query)
    assert embedding == mock_sparse_embedding.default_embedding


@pytest.mark.asyncio
async def test_embedding_query_async(mock_sparse_embedding: MockSparseEmbedding):
    query = "hello"
    embedding = await mock_sparse_embedding.aget_text_embedding(query)
    assert embedding == text_embedding_map[query]


@pytest.mark.asyncio
async def test_embedding_text_async(mock_sparse_embedding: MockSparseEmbedding):
    text = "hello"
    embedding = await mock_sparse_embedding.aget_text_embedding(text)
    assert embedding == text_embedding_map[text]


@pytest.mark.asyncio
async def test_embedding_texts_async(mock_sparse_embedding: MockSparseEmbedding):
    texts = ["hello", "world", "foo"]
    embeddings = await mock_sparse_embedding.aget_text_embedding_batch(texts)
    assert embeddings == [text_embedding_map[text] for text in texts]


def test_similarity_search(mock_sparse_embedding: MockSparseEmbedding):
    embedding1 = mock_sparse_embedding.get_text_embedding("hello")
    embedding2 = mock_sparse_embedding.get_text_embedding("world")
    similarity = mock_sparse_embedding.similarity(embedding1, embedding2)
    assert similarity == 0.0


def test_aggregate_embeddings(mock_sparse_embedding: MockSparseEmbedding):
    queries = ["hello", "world"]
    embedding = mock_sparse_embedding.get_agg_embedding_from_queries(queries)
    assert embedding == {0: 0.125, 1: 0.25}


@pytest.mark.asyncio
async def test_aggregate_embeddings_async(mock_sparse_embedding: MockSparseEmbedding):
    queries = ["hello", "world"]
    embedding = await mock_sparse_embedding.aget_agg_embedding_from_queries(queries)
    assert embedding == {0: 0.125, 1: 0.25}


@pytest.mark.asyncio
async def test_sparse_aget_text_embeddings_no_orphaned_siblings_on_failure() -> None:
    """
    Regression test for #22312 (Level 1: _aget_text_embeddings, sparse variant).
    """
    embed_model = FlakySparseEmbedding()
    embed_model._fail_texts = {"bad text"}
    batch = ["ok 1", "bad text", "ok 2"]

    with pytest.raises(ValueError, match="bad text"):
        await embed_model._aget_text_embeddings(batch)

    assert sorted(embed_model._completed) == ["ok 1", "ok 2"]


@pytest.mark.asyncio
async def test_sparse_aget_text_embedding_batch_no_orphaned_siblings_on_failure() -> (
    None
):
    """
    Regression test for #22312 (Level 2: aget_text_embedding_batch, sparse
    variant, plain asyncio.gather branch).
    """
    embed_model = FlakySparseEmbedding(embed_batch_size=2)
    embed_model._fail_texts = {"bad text"}
    texts = ["ok 1", "bad text", "ok 2", "ok 3"]

    with pytest.raises(ValueError, match="bad text"):
        await embed_model.aget_text_embedding_batch(texts, show_progress=False)

    assert sorted(embed_model._completed) == ["ok 1", "ok 2", "ok 3"]
