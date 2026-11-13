import pytest

from llama_index.core.sparse_embeddings.mock_sparse_embedding import MockSparseEmbedding

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


def test_sparse_similarity_overlapping_and_swapped(
    mock_sparse_embedding: MockSparseEmbedding,
):
    emb_small = {0: 3.0, 1: 4.0}
    emb_large = {1: 4.0, 2: 0.0, 3: 0.0}
    assert mock_sparse_embedding.similarity(emb_small, emb_large) == pytest.approx(0.8)
    assert mock_sparse_embedding.similarity(emb_large, emb_small) == pytest.approx(0.8)
    assert mock_sparse_embedding.similarity({0: 0.0}, emb_large) == 0.0
    assert mock_sparse_embedding.similarity(emb_small, {1: 0.0, 2: 0.0}) == 0.0


@pytest.mark.asyncio
async def test_aggregate_embeddings_async_concurrent(
    mock_sparse_embedding: MockSparseEmbedding,
):
    import asyncio

    active = 0
    max_active = 0
    orig_aget = mock_sparse_embedding._aget_query_embedding

    async def tracked_aget(query: str):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        try:
            return await orig_aget(query)
        finally:
            active -= 1

    mock_sparse_embedding._aget_query_embedding = tracked_aget  # type: ignore[method-assign]
    embedding = await mock_sparse_embedding.aget_agg_embedding_from_queries(
        ["hello", "world", "foo"]
    )
    assert max_active == 3
    assert embedding == {0: 0.25 / 3, 1: 0.5 / 3, 2: 0.75 / 3}
