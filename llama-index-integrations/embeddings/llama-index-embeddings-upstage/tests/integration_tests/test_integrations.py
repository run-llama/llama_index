import os

import pytest


@pytest.fixture()
def upstage_embedding():
    UpstageEmbedding = pytest.importorskip(
        "llama_index.embeddings.upstage", reason="Cannot import UpstageEmbedding"
    ).UpstageEmbedding

    if os.getenv("UPSTAGE_API_KEY") is None:
        pytest.skip("UPSTAGE_API_KEY is not set.")
    return UpstageEmbedding()


def test_upstage_embedding_query_embedding(upstage_embedding):
    query = "hello"
    embedding = upstage_embedding.get_query_embedding(query)
    assert isinstance(embedding, list)


async def test_upstage_embedding_async_query_embedding(upstage_embedding):
    query = "hello"
    embedding = await upstage_embedding.aget_query_embedding(query)
    assert isinstance(embedding, list)


def test_upstage_embedding_text_embedding(upstage_embedding):
    text = "hello"
    embedding = upstage_embedding.get_text_embedding(text)
    assert isinstance(embedding, list)


async def test_upstage_embedding_async_text_embedding(upstage_embedding):
    text = "hello"
    embedding = await upstage_embedding.aget_text_embedding(text)
    assert isinstance(embedding, list)


def test_upstage_embedding_text_embeddings(upstage_embedding):
    texts = ["hello", "world"]
    embeddings = upstage_embedding.get_text_embedding_batch(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(embedding, list) for embedding in embeddings)


def test_upstage_embedding_text_embeddings_fail_large_batch():
    UpstageEmbedding = pytest.importorskip(
        "llama_index.embeddings.upstage", reason="Cannot import UpstageEmbedding"
    ).UpstageEmbedding
    texts = ["hello"] * 2049
    with pytest.raises(ValueError):
        upstage_embedding = UpstageEmbedding(embed_batch_size=2049)
        upstage_embedding.get_text_embedding_batch(texts)


async def test_upstage_embedding_async_text_embeddings(upstage_embedding):
    texts = ["hello", "world"]
    embeddings = await upstage_embedding.aget_text_embedding_batch(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(embedding, list) for embedding in embeddings)
