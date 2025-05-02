import os

import pytest
from pytest_mock import MockerFixture

MOCK_EMBEDDING_DATA = [1.0, 2.0, 3.0]
UPSTAGE_TEST_API_KEY = "UPSTAGE_TEST_API_KEY"


@pytest.fixture()
def setup_environment(monkeypatch):
    monkeypatch.setenv("UPSTAGE_API_KEY", UPSTAGE_TEST_API_KEY)


@pytest.fixture()
def upstage_embedding():
    UpstageEmbedding = pytest.importorskip(
        "llama_index.embeddings.upstage", reason="Cannot import UpstageEmbedding"
    ).UpstageEmbedding

    if os.getenv("UPSTAGE_API_KEY") is None:
        pytest.skip("UPSTAGE_API_KEY is not set.")
    return UpstageEmbedding()


def test_upstage_embedding_query_embedding(
    mocker: MockerFixture, setup_environment, upstage_embedding
):
    query = "hello"
    mock_openai_client = mocker.patch(
        "llama_index.embeddings.upstage.base.UpstageEmbedding._get_query_embedding"
    )
    mock_openai_client.return_value = MOCK_EMBEDDING_DATA

    embedding = upstage_embedding.get_query_embedding(query)
    assert isinstance(embedding, list)


async def test_upstage_embedding_async_query_embedding(
    mocker: MockerFixture, setup_environment, upstage_embedding
):
    query = "hello"
    mock_openai_client = mocker.patch(
        "llama_index.embeddings.upstage.base.UpstageEmbedding._aget_query_embedding"
    )
    mock_openai_client.return_value = MOCK_EMBEDDING_DATA

    embedding = await upstage_embedding.aget_query_embedding(query)
    assert isinstance(embedding, list)


def test_upstage_embedding_text_embedding(
    mocker: MockerFixture, setup_environment, upstage_embedding
):
    text = "hello"
    mock_openai_client = mocker.patch(
        "llama_index.embeddings.upstage.base.UpstageEmbedding._get_text_embedding"
    )
    mock_openai_client.return_value = MOCK_EMBEDDING_DATA

    embedding = upstage_embedding.get_text_embedding(text)
    assert isinstance(embedding, list)


async def test_upstage_embedding_async_text_embedding(
    mocker: MockerFixture, setup_environment, upstage_embedding
):
    text = "hello"
    mock_openai_client = mocker.patch(
        "llama_index.embeddings.upstage.base.UpstageEmbedding._aget_text_embedding"
    )
    mock_openai_client.return_value = MOCK_EMBEDDING_DATA

    embedding = await upstage_embedding.aget_text_embedding(text)
    assert isinstance(embedding, list)


def test_upstage_embedding_text_embeddings(
    mocker: MockerFixture, setup_environment, upstage_embedding
):
    texts = ["hello", "world"]
    mock_openai_client = mocker.patch(
        "llama_index.embeddings.upstage.base.UpstageEmbedding._get_text_embeddings"
    )
    mock_openai_client.return_value = [MOCK_EMBEDDING_DATA] * len(texts)

    embeddings = upstage_embedding.get_text_embedding_batch(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(embedding, list) for embedding in embeddings)


def test_upstage_embedding_text_embeddings_fail_large_batch(
    mocker: MockerFixture, setup_environment
):
    large_batch_size = 2049
    UpstageEmbedding = pytest.importorskip(
        "llama_index.embeddings.upstage", reason="Cannot import UpstageEmbedding"
    ).UpstageEmbedding

    mock_openai_client = mocker.patch(
        "llama_index.embeddings.upstage.base.UpstageEmbedding._get_text_embeddings"
    )
    mock_openai_client.return_value = [MOCK_EMBEDDING_DATA] * large_batch_size

    texts = ["hello"] * large_batch_size
    with pytest.raises(ValueError):
        upstage_embedding = UpstageEmbedding(embed_batch_size=2049)
        upstage_embedding.get_text_embedding_batch(texts)


async def test_upstage_embedding_async_text_embeddings(
    mocker: MockerFixture, setup_environment, upstage_embedding
):
    texts = ["hello", "world"]
    mock_openai_client = mocker.patch(
        "llama_index.embeddings.upstage.base.UpstageEmbedding._aget_text_embeddings"
    )
    mock_openai_client.return_value = [MOCK_EMBEDDING_DATA] * len(texts)

    embeddings = await upstage_embedding.aget_text_embedding_batch(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(embedding, list) for embedding in embeddings)
