from unittest.mock import AsyncMock, Mock

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.oci_data_science import OCIDataScienceEmbedding
from llama_index.embeddings.oci_data_science.client import AsyncClient, Client


def test_oci_data_science_embedding_class():
    names_of_base_classes = [b.__name__ for b in OCIDataScienceEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


response_data = {
    "data": {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"},
            {"embedding": [0.4, 0.5, 0.6], "index": 1, "object": "embedding"},
        ],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "object": "list",
        "usage": {"prompt_tokens": 14, "total_tokens": 14},
    },
    "headers": {},
    "status": "200 OK",
}


@pytest.fixture()
def embeddings():
    endpoint = "https://example.com/api"
    auth = {"signer": Mock()}
    model_name = "odsc-embeddings"
    embed_batch_size = 10
    timeout = 60
    max_retries = 3
    additional_kwargs = {"some_param": "value"}
    default_headers = {"Custom-Header": "value"}
    callback_manager = CallbackManager([])

    embeddings_instance = OCIDataScienceEmbedding(
        endpoint=endpoint,
        model_name=model_name,
        auth={"signer": Mock()},
        embed_batch_size=embed_batch_size,
        timeout=timeout,
        max_retries=max_retries,
        additional_kwargs=additional_kwargs,
        default_headers=default_headers,
        callback_manager=callback_manager,
    )
    # Mock the client
    embeddings_instance._client = Mock(spec=Client)
    embeddings_instance._async_client = AsyncMock(spec=AsyncClient)
    return embeddings_instance


def test_get_query_embedding(embeddings):
    embeddings.client.embeddings.return_value = response_data["data"]

    query = "This is a test query"
    embedding_vector = embeddings.get_query_embedding(query)

    embeddings.client.embeddings.assert_called_once_with(
        input=query,
        payload=embeddings.additional_kwargs,
        headers=embeddings.default_headers,
    )

    assert embedding_vector == [0.1, 0.2, 0.3]


def test_get_text_embedding(embeddings):
    embeddings.client.embeddings.return_value = response_data["data"]

    text = "This is a test text"
    embedding_vector = embeddings.get_text_embedding(text)

    embeddings.client.embeddings.assert_called_once_with(
        input=text,
        payload=embeddings.additional_kwargs,
        headers=embeddings.default_headers,
    )

    assert embedding_vector == [0.1, 0.2, 0.3]


def test_get_text_embedding_batch(embeddings):
    embeddings.client.embeddings.return_value = response_data["data"]

    texts = ["Text one", "Text two"]
    embedding_vectors = embeddings.get_text_embedding_batch(texts)

    embeddings.client.embeddings.assert_called_once_with(
        input=texts,
        payload=embeddings.additional_kwargs,
        headers=embeddings.default_headers,
    )

    assert embedding_vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_aget_query_embedding(embeddings):
    embeddings.async_client.embeddings.return_value = response_data["data"]

    query = "Async test query"
    embedding_vector = await embeddings.aget_query_embedding(query)

    embeddings.async_client.embeddings.assert_called_once_with(
        input=query,
        payload=embeddings.additional_kwargs,
        headers=embeddings.default_headers,
    )

    assert embedding_vector == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_aget_text_embedding(embeddings):
    embeddings.async_client.embeddings.return_value = response_data["data"]

    text = "Async test text"
    embedding_vector = await embeddings.aget_text_embedding(text)

    embeddings.async_client.embeddings.assert_called_once_with(
        input=text,
        payload=embeddings.additional_kwargs,
        headers=embeddings.default_headers,
    )

    assert embedding_vector == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_aget_text_embedding_batch(embeddings):
    embeddings.async_client.embeddings.return_value = response_data["data"]

    texts = ["Async text one", "Async text two"]
    embedding_vectors = await embeddings.aget_text_embedding_batch(texts)

    embeddings.async_client.embeddings.assert_called_once_with(
        input=texts,
        payload=embeddings.additional_kwargs,
        headers=embeddings.default_headers,
    )

    assert embedding_vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
