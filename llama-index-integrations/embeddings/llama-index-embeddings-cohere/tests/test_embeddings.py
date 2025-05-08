import os

import httpx
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from llama_index.embeddings.cohere import CohereEmbedding


def test_embedding_class():
    emb = CohereEmbedding(api_key="token")
    assert isinstance(emb, BaseEmbedding)


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
def test_sync_embedding():
    emb = CohereEmbedding(
        api_key=os.environ["CO_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="clustering",
        embedding_type="float",
        httpx_client=httpx.Client(),
    )

    emb.get_query_embedding("I love Cohere!")


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
@pytest.mark.asyncio
async def test_async_embedding():
    emb = CohereEmbedding(
        api_key=os.environ["CO_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="clustering",
        embedding_type="float",
        httpx_async_client=httpx.AsyncClient(),
    )

    await emb.aget_query_embedding("I love Cohere!")


def test_cohere_embeddings_custom_endpoint_multiprocessing():
    """
    When used in multiprocessing, the CohereEmbedding instance will be serialized and deserialized. This test
    verifies, that custom base_url's are retained in the spawned processes.
    """
    # Arrange: Create a CohereEmbeddings instance with a custom base_url
    custom_base_url = "test_endpoint"
    api_key = "test_api_key"
    embeddings = CohereEmbedding(api_key=api_key, base_url=custom_base_url)

    # Act: Simulate serialization and deserialization
    serialized_data = embeddings.__getstate__()
    deserialized_embeddings = CohereEmbedding.__new__(CohereEmbedding)
    deserialized_embeddings.__setstate__(serialized_data)

    # Assert: Verify that the deserialized instance retains the correct base_url
    assert deserialized_embeddings.base_url == custom_base_url
