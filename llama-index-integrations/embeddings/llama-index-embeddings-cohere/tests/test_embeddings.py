import os

import httpx
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from llama_index.embeddings.cohere import CohereEmbedding


def test_embedding_class():
    emb = CohereEmbedding(cohere_api_key="token")
    assert isinstance(emb, BaseEmbedding)


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
def test_sync_embedding():
    emb = CohereEmbedding(
        cohere_api_key=os.environ["CO_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="clustering",
        embedding_type="float",
        httpx_client=httpx.Client(),
    )

    emb.get_query_embedding("I love Cohere!")


@pytest.mark.skipif(
    os.environ.get("CO_API_KEY") is None, reason="Cohere API key required"
)
@pytest.mark.asyncio()
async def test_async_embedding():
    emb = CohereEmbedding(
        cohere_api_key=os.environ["CO_API_KEY"],
        model_name="embed-english-v3.0",
        input_type="clustering",
        embedding_type="float",
        httpx_async_client=httpx.AsyncClient(),
    )

    await emb.aget_query_embedding("I love Cohere!")
