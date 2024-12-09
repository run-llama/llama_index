from llama_index.core.schema import TextNode
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
import pytest
import weaviate


# @pytest.fixture(scope="module") TODO
# async def async_client():
#    client = weaviate.use_async_with_embedded()
#    asyncio.run(async_client.connect())
#    yield client
#    asyncio.run(client.close())


# TODO do we need a fixture like this?
# @pytest.fixture(scope="module")
# def async_vector_store(async_client):
#     return vector_store


@pytest.mark.asyncio()
async def test_async_basic_flow():
    async_client = weaviate.use_async_with_embedded()
    await async_client.connect()
    try:
        async_vector_store = WeaviateVectorStore(weaviate_aclient=async_client)

        nodes = [
            TextNode(text="Hello world.", embedding=[0.0, 0.0, 0.3]),
            TextNode(text="This is a test.", embedding=[0.3, 0.0, 0.0]),
        ]

        await async_vector_store.async_add(nodes)

        query = VectorStoreQuery(
            query_embedding=[0.3, 0.0, 0.0],
            similarity_top_k=2,
            query_str="world",
            mode=VectorStoreQueryMode.DEFAULT,
        )

        results = await async_vector_store.aquery(query)

        assert len(results.nodes) == 2
        assert results.nodes[0].text == "This is a test."
        assert results.similarities[0] == 1.0

        assert results.similarities[0] > results.similarities[1]

    finally:
        await async_client.close()
