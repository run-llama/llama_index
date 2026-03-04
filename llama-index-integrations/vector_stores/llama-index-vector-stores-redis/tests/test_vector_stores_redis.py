import pytest

from llama_index.core import MockEmbedding, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import RedisVectorStoreSchema


def test_class():
    names_of_base_classes = [b.__name__ for b in RedisVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_default_usage(documents, turtle_test, redis_client):
    vector_store = RedisVectorStore(redis_client=redis_client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=MockEmbedding(embed_dim=1536),
        storage_context=storage_context,
    )

    # create retrievers
    query_engine = index.as_query_engine(llm=MockLLM(), similarity_top_k=1)
    retriever = index.as_retriever(similarity_top_k=1)

    result_nodes = retriever.retrieve(turtle_test["question"])
    query_res = query_engine.query(turtle_test["question"])

    # test they get data
    assert result_nodes[0].metadata == turtle_test["metadata"]
    assert query_res.source_nodes[0].text == turtle_test["text"]

    # test delete
    vector_store.delete([doc.doc_id for doc in documents])
    res = redis_client.ft("llama_index").search("*")
    assert len(res.docs) == 0

    # test delete index
    vector_store.delete_index()


def test_filter_only_query(documents, redis_client):
    schema = RedisVectorStoreSchema()
    schema.index.name = "llama_index_filter_only"
    schema.index.prefix = "llama_index_filter_only"
    schema.add_fields([{"name": "animal", "type": "tag"}])

    vector_store = RedisVectorStore(redis_client=redis_client, schema=schema)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=MockEmbedding(embed_dim=1536),
        storage_context=storage_context,
    )

    filters = MetadataFilters(filters=[MetadataFilter(key="animal", value="turtle")])
    query = VectorStoreQuery(filters=filters, similarity_top_k=5)
    result = vector_store.query(query)

    assert result.nodes is not None
    assert len(result.nodes) == 1
    assert result.nodes[0].metadata["animal"] == "turtle"
    assert result.similarities is None or len(result.similarities) == 1

    index = None  # release before dropping index
    vector_store.delete_index()


@pytest.mark.asyncio
async def test_async_default_usage(
    documents, turtle_test, redis_client_async, redis_client
):
    vector_store = RedisVectorStore(
        redis_client=redis_client, redis_client_async=redis_client_async
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=MockEmbedding(embed_dim=1536),
        storage_context=storage_context,
    )

    # create retrievers
    query_engine = index.as_query_engine(llm=MockLLM(), similarity_top_k=1)
    retriever = index.as_retriever(similarity_top_k=1)

    result_nodes = await retriever.aretrieve(turtle_test["question"])
    query_res = await query_engine.aquery(turtle_test["question"])

    # test they get data
    assert result_nodes[0].metadata == turtle_test["metadata"]
    assert query_res.source_nodes[0].text == turtle_test["text"]

    # test delete
    await vector_store.adelete([doc.doc_id for doc in documents])
    res = await redis_client_async.ft("llama_index").search("*")
    assert len(res.docs) == 0

    # test delete index
    await vector_store.async_delete_index()
