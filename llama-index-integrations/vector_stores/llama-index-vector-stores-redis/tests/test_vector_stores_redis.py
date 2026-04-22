import pytest

from llama_index.core import Document, MockEmbedding, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import RedisVectorStoreSchema


def _build_filterable_schema(index_name: str) -> RedisVectorStoreSchema:
    schema = RedisVectorStoreSchema()
    schema.index.name = index_name
    schema.index.prefix = index_name
    schema.add_fields([{"name": "animal", "type": "tag"}])
    return schema


def _build_store_with_documents(
    documents,
    redis_client,
    index_name: str,
    *,
    legacy_filters: bool = False,
):
    schema = _build_filterable_schema(index_name)
    vector_store = RedisVectorStore(
        redis_client=redis_client,
        schema=schema,
        legacy_filters=legacy_filters,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=MockEmbedding(embed_dim=1536),
        storage_context=storage_context,
    )
    return vector_store, index


def _build_async_store_with_documents(
    documents,
    redis_client,
    redis_client_async,
    index_name: str,
):
    schema = _build_filterable_schema(index_name)
    vector_store = RedisVectorStore(
        redis_client=redis_client,
        redis_client_async=redis_client_async,
        schema=schema,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=MockEmbedding(embed_dim=1536),
        storage_context=storage_context,
    )
    return vector_store, index


def _build_documents_with_unindexed_metadata(dummy_embedding):
    return [
        Document(
            text="merchant one turtles",
            metadata={"animal": "turtle", "merchant_id": "merchant-1"},
            doc_id="merchant-doc-1",
            embedding=dummy_embedding,
        ),
        Document(
            text="merchant two whales",
            metadata={"animal": "whale", "merchant_id": "merchant-2"},
            doc_id="merchant-doc-2",
            embedding=dummy_embedding,
        ),
    ]


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
    for doc in documents:
        vector_store.delete(doc.doc_id)
    res = redis_client.ft("llama_index").search("*")
    assert len(res.docs) == 0

    # test delete index
    vector_store.delete_index()


def test_filter_only_query(documents, redis_client):
    schema = _build_filterable_schema("llama_index_filter_only")

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


def test_filter_only_query_with_or_condition(documents, redis_client):
    schema = _build_filterable_schema("llama_index_filter_or")

    vector_store = RedisVectorStore(redis_client=redis_client, schema=schema)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=MockEmbedding(embed_dim=1536),
        storage_context=storage_context,
    )

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="animal", value="turtle"),
            MetadataFilter(key="animal", value="whale"),
        ],
        condition=FilterCondition.OR,
    )
    query = VectorStoreQuery(filters=filters, similarity_top_k=5)
    result = vector_store.query(query)

    assert result.nodes is not None
    assert len(result.nodes) == 2
    assert {node.metadata["animal"] for node in result.nodes} == {"turtle", "whale"}

    index = None
    vector_store.delete_index()


def test_get_nodes_by_filters_and_node_ids(documents, redis_client):
    vector_store, index = _build_store_with_documents(
        documents,
        redis_client,
        "redis_get",
    )
    try:
        filtered_nodes = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            )
        )
        assert len(filtered_nodes) == 1
        assert filtered_nodes[0].metadata["animal"] == "turtle"

        filtered_node_id = filtered_nodes[0].node_id
        node_by_id = vector_store.get_nodes(node_ids=[filtered_node_id])
        assert len(node_by_id) == 1
        assert node_by_id[0].node_id == filtered_node_id

        combined_nodes = vector_store.get_nodes(
            node_ids=[filtered_node_id],
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            ),
        )
        assert len(combined_nodes) == 1
        assert combined_nodes[0].node_id == filtered_node_id
    finally:
        index = None
        vector_store.delete_index()


def test_delete_nodes_by_filters(documents, redis_client):
    vector_store, index = _build_store_with_documents(
        documents,
        redis_client,
        "redis_delete",
    )
    try:
        vector_store.delete_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            )
        )
        remaining_nodes = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="whale")]
            )
        )
        assert len(remaining_nodes) == 1
        assert remaining_nodes[0].metadata["animal"] == "whale"
    finally:
        index = None
        vector_store.delete_index()


def test_delete_nodes_by_node_ids(documents, redis_client):
    vector_store, index = _build_store_with_documents(
        documents,
        redis_client,
        "redis_delete",
    )
    try:
        turtle_node_id = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            )
        )[0].node_id

        vector_store.delete_nodes(node_ids=[turtle_node_id])

        assert vector_store.get_nodes(node_ids=[turtle_node_id]) == []
        assert (
            vector_store.get_nodes(
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="animal", value="turtle")]
                )
            )
            == []
        )
        whale_nodes = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="whale")]
            )
        )
        assert len(whale_nodes) == 1
        assert whale_nodes[0].metadata["animal"] == "whale"
    finally:
        index = None
        vector_store.delete_index()


def test_unindexed_filters_fail_closed_for_metadata_operations(
    dummy_embedding, redis_client
):
    vector_store, index = _build_store_with_documents(
        _build_documents_with_unindexed_metadata(dummy_embedding),
        redis_client,
        "redis_unindexed_fail_closed",
    )
    try:
        merchant_one_filters = MetadataFilters(
            filters=[MetadataFilter(key="merchant_id", value="merchant-1")]
        )
        merchant_two_filters = MetadataFilters(
            filters=[MetadataFilter(key="merchant_id", value="merchant-2")]
        )

        result = vector_store.query(
            VectorStoreQuery(filters=merchant_one_filters, similarity_top_k=5)
        )
        assert result.ids == []
        assert result.nodes == []

        merchant_one_nodes = vector_store.get_nodes(filters=merchant_one_filters)
        assert merchant_one_nodes == []

        vector_store.delete_nodes(filters=merchant_one_filters)

        merchant_two_nodes = vector_store.get_nodes(filters=merchant_two_filters)
        assert merchant_two_nodes == []

        whale_nodes = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="whale")]
            )
        )
        assert len(whale_nodes) == 1
        assert whale_nodes[0].metadata["animal"] == "whale"
    finally:
        index = None
        vector_store.delete_index()


def test_unindexed_embedding_query_returns_no_matches(dummy_embedding, redis_client):
    vector_store, index = _build_store_with_documents(
        _build_documents_with_unindexed_metadata(dummy_embedding),
        redis_client,
        "redis_unindexed_embedding",
    )
    try:
        result = vector_store.query(
            VectorStoreQuery(
                query_embedding=dummy_embedding,
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="merchant_id", value="merchant-1")]
                ),
                similarity_top_k=1,
            )
        )
        assert result.ids == []
        assert result.nodes == []
    finally:
        index = None
        vector_store.delete_index()


def test_legacy_filters_get_nodes_by_filters_and_node_ids(documents, redis_client):
    vector_store, index = _build_store_with_documents(
        documents,
        redis_client,
        "redis_legacy_get",
        legacy_filters=True,
    )
    try:
        filtered_nodes = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            )
        )
        assert len(filtered_nodes) == 1
        assert filtered_nodes[0].metadata["animal"] == "turtle"

        filtered_node_id = filtered_nodes[0].node_id
        combined_nodes = vector_store.get_nodes(
            node_ids=[filtered_node_id],
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            ),
        )
        assert len(combined_nodes) == 1
        assert combined_nodes[0].node_id == filtered_node_id
    finally:
        index = None
        vector_store.delete_index()


def test_legacy_filters_delete_nodes(documents, redis_client):
    vector_store, index = _build_store_with_documents(
        documents,
        redis_client,
        "redis_legacy_delete",
        legacy_filters=True,
    )
    try:
        vector_store.delete_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            )
        )
        remaining_nodes = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="whale")]
            )
        )
        assert len(remaining_nodes) == 1
        assert remaining_nodes[0].metadata["animal"] == "whale"
    finally:
        index = None
        vector_store.delete_index()


def test_legacy_filters_delete_nodes_by_node_ids(documents, redis_client):
    vector_store, index = _build_store_with_documents(
        documents,
        redis_client,
        "redis_legacy_delete_by_id",
        legacy_filters=True,
    )
    try:
        turtle_node_id = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            )
        )[0].node_id

        vector_store.delete_nodes(node_ids=[turtle_node_id])

        assert vector_store.get_nodes(node_ids=[turtle_node_id]) == []
        assert (
            vector_store.get_nodes(
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="animal", value="turtle")]
                )
            )
            == []
        )
        whale_nodes = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="whale")]
            )
        )
        assert len(whale_nodes) == 1
        assert whale_nodes[0].metadata["animal"] == "whale"
    finally:
        index = None
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
    for doc in documents:
        await vector_store.adelete(doc.doc_id)
    res = await redis_client_async.ft("llama_index").search("*")
    assert len(res.docs) == 0

    # test delete index
    await vector_store.async_delete_index()


@pytest.mark.asyncio
async def test_async_filter_only_query_with_or_condition(
    documents, redis_client_async, redis_client
):
    vector_store, index = _build_async_store_with_documents(
        documents,
        redis_client,
        redis_client_async,
        "redis_async_filter_or",
    )
    try:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="animal", value="turtle"),
                MetadataFilter(key="animal", value="whale"),
            ],
            condition=FilterCondition.OR,
        )
        result = await vector_store.aquery(
            VectorStoreQuery(filters=filters, similarity_top_k=5)
        )

        assert result.nodes is not None
        assert len(result.nodes) == 2
        assert {node.metadata["animal"] for node in result.nodes} == {
            "turtle",
            "whale",
        }
    finally:
        index = None
        await vector_store.async_delete_index()


@pytest.mark.asyncio
async def test_async_get_nodes_by_filters_and_node_ids(
    documents, redis_client_async, redis_client
):
    vector_store, index = _build_async_store_with_documents(
        documents,
        redis_client,
        redis_client_async,
        "redis_async_get",
    )
    try:
        filtered_nodes = await vector_store.aget_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            )
        )
        assert len(filtered_nodes) == 1
        assert filtered_nodes[0].metadata["animal"] == "turtle"

        filtered_node_id = filtered_nodes[0].node_id
        node_by_id = await vector_store.aget_nodes(node_ids=[filtered_node_id])
        assert len(node_by_id) == 1
        assert node_by_id[0].node_id == filtered_node_id

        combined_nodes = await vector_store.aget_nodes(
            node_ids=[filtered_node_id],
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            ),
        )
        assert len(combined_nodes) == 1
        assert combined_nodes[0].node_id == filtered_node_id
    finally:
        index = None
        await vector_store.async_delete_index()


@pytest.mark.asyncio
async def test_async_delete_nodes_by_filters(
    documents, redis_client_async, redis_client
):
    vector_store, index = _build_async_store_with_documents(
        documents,
        redis_client,
        redis_client_async,
        "redis_async_delete",
    )
    try:
        await vector_store.adelete_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="turtle")]
            )
        )
        remaining_nodes = await vector_store.aget_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="whale")]
            )
        )
        assert len(remaining_nodes) == 1
        assert remaining_nodes[0].metadata["animal"] == "whale"
    finally:
        index = None
        await vector_store.async_delete_index()


@pytest.mark.asyncio
async def test_async_delete_nodes_by_node_ids(
    documents, redis_client_async, redis_client
):
    vector_store, index = _build_async_store_with_documents(
        documents,
        redis_client,
        redis_client_async,
        "redis_async_delete_by_id",
    )
    try:
        turtle_node_id = (
            await vector_store.aget_nodes(
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="animal", value="turtle")]
                )
            )
        )[0].node_id

        await vector_store.adelete_nodes(node_ids=[turtle_node_id])

        assert await vector_store.aget_nodes(node_ids=[turtle_node_id]) == []
        assert (
            await vector_store.aget_nodes(
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="animal", value="turtle")]
                )
            )
            == []
        )
        whale_nodes = await vector_store.aget_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="whale")]
            )
        )
        assert len(whale_nodes) == 1
        assert whale_nodes[0].metadata["animal"] == "whale"
    finally:
        index = None
        await vector_store.async_delete_index()


@pytest.mark.asyncio
async def test_async_unindexed_filters_fail_closed_for_metadata_operations(
    dummy_embedding, redis_client_async, redis_client
):
    vector_store, index = _build_async_store_with_documents(
        _build_documents_with_unindexed_metadata(dummy_embedding),
        redis_client,
        redis_client_async,
        "redis_async_unindexed_fail_closed",
    )
    try:
        merchant_one_filters = MetadataFilters(
            filters=[MetadataFilter(key="merchant_id", value="merchant-1")]
        )
        merchant_two_filters = MetadataFilters(
            filters=[MetadataFilter(key="merchant_id", value="merchant-2")]
        )

        result = await vector_store.aquery(
            VectorStoreQuery(filters=merchant_one_filters, similarity_top_k=5)
        )
        assert result.ids == []
        assert result.nodes == []

        merchant_one_nodes = await vector_store.aget_nodes(filters=merchant_one_filters)
        assert merchant_one_nodes == []

        await vector_store.adelete_nodes(filters=merchant_one_filters)

        merchant_two_nodes = await vector_store.aget_nodes(filters=merchant_two_filters)
        assert merchant_two_nodes == []

        whale_nodes = await vector_store.aget_nodes(
            filters=MetadataFilters(
                filters=[MetadataFilter(key="animal", value="whale")]
            )
        )
        assert len(whale_nodes) == 1
        assert whale_nodes[0].metadata["animal"] == "whale"
    finally:
        index = None
        await vector_store.async_delete_index()


@pytest.mark.asyncio
async def test_async_unindexed_embedding_query_returns_no_matches(
    dummy_embedding, redis_client_async, redis_client
):
    vector_store, index = _build_async_store_with_documents(
        _build_documents_with_unindexed_metadata(dummy_embedding),
        redis_client,
        redis_client_async,
        "redis_async_unindexed_embedding",
    )
    try:
        result = await vector_store.aquery(
            VectorStoreQuery(
                query_embedding=dummy_embedding,
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="merchant_id", value="merchant-1")]
                ),
                similarity_top_k=1,
            )
        )
        assert result.ids == []
        assert result.nodes == []
    finally:
        index = None
        await vector_store.async_delete_index()
