import os
import random
import uuid
from contextlib import contextmanager
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterator

import pytest

import asyncio

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore

from actian_vectorai import Distance, VectorAIClient, VectorParams, AsyncVectorAIClient


VECTORAI_SERVER_URL = os.getenv("VECTORAI_SERVER_URL", "localhost:6574")

EMBED_DIM = 128


def get_mock_embedding() -> list[float]:
    return [random.gauss(-1, 1) for _ in range(EMBED_DIM)]


@contextmanager
def _managed_vector_store(
    nodes_to_add: list[TextNode],
) -> Iterator[ActianVectorAIVectorStore]:
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL, collection_name=collection_name
    ) as vector_store:
        vector_store.add(nodes_to_add)
        assert vector_store.client.points.count(collection_name) == len(nodes_to_add)
        try:
            yield vector_store
        finally:
            if vector_store.client.collections.exists(collection_name):
                vector_store.client.collections.delete(collection_name)


@contextmanager
def _empty_vector_store() -> Iterator[ActianVectorAIVectorStore]:
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL, collection_name=collection_name
    ) as vector_store:
        try:
            yield vector_store
        finally:
            if vector_store.client.collections.exists(collection_name):
                vector_store.client.collections.delete(collection_name)


@asynccontextmanager
async def _amanaged_vector_store(
    nodes_to_add: list[TextNode],
) -> AsyncIterator[ActianVectorAIVectorStore]:
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    async with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL, collection_name=collection_name
    ) as vector_store:
        await vector_store.async_add(nodes_to_add)
        assert await vector_store.async_client.points.count(collection_name) == len(
            nodes_to_add
        )
        try:
            yield vector_store
        finally:
            if await vector_store.async_client.collections.exists(collection_name):
                await vector_store.async_client.collections.delete(collection_name)


@asynccontextmanager
async def _aempty_vector_store() -> AsyncIterator[ActianVectorAIVectorStore]:
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    async with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL, collection_name=collection_name
    ) as vector_store:
        try:
            yield vector_store
        finally:
            if await vector_store.async_client.collections.exists(collection_name):
                await vector_store.async_client.collections.delete(collection_name)


nodes = [
    TextNode(
        id_="c7ed938f-f8ef-4970-bf74-c240f33522f2",
        text="The 'uv' package manager provides a high-performance interface for Python dependency resolution.",
        embedding=get_mock_embedding(),
        metadata={
            "category": "tools",
            "score": 0.1,
            "tag": "alpha_token",
            "optional": [1, 2, 3],
        },
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_01")},
    ),
    TextNode(
        id_="0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
        text="PostgreSQL utilizes HugePages to reduce TLB miss overhead in high-concurrency environments.",
        embedding=get_mock_embedding(),
        metadata={
            "category": "database",
            "score": 0.5,
            "tag": "beta_token",
            "optional": [1, 2],
        },
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_02")},
    ),
    TextNode(
        id_="2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
        text="LlamaIndex nodes are the atomic units of data used for building RAG applications.",
        embedding=get_mock_embedding(),
        metadata={"category": "ai", "score": 0.9, "tag": "gamma_token"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_03")},
    ),
    TextNode(
        id_="cd617e5a-fb20-4f6e-a616-259a711268bb",
        text="NMC batteries offer higher energy density compared to LFP cells but require careful thermal management.",
        embedding=get_mock_embedding(),
        metadata={
            "category": "energy",
            "score": 0.7,
            "tag": "unique_phrase_42",
            "optional": [2, 3],
        },
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_03")},
    ),
    TextNode(
        id_="1ac88386-031d-4453-a636-b507365eb377",
        text="A 100-count cotton fabric provides a smoother texture suitable for high-end domestic bedding.",
        embedding=get_mock_embedding(),
        metadata={
            "category": "textile",
            "score": 0.3,
            "tag": "zeta_token",
            "optional": [],
        },
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc_05")},
    ),
]


def test_class_name() -> None:
    assert ActianVectorAIVectorStore.class_name() == "ActianVectorAIVectorStore"


def test_init_vector_store() -> None:
    with _empty_vector_store() as vector_store:
        assert vector_store.collection_name.startswith("test_collection_")
        info = vector_store.client.health_check()
        assert info["title"] == "Actian VectorAI DB"


def test_init_failed_client_and_async_client_in_the_same_loop() -> None:
    client = VectorAIClient(VECTORAI_SERVER_URL)
    with pytest.raises(
        ValueError,
        match=(
            "async_client cannot be the same instance as the async client used by the provided synchronous client. Please provide a different AsyncVectorAIClient instance if you wish to provide an async client."
        ),
    ):
        with ActianVectorAIVectorStore(
            client=client,
            async_client=client._async_client,
            collection_name="test_collection",
        ) as _:
            pass


def test_basic_add_and_delete() -> None:
    with _empty_vector_store() as vector_store:
        collection_name = vector_store.collection_name
        vector_store.add(nodes)

        assert vector_store.client.points.count(collection_name) == len(nodes)

        vector_store.delete("doc_03")

        assert vector_store.client.points.count(collection_name) == len(nodes) - 2

        if vector_store.client.collections.exists(collection_name):
            vector_store.client.collections.delete(collection_name)


def test_clear() -> None:
    with _managed_vector_store(nodes) as vector_store:
        assert vector_store.client.collections.exists(vector_store.collection_name)

        vector_store.clear()

        assert not vector_store.client.collections.exists(vector_store.collection_name)


def test_query_vector_search() -> None:
    with _managed_vector_store(nodes) as vector_store:
        expected_node = nodes[2]
        query = VectorStoreQuery(
            query_embedding=expected_node.embedding,
            similarity_top_k=3,
            doc_ids=["doc_03"],
            node_ids=[
                "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
                "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
            ],
        )

        result = vector_store.query(query)

        assert result.ids is not None
        assert result.nodes is not None
        assert result.similarities is not None
        assert len(result.ids) == 1
        assert len(result.nodes) == 1
        assert len(result.similarities) == 1
        assert result.ids[0] == expected_node.node_id
        assert result.nodes[0].node_id == expected_node.node_id
        assert result.nodes[0].metadata == expected_node.metadata


def test_add_with_external_client() -> None:
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        vector_store = ActianVectorAIVectorStore(
            client=client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        vector_store.add(nodes)

        assert client.points.count(vector_store.collection_name) == len(nodes)

        client.collections.delete(vector_store.collection_name)


def test_delete_with_external_client() -> None:
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        vector_store = ActianVectorAIVectorStore(
            client=client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        vector_store.add(nodes)
        vector_store.delete("doc_03")

        assert client.points.count(vector_store.collection_name) == len(nodes) - 2

        client.collections.delete(vector_store.collection_name)


def test_delete_nodes_with_external_client() -> None:
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        vector_store = ActianVectorAIVectorStore(
            client=client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        vector_store.add(nodes)
        vector_store.delete_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="ai"
                    )
                ]
            )
        )

        assert client.points.count(vector_store.collection_name) == len(nodes) - 1

        client.collections.delete(vector_store.collection_name)


def test_clear_with_external_client() -> None:
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        vector_store = ActianVectorAIVectorStore(
            client=client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        vector_store.add(nodes)
        vector_store.clear()

        assert not client.collections.exists(vector_store.collection_name)


def test_query_with_external_client() -> None:
    with VectorAIClient(VECTORAI_SERVER_URL) as client:
        vector_store = ActianVectorAIVectorStore(
            client=client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        vector_store.add(nodes)

        expected_node = nodes[2]
        query = VectorStoreQuery(
            query_embedding=expected_node.embedding,
            similarity_top_k=3,
            doc_ids=["doc_03"],
            node_ids=[
                "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
                "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
            ],
        )

        result = vector_store.query(query)

        assert result.ids is not None
        assert result.nodes is not None
        assert result.similarities is not None
        assert len(result.ids) == 1
        assert len(result.nodes) == 1
        assert len(result.similarities) == 1
        assert result.ids[0] == expected_node.node_id
        assert result.nodes[0].node_id == expected_node.node_id
        assert result.nodes[0].metadata == expected_node.metadata

        client.collections.delete(vector_store.collection_name)


@pytest.mark.parametrize(
    ("filters", "expected_remaining_count"),
    [
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="ai"
                    )
                ]
            ),
            len(nodes) - 1,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category",
                        operator=FilterOperator.IN,
                        value=["tools", "database"],
                    )
                ]
            ),
            len(nodes) - 2,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="ai"
                    ),
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="textile"
                    ),
                ],
                condition=FilterCondition.OR,
            ),
            len(nodes) - 2,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="ai"
                    ),
                    MetadataFilter(
                        key="ref_doc_id", operator=FilterOperator.EQ, value="doc_03"
                    ),
                ],
                condition=FilterCondition.AND,
            ),
            len(nodes) - 1,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="ref_doc_id", operator=FilterOperator.EQ, value="doc_03"
                    ),
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="energy"
                    ),
                ],
                condition=FilterCondition.NOT,
            ),
            2,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category",
                        operator=FilterOperator.NIN,
                        value=["ai", "energy"],
                    )
                ]
            ),
            2,
        ),
        (
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="nonexistent"
                    )
                ]
            ),
            len(nodes),
        ),
    ],
)
def test_delete_nodes_with_metadata_filters(
    filters: MetadataFilters,
    expected_remaining_count: int,
) -> None:
    with _managed_vector_store(nodes) as vector_store:
        vector_store.delete_nodes(filters=filters)

        assert (
            vector_store.client.points.count(vector_store.collection_name)
            == expected_remaining_count
        )


@pytest.mark.parametrize(
    ("operator", "key", "value", "expected_remaining_count", "raises_not_implemented"),
    [
        (FilterOperator.EQ, "category", "ai", 4, False),
        (FilterOperator.EQ, "score", 0.5, 4, False),
        (FilterOperator.GT, "score", 0.7, 4, False),
        (FilterOperator.LT, "score", 0.3, 4, False),
        (FilterOperator.NE, "score", 0.5, 1, False),
        (FilterOperator.NE, "category", "ai", 1, False),
        (FilterOperator.GTE, "score", 0.7, 3, False),
        (FilterOperator.LTE, "score", 0.3, 3, False),
        (FilterOperator.IN, "category", ["tools", "database"], 3, False),
        (FilterOperator.IN, "category", "tools,database", 3, False),
        (FilterOperator.NIN, "category", ["ai", "energy"], 2, False),
        (FilterOperator.NIN, "category", "ai,energy", 2, False),
        (FilterOperator.ANY, "tag", ["PHRASE"], 4, True),
        (FilterOperator.ALL, "tag", ["PHRASE"], 4, True),
        (FilterOperator.TEXT_MATCH, "tag", "phrase", 4, False),
        (FilterOperator.TEXT_MATCH_INSENSITIVE, "tag", "PHRASE", 4, True),
        (FilterOperator.CONTAINS, "tag", "unique_phrase_42", 4, True),
        (FilterOperator.IS_EMPTY, "optional", None, 3, False),
    ],
)
def test_delete_nodes_with_each_supported_filter_operator(
    operator: FilterOperator,
    key: str,
    value: object,
    expected_remaining_count: int,
    raises_not_implemented: bool,
) -> None:
    with _managed_vector_store(nodes) as vector_store:
        if raises_not_implemented:
            with pytest.raises(NotImplementedError):
                vector_store.delete_nodes(
                    filters=MetadataFilters(
                        filters=[
                            MetadataFilter(key=key, operator=operator, value=value)
                        ]
                    )
                )
        else:
            vector_store.delete_nodes(
                filters=MetadataFilters(
                    filters=[MetadataFilter(key=key, operator=operator, value=value)]
                )
            )

            assert (
                vector_store.client.points.count(vector_store.collection_name)
                == expected_remaining_count
            )


@pytest.mark.asyncio
async def test_async_add() -> None:
    async with _aempty_vector_store() as vector_store:
        collection_name = vector_store.collection_name
        ids = await vector_store.async_add(nodes)

        assert ids == [node.node_id for node in nodes]
        assert await vector_store.async_client.points.count(collection_name) == len(
            nodes
        )


@pytest.mark.asyncio
async def test_adelete() -> None:
    async with _aempty_vector_store() as vector_store:
        await vector_store.async_add(nodes)
        await vector_store.adelete("doc_03")

        assert (
            await vector_store.async_client.points.count(vector_store.collection_name)
            == len(nodes) - 2
        )


@pytest.mark.asyncio
async def test_adelete_nodes() -> None:
    async with _aempty_vector_store() as vector_store:
        collection_name = vector_store.collection_name
        await vector_store.async_add(nodes)
        await vector_store.adelete_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="ai"
                    )
                ]
            )
        )

        assert (
            await vector_store.async_client.points.count(vector_store.collection_name)
            == len(nodes) - 1
        )


@pytest.mark.asyncio
async def test_aclear() -> None:
    async with _amanaged_vector_store(nodes) as vector_store:
        await vector_store.aclear()
        assert not await vector_store.async_client.collections.exists(
            vector_store.collection_name
        )


@pytest.mark.asyncio
async def test_aquery_vector_search() -> None:
    async with _amanaged_vector_store(nodes) as vector_store:
        expected_node = nodes[2]
        query = VectorStoreQuery(
            query_embedding=expected_node.embedding,
            similarity_top_k=3,
            doc_ids=["doc_03"],
            node_ids=[
                "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
                "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
            ],
        )

        result = await vector_store.aquery(query)

        assert result.ids is not None
        assert result.nodes is not None
        assert result.similarities is not None
        assert len(result.ids) == 1
        assert len(result.nodes) == 1
        assert len(result.similarities) == 1
        assert result.ids[0] == expected_node.node_id
        assert result.nodes[0].node_id == expected_node.node_id
        assert result.nodes[0].metadata == expected_node.metadata


@pytest.mark.asyncio
async def test_adelete_nodes_parallel() -> None:
    async with _amanaged_vector_store(nodes) as vector_store:
        # Run multiple delete operations in parallel to test thread safety
        await asyncio.gather(
            vector_store.adelete_nodes(
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="category", operator=FilterOperator.EQ, value="ai"
                        )
                    ]
                )
            ),
            vector_store.adelete_nodes(
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="category", operator=FilterOperator.EQ, value="energy"
                        )
                    ]
                )
            ),
            vector_store.adelete_nodes(
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="category", operator=FilterOperator.EQ, value="tools"
                        )
                    ]
                )
            ),
        )

        # After deleting "ai", "energy", and "tools" categories, only "database" and "textile" should remain
        assert (
            await vector_store.async_client.points.count(vector_store.collection_name)
            == 2
        )


@pytest.mark.asyncio
async def test_async_add_with_external_client() -> None:
    async with AsyncVectorAIClient(VECTORAI_SERVER_URL) as async_client:
        vector_store = ActianVectorAIVectorStore(
            async_client=async_client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        await vector_store.async_add(nodes)
        assert await async_client.points.count(vector_store.collection_name) == len(
            nodes
        )
        await async_client.collections.delete(vector_store.collection_name)


@pytest.mark.asyncio
async def test_adelete_with_external_client() -> None:
    async with AsyncVectorAIClient(VECTORAI_SERVER_URL) as async_client:
        vector_store = ActianVectorAIVectorStore(
            async_client=async_client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        await vector_store.async_add(nodes)
        await vector_store.adelete("doc_03")

        assert (
            await async_client.points.count(vector_store.collection_name)
            == len(nodes) - 2
        )

        await async_client.collections.delete(vector_store.collection_name)


@pytest.mark.asyncio
async def test_adelete_nodes_with_external_client() -> None:
    async with AsyncVectorAIClient(VECTORAI_SERVER_URL) as async_client:
        vector_store = ActianVectorAIVectorStore(
            async_client=async_client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        await vector_store.async_add(nodes)
        await vector_store.adelete_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="ai"
                    )
                ]
            )
        )

        assert (
            await async_client.points.count(vector_store.collection_name)
            == len(nodes) - 1
        )

        await async_client.collections.delete(vector_store.collection_name)


@pytest.mark.asyncio
async def test_aclear_with_external_client() -> None:
    async with AsyncVectorAIClient(VECTORAI_SERVER_URL) as async_client:
        vector_store = ActianVectorAIVectorStore(
            async_client=async_client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        await vector_store.async_add(nodes)
        await vector_store.aclear()

        assert not await async_client.collections.exists(vector_store.collection_name)


@pytest.mark.asyncio
async def test_aquery_with_external_client() -> None:
    async with AsyncVectorAIClient(VECTORAI_SERVER_URL) as async_client:
        vector_store = ActianVectorAIVectorStore(
            async_client=async_client,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        await vector_store.async_add(nodes)

        expected_node = nodes[2]
        query = VectorStoreQuery(
            query_embedding=expected_node.embedding,
            similarity_top_k=3,
            doc_ids=["doc_03"],
            node_ids=[
                "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
                "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
            ],
        )

        result = await vector_store.aquery(query)

        assert result.ids is not None
        assert result.nodes is not None
        assert result.similarities is not None
        assert len(result.ids) == 1
        assert len(result.nodes) == 1
        assert len(result.similarities) == 1
        assert result.ids[0] == expected_node.node_id
        assert result.nodes[0].node_id == expected_node.node_id
        assert result.nodes[0].metadata == expected_node.metadata

        await async_client.collections.delete(vector_store.collection_name)


def test_query_with_connect_and_close_client() -> None:
    vector_store = ActianVectorAIVectorStore(
        url=VECTORAI_SERVER_URL,
        collection_name=f"test_collection_{uuid.uuid4().hex}",
    )

    vector_store.connect()

    vector_store.add(nodes)

    expected_node = nodes[2]
    query = VectorStoreQuery(
        query_embedding=expected_node.embedding,
        similarity_top_k=3,
        doc_ids=["doc_03"],
        node_ids=[
            "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
            "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
        ],
    )

    result = vector_store.query(query)

    assert result.ids is not None
    assert result.nodes is not None
    assert result.similarities is not None
    assert len(result.ids) == 1
    assert len(result.nodes) == 1
    assert len(result.similarities) == 1
    assert result.ids[0] == expected_node.node_id
    assert result.nodes[0].node_id == expected_node.node_id
    assert result.nodes[0].metadata == expected_node.metadata

    vector_store.clear()
    vector_store.close()


@pytest.mark.asyncio
async def test_aquery_with_connect_and_close_async_client() -> None:
    vector_store = ActianVectorAIVectorStore(
        url=VECTORAI_SERVER_URL,
        collection_name=f"test_collection_{uuid.uuid4().hex}",
    )
    await vector_store.aconnect()

    await vector_store.async_add(nodes)

    expected_node = nodes[2]
    query = VectorStoreQuery(
        query_embedding=expected_node.embedding,
        similarity_top_k=3,
        doc_ids=["doc_03"],
        node_ids=[
            "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
            "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
        ],
    )

    result = await vector_store.aquery(query)

    assert result.ids is not None
    assert result.nodes is not None
    assert result.similarities is not None
    assert len(result.ids) == 1
    assert len(result.nodes) == 1
    assert len(result.similarities) == 1
    assert result.ids[0] == expected_node.node_id
    assert result.nodes[0].node_id == expected_node.node_id
    assert result.nodes[0].metadata == expected_node.metadata

    await vector_store.aclear()
    await vector_store.aclose()


def test_query_with_existing_colletion() -> None:
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL,
        collection_name=collection_name,
        dense_vector_name="dense_vector",
        dense_vector_params=VectorParams(size=EMBED_DIM, distance=Distance.Cosine),
    ) as vector_store:
        vector_store.add(nodes)

    with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL,
        collection_name=collection_name,
        dense_vector_name="dense_vector",
    ) as vector_store:
        expected_node = nodes[2]
        query = VectorStoreQuery(
            query_embedding=expected_node.embedding,
            similarity_top_k=3,
            doc_ids=["doc_03"],
            node_ids=[
                "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
                "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
            ],
        )

        result = vector_store.query(query)

        assert result.ids is not None
        assert result.nodes is not None
        assert result.similarities is not None
        assert len(result.ids) == 1
        assert len(result.nodes) == 1
        assert len(result.similarities) == 1
        assert result.ids[0] == expected_node.node_id
        assert result.nodes[0].node_id == expected_node.node_id
        assert result.nodes[0].metadata == expected_node.metadata

        vector_store.clear()


def test_store_text_and_metadata() -> None:
    with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL,
        collection_name=f"test_collection_{uuid.uuid4().hex}",
        stores_text=True,
    ) as vector_store:
        vector_store.add(nodes)

        expected_node = nodes[2]
        query = VectorStoreQuery(
            query_embedding=expected_node.embedding,
            similarity_top_k=3,
            doc_ids=["doc_03"],
            node_ids=[
                "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
                "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
            ],
        )

        result = vector_store.query(query)

        assert result.ids is not None
        assert result.nodes is not None
        assert result.similarities is not None
        assert len(result.ids) == 1
        assert len(result.nodes) == 1
        assert len(result.similarities) == 1
        assert result.ids[0] == expected_node.node_id
        assert result.nodes[0].node_id == expected_node.node_id
        assert result.nodes[0].metadata == expected_node.metadata
        assert result.nodes[0].get_content() == expected_node.text

        vector_store.clear()


def test_clear_existing_collection() -> None:
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL, collection_name=collection_name
    ) as vector_store:
        vector_store.add(nodes)

    with ActianVectorAIVectorStore(
        VECTORAI_SERVER_URL,
        collection_name=collection_name,
        clear_existing_collection=True,
    ) as vector_store:
        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=3,
            doc_ids=["doc_03"],
            node_ids=[
                "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
                "0fc53ba5-bf74-40f2-bc26-cda9ed5b3b3e",
            ],
        )

        result = vector_store.query(query)

        assert result.ids == []
        assert result.nodes == []
        assert result.similarities == []


def test_get_nodes_all() -> None:
    with _managed_vector_store(nodes) as vector_store:
        result = vector_store.get_nodes()

        assert len(result) == len(nodes)
        returned_ids = {n.node_id for n in result}
        assert returned_ids == {n.node_id for n in nodes}


def test_get_nodes_by_node_ids() -> None:
    target_ids = [
        "c7ed938f-f8ef-4970-bf74-c240f33522f2",
        "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
    ]
    with _managed_vector_store(nodes) as vector_store:
        result = vector_store.get_nodes(node_ids=target_ids)

        assert len(result) == 2
        assert {n.node_id for n in result} == set(target_ids)


def test_get_nodes_with_filter() -> None:
    with _managed_vector_store(nodes) as vector_store:
        result = vector_store.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="ai"
                    )
                ]
            )
        )

        assert len(result) == 1
        assert result[0].node_id == "2bda1c3d-600d-46b3-9016-2709b0dcc4c7"


def test_get_nodes_empty_collection() -> None:
    with _empty_vector_store() as vector_store:
        result = vector_store.get_nodes()

        assert result == []


@pytest.mark.asyncio
async def test_aget_nodes_all() -> None:
    async with _amanaged_vector_store(nodes) as vector_store:
        result = await vector_store.aget_nodes()

        assert len(result) == len(nodes)
        returned_ids = {n.node_id for n in result}
        assert returned_ids == {n.node_id for n in nodes}


@pytest.mark.asyncio
async def test_aget_nodes_by_node_ids() -> None:
    target_ids = [
        "c7ed938f-f8ef-4970-bf74-c240f33522f2",
        "2bda1c3d-600d-46b3-9016-2709b0dcc4c7",
    ]
    async with _amanaged_vector_store(nodes) as vector_store:
        result = await vector_store.aget_nodes(node_ids=target_ids)

        assert len(result) == 2
        assert {n.node_id for n in result} == set(target_ids)


@pytest.mark.asyncio
async def test_aget_nodes_with_filter() -> None:
    async with _amanaged_vector_store(nodes) as vector_store:
        result = await vector_store.aget_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="category", operator=FilterOperator.EQ, value="ai"
                    )
                ]
            )
        )

        assert len(result) == 1
        assert result[0].node_id == "2bda1c3d-600d-46b3-9016-2709b0dcc4c7"


@pytest.mark.asyncio
async def test_aget_nodes_empty_collection() -> None:
    async with _aempty_vector_store() as vector_store:
        result = await vector_store.aget_nodes()

        assert result == []
