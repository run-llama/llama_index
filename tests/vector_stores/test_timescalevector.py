import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Generator, List

import pytest
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import TimescaleVectorStore
from llama_index.vector_stores.timescalevector import IndexType
from llama_index.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStoreQuery,
)

# from testing find install here https://github.com/timescale/python-vector/

TEST_SERVICE_URL = os.environ.get(
    "TEST_TIMESCALE_SERVICE_URL",
    "postgres://tsdbadmin:<password>@<id>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require",
)
TEST_TABLE_NAME = "lorem_ipsum"

try:
    from timescale_vector import client

    cli = client.Sync(TEST_SERVICE_URL, TEST_TABLE_NAME, 1536)
    with cli.connect() as test_conn:
        pass

    cli.close()

    timescale_not_available = False
except (ImportError, Exception):
    timescale_not_available = True


@pytest.fixture(scope="session")
def conn() -> Any:
    import psycopg2

    return psycopg2.connect(TEST_SERVICE_URL)  # type: ignore


@pytest.fixture()
def db(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP TABLE IF EXISTS {TEST_TABLE_NAME}")
        conn.commit()
    yield
    with conn.cursor() as c:
        # c.execute(f"DROP TABLE IF EXISTS {TEST_TABLE_NAME}")
        conn.commit()


@pytest.fixture()
def tvs(db: None) -> Any:
    tvs = TimescaleVectorStore.from_params(
        service_url=TEST_SERVICE_URL,
        table_name=TEST_TABLE_NAME,
    )

    yield tvs

    try:
        asyncio.get_event_loop().run_until_complete(tvs.close())
    except RuntimeError:
        asyncio.run(tvs.close())


@pytest.fixture()
def tvs_tp(db: None) -> Any:
    tvs = TimescaleVectorStore.from_params(
        service_url=TEST_SERVICE_URL,
        table_name=TEST_TABLE_NAME,
        time_partition_interval=timedelta(hours=1),
    )

    yield tvs

    try:
        asyncio.get_event_loop().run_until_complete(tvs.close())
    except RuntimeError:
        asyncio.run(tvs.close())


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            embedding=[1.0] * 1536,
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=[0.1] * 1536,
        ),
    ]


@pytest.mark.skipif(
    timescale_not_available, reason="timescale vector store is not available"
)
@pytest.mark.asyncio()
async def test_instance_creation(db: None) -> None:
    tvs = TimescaleVectorStore.from_params(
        service_url=TEST_SERVICE_URL,
        table_name=TEST_TABLE_NAME,
    )
    assert isinstance(tvs, TimescaleVectorStore)
    await tvs.close()


@pytest.mark.skipif(
    timescale_not_available, reason="timescale vector store is not available"
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [(True), (False)])
async def test_add_to_db_and_query(
    tvs: TimescaleVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await tvs.async_add(node_embeddings)
    else:
        tvs.add(node_embeddings)
    assert isinstance(tvs, TimescaleVectorStore)
    q = VectorStoreQuery(query_embedding=[1] * 1536, similarity_top_k=1)
    if use_async:
        res = await tvs.aquery(q)
    else:
        res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(
    timescale_not_available, reason="timescale vector store is not available"
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [(True), (False)])
async def test_add_to_db_and_query_with_metadata_filters(
    tvs: TimescaleVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await tvs.async_add(node_embeddings)
    else:
        tvs.add(node_embeddings)
    assert isinstance(tvs, TimescaleVectorStore)
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=[0.5] * 1536, similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await tvs.aquery(q)
    else:
        res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"
    assert res.ids is not None
    assert res.ids[0] == "bbb"


@pytest.mark.skipif(
    timescale_not_available, reason="timescale vector store is not available"
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [(True), (False)])
async def test_async_add_to_db_query_and_delete(
    tvs: TimescaleVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await tvs.async_add(node_embeddings)
    else:
        tvs.add(node_embeddings)
    assert isinstance(tvs, TimescaleVectorStore)

    q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=1)

    if use_async:
        res = await tvs.aquery(q)
    else:
        res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"
    tvs.delete("bbb")

    if use_async:
        res = await tvs.aquery(q)
    else:
        res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(
    timescale_not_available, reason="timescale vector store is not available"
)
def test_add_to_db_query_and_delete(
    tvs: TimescaleVectorStore, node_embeddings: List[TextNode]
) -> None:
    tvs.add(node_embeddings)
    assert isinstance(tvs, TimescaleVectorStore)

    q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=1)
    res = tvs.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"

    tvs.create_index()
    tvs.drop_index()

    tvs.create_index(IndexType.TIMESCALE_VECTOR, max_alpha=1.0, num_neighbors=50)
    tvs.drop_index()

    tvs.create_index(IndexType.PGVECTOR_IVFFLAT, num_lists=20, num_records=1000)
    tvs.drop_index()

    tvs.create_index(IndexType.PGVECTOR_HNSW, m=16, ef_construction=64)
    tvs.drop_index()


@pytest.mark.skipif(
    timescale_not_available, reason="timescale vector store is not available"
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [(True), (False)])
async def test_time_partitioning_default_uuid(
    tvs_tp: TimescaleVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await tvs_tp.async_add(node_embeddings)
    else:
        tvs_tp.add(node_embeddings)
    assert isinstance(tvs_tp, TimescaleVectorStore)

    q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=1)

    if use_async:
        res = await tvs_tp.aquery(q)
    else:
        res = tvs_tp.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(
    timescale_not_available, reason="timescale vector store is not available"
)
@pytest.mark.asyncio()
@pytest.mark.parametrize("use_async", [(True), (False)])
async def test_time_partitioning_explicit_uuid(
    tvs_tp: TimescaleVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    t0 = datetime(2018, 1, 1, 0, 0, 0)
    t = t0
    for node in node_embeddings:
        node.id_ = str(client.uuid_from_time(t))
        t = t + timedelta(days=1)
    if use_async:
        await tvs_tp.async_add(node_embeddings)
    else:
        tvs_tp.add(node_embeddings)
    assert isinstance(tvs_tp, TimescaleVectorStore)

    q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=1)

    if use_async:
        res = await tvs_tp.aquery(q)
    else:
        res = tvs_tp.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == node_embeddings[1].node_id
    assert res.ids is not None
    assert res.ids[0] != node_embeddings[1].node_id

    # make sure time filter works. This query should return only the first node
    q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=4)
    if use_async:
        res = await tvs_tp.aquery(q, end_date=t0 + timedelta(minutes=1))
    else:
        res = tvs_tp.query(q, end_date=t0 + timedelta(minutes=1))

    assert res.nodes
    assert len(res.nodes) == 1

    # here the filter should return both nodes
    q = VectorStoreQuery(query_embedding=[0.1] * 1536, similarity_top_k=4)
    if use_async:
        res = await tvs_tp.aquery(q, end_date=t0 + timedelta(days=3))
    else:
        res = tvs_tp.query(q, end_date=t0 + timedelta(days=3))

    assert res.nodes
    assert len(res.nodes) == 2
