from typing import List, Any, Dict, Union, Generator

import pytest
import asyncio

from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.vector_stores import PGVectorStore
from llama_index.vector_stores.types import (
    NodeWithEmbedding,
    VectorStoreQuery,
    MetadataFilters,
    ExactMatchFilter,
)

# from testing find install here https://github.com/pgvector/pgvector#installation-notes


PARAMS: Dict[str, Union[str, int]] = dict(
    host="localhost", user="postgres", password="password", port=5432
)
TEST_DB = "test_vector_db"
TEST_TABLE_NAME = "lorem_ipsum"


try:
    import sqlalchemy  # noqa: F401
    import pgvector  # noqa: F401
    import psycopg2  # noqa: F401
    import asyncpg  # noqa: F401
    import sqlalchemy.ext.asyncio  # noqa: F401

    # connection check
    conn__ = psycopg2.connect(**PARAMS)  # type: ignore
    conn__.close()

    postgres_not_available = False
except (ImportError, Exception):
    postgres_not_available = True


@pytest.fixture(scope="session")
def conn() -> Any:
    import psycopg2

    conn_ = psycopg2.connect(**PARAMS)  # type: ignore
    return conn_


@pytest.fixture()
def db(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        c.execute(f"CREATE DATABASE {TEST_DB}")
        conn.commit()
    yield
    with conn.cursor() as c:
        c.execute(f"DROP DATABASE {TEST_DB}")
        conn.commit()


@pytest.fixture
def pg(db: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture(scope="session")
def node_embeddings() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0] * 1536,
            node=TextNode(
                text="lorem ipsum",
                id_="aaa",
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0] * 1536,
            node=TextNode(
                text="dolor sit amet",
                id_="bbb",
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
                extra_info={"test_key": "test_value"},
            ),
        ),
    ]

@pytest.fixture(scope="session")
def text_nodes() -> List[TextNode]:
    return [
        TextNode(
            text="The quick brown fox jumped over the lazy dog.",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
        ),
        TextNode(
            text="The fox and the hound",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ddd")},
            extra_info={"test_key": "test_value"},
        ),
    ]

@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_instance_creation(db: None) -> None:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
    )
    assert isinstance(pg, PGVectorStore)
    await pg.close()


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [(True,), (False,)])
async def test_add_to_db_and_query(
    pg: PGVectorStore, node_embeddings: List[NodeWithEmbedding], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings)
    else:
        pg.add(node_embeddings)
    assert isinstance(pg, PGVectorStore)
    q = VectorStoreQuery(query_embedding=[1] * 1536, similarity_top_k=1)
    if use_async:
        res = await pg.aquery(q)
    else:
        res = pg.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [(True,), (False,)])
async def test_add_to_db_and_query_with_metadata_filters(
    pg: PGVectorStore, node_embeddings: List[NodeWithEmbedding], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings)
    else:
        pg.add(node_embeddings)
    assert isinstance(pg, PGVectorStore)
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=[0.5] * 1536, similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg.aquery(q)
    else:
        res = pg.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [(True,), (False,)])
async def test_add_to_db_query_and_delete(
    pg: PGVectorStore, node_embeddings: List[NodeWithEmbedding], use_async: bool
) -> None:
    if use_async:
        await pg.async_add(node_embeddings)
    else:
        pg.add(node_embeddings)
    assert isinstance(pg, PGVectorStore)

    q = VectorStoreQuery(query_embedding=[0] * 1536, similarity_top_k=1)

    if use_async:
        res = await pg.aquery(q)
    else:
        res = pg.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"
    pg.delete("bbb")

    if use_async:
        res = await pg.aquery(q)
    else:
        res = pg.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"

@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [(True,), ])
async def test_hybrid_query(
        pg: PGVectorStore, node_embeddings: List[NodeWithEmbedding], text_nodes: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg.async_add_sparse_data(text_nodes)
    else:
        pg.add(node_embeddings)
    assert isinstance(pg, PGVectorStore)

    q = VectorStoreQuery(query_embedding=[0] * 1536, query_str="mr fox", similarity_top_k=2)

    if use_async:
        res = await pg.a_hybrid_query(q)
    else:
        res = pg.query(q)
    assert res.nodes
