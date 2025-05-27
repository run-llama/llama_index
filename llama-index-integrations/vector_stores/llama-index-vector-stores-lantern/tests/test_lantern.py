import asyncio
from typing import Any, Dict, Generator, List, Union

import pytest
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
)
from llama_index.vector_stores.lantern import LanternVectorStore

# for testing find install info here https://github.com/lanterndata/lantern#-quick-install


PARAMS: Dict[str, Union[str, int]] = {
    "host": "localhost",
    "user": "postgres",
    "password": "mark90",
    "port": 5432,
}
TEST_DB = "test_vector_db"
TEST_TABLE_NAME = "lorem_ipsum"
TEST_SCHEMA_NAME = "test"
TEST_EMBED_DIM = 3

try:
    import asyncpg  # noqa
    import pgvector  # noqa
    import psycopg2
    import sqlalchemy
    import sqlalchemy.ext.asyncio  # noqa

    # connection check
    conn__ = psycopg2.connect(**PARAMS)  # type: ignore
    conn__.close()

    postgres_not_available = False
except (ImportError, Exception):
    postgres_not_available = True


def _get_sample_vector(num: float) -> List[float]:
    """
    Get sample embedding vector of the form [num, 1, 1, ..., 1]
    where the length of the vector is TEST_EMBED_DIM.
    """
    return [num] + [1.0] * (TEST_EMBED_DIM - 1)


@pytest.fixture(scope="session")
def conn() -> Any:
    import psycopg2

    return psycopg2.connect(**PARAMS)  # type: ignore


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


@pytest.fixture()
def lantern(db: None) -> Any:
    lantern_db = LanternVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
    )

    yield lantern_db

    asyncio.run(lantern_db.close())


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_instance_creation(db: None) -> None:
    lantern_db = LanternVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
    )
    assert isinstance(lantern_db, LanternVectorStore)
    assert not hasattr(lantern_db, "_engine")
    assert lantern_db.client is None
    await lantern_db.close()


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query(
    lantern_db: LanternVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await lantern_db.async_add(node_embeddings)
    else:
        lantern_db.add(node_embeddings)
    assert isinstance(lantern_db, LanternVectorStore)
    assert hasattr(lantern_db, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(1.0), similarity_top_k=1)
    if use_async:
        res = await lantern_db.aquery(q)
    else:
        res = lantern_db.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_index_nodes(
    lantern_db: LanternVectorStore,
    index_node_embeddings: List[BaseNode],
    use_async: bool,
) -> None:
    if use_async:
        await lantern_db.async_add(index_node_embeddings)
    else:
        lantern_db.add(index_node_embeddings)
    assert isinstance(lantern_db, LanternVectorStore)
    assert hasattr(lantern_db, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(5.0), similarity_top_k=2)
    if use_async:
        res = await lantern_db.aquery(q)
    else:
        res = lantern_db.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "aaa_ref"
    assert isinstance(res.nodes[0], IndexNode)
    assert hasattr(res.nodes[0], "index_id")
    assert res.nodes[1].node_id == "bbb"
    assert isinstance(res.nodes[1], TextNode)
