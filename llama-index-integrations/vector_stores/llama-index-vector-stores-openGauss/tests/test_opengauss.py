import pytest
from typing import Any, Dict, Generator, List, Union
import asyncio
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.openGauss import OpenGaussStore

PARAMS: Dict[str, Union[str, int]] = {
    "host": "localhost",
    "user": "postgres",
    "password": "mark90",
    "port": 5432,
}
TEST_DB = "test_vector_db"
TEST_TABLE_NAME = "lorem_ipsum"
TEST_SCHEMA_NAME = "test"
TEST_EMBED_DIM = 2

try:
    import asyncpg  # noqa
    import psycopg2  # noqa

    postgres_not_available = False
except ImportError:
    postgres_not_available = True


def _get_sample_vector(num: float) -> List[float]:
    return [num] + [1.0] * (TEST_EMBED_DIM - 1)


@pytest.fixture(scope="session")
def conn() -> Any:
    import psycopg2

    if postgres_not_available:
        pytest.skip("psycopg2 or asyncpg not installed")

    try:
        return psycopg2.connect(**PARAMS)
    except Exception as e:
        pytest.skip(f"Database connection failed: {e!s}")


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
def pg_hybrid(db: None) -> Any:
    store = OpenGaussStore.from_params(
        **PARAMS,
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
    )

    yield store

    asyncio.run(store.close())


@pytest.fixture()
def opengauss_store(db: None) -> Generator[OpenGaussStore, None, None]:
    store = OpenGaussStore.from_params(
        **PARAMS,
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
    )
    yield store
    asyncio.run(store.close())


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            extra_info={"test_num": 1},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="consectetur adipiscing elit",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_list": ["test_value"]},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="sed do eiusmod tempor",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            extra_info={"test_key_2": "test_val_2"},
            embedding=_get_sample_vector(0.1),
        ),
    ]


@pytest.fixture(scope="session")
def hybrid_node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="The quick brown fox jumped over the lazy dog.",
            id_="ccc",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ccc")},
            embedding=_get_sample_vector(5.0),
        ),
        TextNode(
            text="The fox and the hound",
            id_="ddd",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="ddd")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(10.0),
        ),
    ]


@pytest.mark.skipif(postgres_not_available, reason="openGauss not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_basic_search(
    opengauss_store: OpenGaussStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await opengauss_store.async_add(node_embeddings)
    else:
        opengauss_store.add(node_embeddings)
    assert isinstance(opengauss_store, OpenGaussStore)
    assert hasattr(opengauss_store, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(1.0), similarity_top_k=1)
    if use_async:
        res = await opengauss_store.aquery(q)
    else:
        res = opengauss_store.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(postgres_not_available, reason="openGauss is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_sparse_query(
    pg_hybrid: OpenGaussStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hybrid.async_add(hybrid_node_embeddings)
    else:
        pg_hybrid.add(hybrid_node_embeddings)
    assert isinstance(pg_hybrid, OpenGaussStore)
    assert hasattr(pg_hybrid, "_engine")

    # text search should work when query is a sentence and not just a single word
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
    )

    if use_async:
        res = await pg_hybrid.aquery(q)
    else:
        res = pg_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "ccc"
    assert res.nodes[1].node_id == "ddd"


@pytest.mark.parametrize("hybird", [True, False])
def test_opengauss_init(hybird: bool) -> None:
    store = OpenGaussStore.from_params(
        **PARAMS,
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=hybird,
        embed_dim=TEST_EMBED_DIM,
    )
