import asyncio
from typing import Any, Dict, Generator, List, Union, Optional

import pytest
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.postgres import PGVectorStore

# from testing find install here https://github.com/pgvector/pgvector#installation-notes


PARAMS: Dict[str, Union[str, int]] = {
    "host": "localhost",
    "user": "postgres",
    "password": "mark90",
    "port": 5432,
}
TEST_DB = "test_vector_db"
TEST_DB_HNSW = "test_vector_db_hnsw"
TEST_TABLE_NAME = "lorem_ipsum"
TEST_SCHEMA_NAME = "test"
TEST_EMBED_DIM = 2

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
def db_hnsw(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {TEST_DB_HNSW}")
        c.execute(f"CREATE DATABASE {TEST_DB_HNSW}")
        conn.commit()
    yield
    with conn.cursor() as c:
        c.execute(f"DROP DATABASE {TEST_DB_HNSW}")
        conn.commit()


@pytest.fixture()
def pg(db: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_hybrid(db: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_indexed_metadata(db: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
        indexed_metadata_keys=[("test_text", "text"), ("test_int", "int")],
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw(db_hnsw: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB_HNSW,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw_hybrid(db_hnsw: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB_HNSW,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hybrid_search=True,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )

    yield pg

    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw_multiple(db_hnsw: None) -> Generator[List[PGVectorStore], None, None]:
    """
    This creates multiple instances of PGVectorStore.
    """
    pgs = []
    for _ in range(2):
        pg = PGVectorStore.from_params(
            **PARAMS,  # type: ignore
            database=TEST_DB_HNSW,
            table_name=TEST_TABLE_NAME,
            schema_name=TEST_SCHEMA_NAME,
            embed_dim=TEST_EMBED_DIM,
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
            },
        )
        pgs.append(pg)

    yield pgs

    for pg in pgs:
        asyncio.run(pg.close())


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


@pytest.fixture(scope="session")
def index_node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            embedding=_get_sample_vector(0.1),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(1.0),
        ),
        IndexNode(
            text="The quick brown fox jumped over the lazy dog.",
            id_="aaa_ref",
            index_id="aaa",
            embedding=_get_sample_vector(5.0),
        ),
    ]


@pytest.fixture()
def pg_halfvec(db: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME + "_halfvec",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        use_halfvec=True,
    )
    yield pg
    asyncio.run(pg.close())


@pytest.fixture()
def pg_halfvec_hybrid(db: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME + "_halfvec_hybrid",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hybrid_search=True,
        use_halfvec=True,
    )
    yield pg
    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw_halfvec(db_hnsw: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB_HNSW,
        table_name=TEST_TABLE_NAME + "_hnsw_halfvec",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        use_halfvec=True,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )
    yield pg
    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw_hybrid_halfvec(db_hnsw: None) -> Any:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB_HNSW,
        table_name=TEST_TABLE_NAME + "_hnsw_halfvec_hybrid",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        hybrid_search=True,
        use_halfvec=True,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )
    yield pg
    asyncio.run(pg.close())


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_instance_creation(db: None) -> None:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
    )
    assert isinstance(pg, PGVectorStore)

    assert pg.client is None
    await pg.close()


@pytest.fixture()
def pg_fixture(request):
    if request.param == "pg":
        return request.getfixturevalue("pg")
    elif request.param == "pg_halfvec":
        return request.getfixturevalue("pg_halfvec")
    else:
        raise ValueError(f"Unknown param: {request.param}")


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query(
    pg_fixture: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)
    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(1.0), similarity_top_k=1)
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_query_hnsw(
    pg_hnsw: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
):
    if use_async:
        await pg_hnsw.async_add(node_embeddings)
    else:
        pg_hnsw.add(node_embeddings)

    assert isinstance(pg_hnsw, PGVectorStore)
    assert hasattr(pg_hnsw, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(1.0), similarity_top_k=1)
    if use_async:
        res = await pg_hnsw.aquery(q)
    else:
        res = pg_hnsw.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters(
    pg_fixture: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)
    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters_with_in_operator(
    pg_fixture: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)
    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value", "another_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters_with_in_operator_and_single_element(
    pg_fixture: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)
    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters_with_contains_operator(
    pg_fixture: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)
    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_list",
                value="test_value",
                operator=FilterOperator.CONTAINS,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "ccc"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_with_metadata_filters_with_is_empty(
    pg_fixture: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)
    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="nonexistent_key", value=None, operator=FilterOperator.IS_EMPTY
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert res.nodes
    # All nodes should match since none have the nonexistent_key
    assert len(res.nodes) == len(node_embeddings)


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_query_and_delete(
    pg_fixture: PGVectorStore, node_embeddings: List[TextNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)
    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.1), similarity_top_k=1)

    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_sparse_query(
    pg_hybrid: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hybrid.async_add(hybrid_node_embeddings)
    else:
        pg_hybrid.add(hybrid_node_embeddings)
    assert isinstance(pg_hybrid, PGVectorStore)
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


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_hybrid_query(
    pg_hybrid: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hybrid.async_add(hybrid_node_embeddings)
    else:
        pg_hybrid.add(hybrid_node_embeddings)
    assert isinstance(pg_hybrid, PGVectorStore)
    assert hasattr(pg_hybrid, "_engine")

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )

    if use_async:
        res = await pg_hybrid.aquery(q)
    else:
        res = pg_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 3
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"

    # if sparse_top_k is not specified, it should default to similarity_top_k
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hybrid.aquery(q)
    else:
        res = pg_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 4
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"
    assert res.nodes[3].node_id == "ddd"

    # text search should work when query is a sentence and not just a single word
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hybrid.aquery(q)
    else:
        res = pg_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 4
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"
    assert res.nodes[3].node_id == "ddd"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_hybrid_query(
    pg_hnsw_hybrid: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hnsw_hybrid.async_add(hybrid_node_embeddings)
    else:
        pg_hnsw_hybrid.add(hybrid_node_embeddings)
    assert isinstance(pg_hnsw_hybrid, PGVectorStore)
    assert hasattr(pg_hnsw_hybrid, "_engine")

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )

    if use_async:
        res = await pg_hnsw_hybrid.aquery(q)
    else:
        res = pg_hnsw_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 3
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"

    # if sparse_top_k is not specified, it should default to similarity_top_k
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hnsw_hybrid.aquery(q)
    else:
        res = pg_hnsw_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 4
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"
    assert res.nodes[3].node_id == "ddd"

    # text search should work when query is a sentence and not just a single word
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hnsw_hybrid.aquery(q)
    else:
        res = pg_hnsw_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 4
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"
    assert res.nodes[3].node_id == "ddd"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_hybrid_query_with_metadata_filters(
    pg_hybrid: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    if use_async:
        await pg_hybrid.async_add(hybrid_node_embeddings)
    else:
        pg_hybrid.add(hybrid_node_embeddings)
    assert isinstance(pg_hybrid, PGVectorStore)
    assert hasattr(pg_hybrid, "_engine")
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="test_key", value="test_value")]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=10,
        filters=filters,
        mode=VectorStoreQueryMode.HYBRID,
    )
    if use_async:
        res = await pg_hybrid.aquery(q)
    else:
        res = pg_hybrid.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "bbb"
    assert res.nodes[1].node_id == "ddd"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
def test_hybrid_query_fails_if_no_query_str_provided(
    pg_hybrid: PGVectorStore, hybrid_node_embeddings: List[TextNode]
) -> None:
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(1.0),
        similarity_top_k=10,
        mode=VectorStoreQueryMode.HYBRID,
    )

    with pytest.raises(Exception) as exc:
        pg_hybrid.query(q)

        assert str(exc) == "query_str must be specified for a sparse vector query."


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_add_to_db_and_query_index_nodes(
    pg_fixture: PGVectorStore, index_node_embeddings: List[BaseNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(index_node_embeddings)
    else:
        pg_fixture.add(index_node_embeddings)
    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(5.0), similarity_top_k=2)
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    assert res.nodes[0].node_id == "aaa_ref"
    assert isinstance(res.nodes[0], IndexNode)
    assert hasattr(res.nodes[0], "index_id")
    assert res.nodes[1].node_id == "bbb"
    assert isinstance(res.nodes[1], TextNode)


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_delete_nodes(
    pg_fixture: PGVectorStore, node_embeddings: List[BaseNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)

    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    # test deleting nothing
    if use_async:
        await pg_fixture.adelete_nodes()
    else:
        pg_fixture.delete_nodes()
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i in res.ids for i in ["aaa", "bbb", "ccc"])

    # test deleting element that doesn't exist
    if use_async:
        await pg_fixture.adelete_nodes(["asdf"])
    else:
        pg_fixture.delete_nodes(["asdf"])
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i in res.ids for i in ["aaa", "bbb", "ccc"])

    # test deleting list
    if use_async:
        await pg_fixture.adelete_nodes(["aaa", "bbb"])
    else:
        pg_fixture.delete_nodes(["aaa", "bbb"])
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i not in res.ids for i in ["aaa", "bbb"])
    assert "ccc" in res.ids


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_delete_nodes_metadata(
    pg_fixture: PGVectorStore, node_embeddings: List[BaseNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)

    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    # test deleting multiple IDs but only one satisfies filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key",
                value=["test_value", "another_value"],
                operator=FilterOperator.IN,
            )
        ]
    )
    if use_async:
        await pg_fixture.adelete_nodes(["aaa", "bbb"], filters=filters)
    else:
        pg_fixture.delete_nodes(["aaa", "bbb"], filters=filters)
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i in res.ids for i in ["aaa", "ccc", "ddd"])
    assert "bbb" not in res.ids

    # test deleting one ID which satisfies the filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=1,
                operator=FilterOperator.EQ,
            )
        ]
    )
    if use_async:
        await pg_fixture.adelete_nodes(["aaa"], filters=filters)
    else:
        pg_fixture.delete_nodes(["aaa"], filters=filters)
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i not in res.ids for i in ["bbb", "aaa"])
    assert all(i in res.ids for i in ["ccc", "ddd"])

    # test deleting one ID which doesn't satisfy the filter
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_num",
                value=1,
                operator=FilterOperator.EQ,
            )
        ]
    )
    if use_async:
        await pg_fixture.adelete_nodes(["ccc"], filters=filters)
    else:
        pg_fixture.delete_nodes(["ccc"], filters=filters)
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i not in res.ids for i in ["bbb", "aaa"])
    assert all(i in res.ids for i in ["ccc", "ddd"])

    # test deleting purely based on filters
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_2",
                value="test_val_2",
                operator=FilterOperator.EQ,
            )
        ]
    )
    if use_async:
        await pg_fixture.adelete_nodes(filters=filters)
    else:
        pg_fixture.delete_nodes(filters=filters)
    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i not in res.ids for i in ["bbb", "aaa", "ddd"])
    assert "ccc" in res.ids


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_hnsw_index_creation(
    pg_hnsw_multiple: List[PGVectorStore],
    node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """
    This test will make sure that creating multiple PGVectorStores handles db initialization properly.
    """
    # calling add will make the db initialization run
    for pg in pg_hnsw_multiple:
        if use_async:
            await pg.async_add(node_embeddings)
        else:
            pg.add(node_embeddings)

    # these are the actual table and index names that PGVectorStore automatically created
    data_test_table_name = f"data_{TEST_TABLE_NAME}"
    data_test_index_name = f"data_{TEST_TABLE_NAME}_embedding_idx"

    # create a connection to the TEST_DB_HNSW database to make sure that one, and only one, index was created
    with psycopg2.connect(**PARAMS, database=TEST_DB_HNSW) as hnsw_conn:
        with hnsw_conn.cursor() as c:
            c.execute(
                f"SELECT COUNT(*) FROM pg_indexes WHERE schemaname = '{TEST_SCHEMA_NAME}' AND tablename = '{data_test_table_name}' AND indexname LIKE '{data_test_index_name}%';"
            )
            index_count = c.fetchone()[0]

    assert index_count == 1, (
        f"Expected exactly one '{data_test_index_name}' index, but found {index_count}."
    )


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("pg_fixture", ["pg", "pg_halfvec"], indirect=True)
@pytest.mark.parametrize("use_async", [True, False])
async def test_clear(
    pg_fixture: PGVectorStore, node_embeddings: List[BaseNode], use_async: bool
) -> None:
    if use_async:
        await pg_fixture.async_add(node_embeddings)
    else:
        pg_fixture.add(node_embeddings)

    assert isinstance(pg_fixture, PGVectorStore)
    assert hasattr(pg_fixture, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i in res.ids for i in ["bbb", "aaa", "ddd", "ccc"])

    if use_async:
        await pg_fixture.aclear()
    else:
        pg_fixture.clear()

    if use_async:
        res = await pg_fixture.aquery(q)
    else:
        res = pg_fixture.query(q)
    assert all(i not in res.ids for i in ["bbb", "aaa", "ddd", "ccc"])
    assert len(res.ids) == 0


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.parametrize(
    ("node_ids", "filters", "expected_node_ids"),
    [
        (["aaa", "bbb"], None, ["aaa", "bbb"]),
        (
            None,
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="test_num",
                        value=1,
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
            ["aaa"],
        ),
        (
            ["bbb", "ccc"],
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="test_key",
                        value="test_value",
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
            ["bbb"],
        ),
        (
            ["ccc"],
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="test_key",
                        value="test_value",
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
            [],
        ),
        (
            ["aaa", "bbb"],
            MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="test_num",
                        value=999,
                        operator=FilterOperator.EQ,
                    )
                ]
            ),
            [],
        ),
    ],
)
def test_get_nodes_parametrized(
    pg: PGVectorStore,
    node_embeddings: List[TextNode],
    node_ids: Optional[List[str]],
    filters: Optional[MetadataFilters],
    expected_node_ids: List[str],
) -> None:
    """Test get_nodes method with various combinations of node_ids and filters."""
    pg.add(node_embeddings)
    nodes = pg.get_nodes(node_ids=node_ids, filters=filters)
    retrieved_ids = [node.node_id for node in nodes]
    assert set(retrieved_ids) == set(expected_node_ids)
    assert len(retrieved_ids) == len(expected_node_ids)


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_custom_engines(db: None, node_embeddings: List[TextNode]) -> None:
    """Test that PGVectorStore works correctly with custom engines."""
    from sqlalchemy import create_engine
    from sqlalchemy.ext.asyncio import create_async_engine

    # Create custom engines
    engine = create_engine(
        f"postgresql+psycopg2://{PARAMS['user']}:{PARAMS['password']}@{PARAMS['host']}:{PARAMS['port']}/{TEST_DB}",
        echo=False,
    )

    async_engine = create_async_engine(
        f"postgresql+asyncpg://{PARAMS['user']}:{PARAMS['password']}@{PARAMS['host']}:{PARAMS['port']}/{TEST_DB}",
    )

    # Create PGVectorStore with custom engines
    pg = PGVectorStore(
        embed_dim=TEST_EMBED_DIM,
        engine=engine,
        async_engine=async_engine,
    )

    # Test sync add
    pg.add(node_embeddings[:2])

    # Test async add
    await pg.async_add(node_embeddings[2:])

    # Query to verify nodes were added correctly
    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)

    # Test sync query
    res = pg.query(q)
    assert len(res.nodes) == 4
    assert set(res.ids) == {"aaa", "bbb", "ccc", "ddd"}

    # Test async query
    res = await pg.aquery(q)
    assert len(res.nodes) == 4
    assert set(res.ids) == {"aaa", "bbb", "ccc", "ddd"}

    # Clean up
    await pg.aclear()
    await pg.close()
    await async_engine.dispose()
    engine.dispose()


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
def test_custom_sync_engine_only(db: None, node_embeddings: List[TextNode]) -> None:
    """Test that PGVectorStore works correctly with only a custom sync engine."""
    from sqlalchemy import create_engine

    # Create custom sync engine only
    engine = create_engine(
        f"postgresql+psycopg2://{PARAMS['user']}:{PARAMS['password']}@{PARAMS['host']}:{PARAMS['port']}/{TEST_DB}",
        echo=False,
    )

    # Create PGVectorStore with custom sync engine only
    with pytest.raises(ValueError):
        _ = PGVectorStore(
            embed_dim=TEST_EMBED_DIM,
            engine=engine,
        )


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_custom_async_engine_only(
    db: None, node_embeddings: List[TextNode]
) -> None:
    """Test that PGVectorStore works correctly with only a custom async engine."""
    from sqlalchemy.ext.asyncio import create_async_engine

    # Create custom async engine only
    async_engine = create_async_engine(
        f"postgresql+asyncpg://{PARAMS['user']}:{PARAMS['password']}@{PARAMS['host']}:{PARAMS['port']}/{TEST_DB}",
    )

    # Create PGVectorStore with custom async engine only
    with pytest.raises(ValueError):
        _ = PGVectorStore(
            embed_dim=TEST_EMBED_DIM,
            async_engine=async_engine,
        )


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_indexed_metadata(
    pg_indexed_metadata: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
) -> None:
    from sqlalchemy import text

    if pg_indexed_metadata is None:
        pytest.skip("Postgres not available")

    # add metadata keys to nodes
    for idx, node in enumerate(hybrid_node_embeddings):
        node.metadata["test_text"] = str(idx)
        node.metadata["test_int"] = idx

    await pg_indexed_metadata.async_add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )

    res = await pg_indexed_metadata.aquery(q)
    assert res.nodes
    assert len(res.nodes) == 3
    assert res.nodes[0].node_id == "aaa"
    assert res.nodes[1].node_id == "bbb"
    assert res.nodes[2].node_id == "ccc"

    # TODO: Use async_session to query that the indexes were created
    async with pg_indexed_metadata._async_session() as session:
        # Replace with your actual table name
        result = await session.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = :table_name"
            ),
            {"table_name": f"data_{TEST_TABLE_NAME}"},
        )
        indexes = result.fetchall()
        # Now check that your expected index names are present
        index_names = [row.indexname for row in indexes]
        assert f"{TEST_TABLE_NAME}_idx" in index_names
        assert f"data_{TEST_TABLE_NAME}_pkey" in index_names
        # Optionally, check the indexdef for the correct type cast
        for key, pg_type in pg_indexed_metadata.indexed_metadata_keys:
            index_name = f"{TEST_TABLE_NAME}_idx_{key}_{pg_type}"
            assert any(
                index_name == row.indexname and f"metadata_ ->> '{key}'" in row.indexdef
                for row in indexes
            ), f"Index {index_name} not found or incorrect type cast in indexdef"
        from sqlalchemy import text

        result = await session.execute(
            text("""
                EXPLAIN ANALYZE
                SELECT * FROM test.data_lorem_ipsum
                WHERE (metadata_ ->> 'test_int')::int = 42
            """)
        )
        explain_output = result.fetchall()
        for row in explain_output:
            print(row[0])
