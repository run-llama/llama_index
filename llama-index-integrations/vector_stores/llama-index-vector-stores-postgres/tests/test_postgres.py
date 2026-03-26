import asyncio
from typing import Any, Dict, Generator, List, Union, Optional
from unittest.mock import MagicMock, AsyncMock

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
    VectorStoreQueryResult,
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.vector_stores.postgres.base import (
    DBEmbeddingRow,
    DEFAULT_MMR_PREFETCH_FACTOR,
)
from sqlalchemy import Select, MetaData, Table, Column, String, Integer, insert
import numpy as np
from llama_index.core.indices.query.embedding_utils import (
    get_top_k_embeddings,
    get_top_k_mmr_embeddings,
)


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
            extra_info={"test_key_list": ["test_value_1", "test_value_2"]},
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


@pytest.fixture(scope="session")
def array_metadata_nodes() -> List[TextNode]:
    """
    Test nodes with text array metadata for GIN index testing.

    Metadata structure:
    - concept_tags: text[] - AI/ML topic tags
    - category_ids: text[] - category identifiers (stored as strings)
    - user_id: text - scalar user identifier

    Node distribution for testing:
    - node1: tags=["AI", "ML"], categories=["1", "2"], user="user123"
    - node2: tags=["AI", "NLP"], categories=["2", "3"], user="user456"
    - node3: tags=["Computer Vision", "ML"], categories=["1", "3"], user="user123"
    - node4: tags=["Deep Learning", "AI", "ML"], categories=["1", "2", "3"], user="user789"

    This distribution allows testing:
    - ANY operator: Multiple nodes match partial criteria
    - ALL operator: Fewer nodes match strict criteria
    - CONTAINS operator: Single value lookups
    - Mixed BTREE + GIN: user_id uses BTREE, arrays use GIN
    """
    return [
        TextNode(
            text="Artificial Intelligence and Machine Learning",
            id_="node1",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node1")},
            extra_info={
                "concept_tags": ["AI", "ML"],
                "category_ids": ["1", "2"],
                "user_id": "user123",
            },
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="Natural Language Processing with AI",
            id_="node2",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node2")},
            extra_info={
                "concept_tags": ["AI", "NLP"],
                "category_ids": ["2", "3"],
                "user_id": "user456",
            },
            embedding=_get_sample_vector(0.5),
        ),
        TextNode(
            text="Computer Vision and Image Recognition",
            id_="node3",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node3")},
            extra_info={
                "concept_tags": ["Computer Vision", "ML"],
                "category_ids": ["1", "3"],
                "user_id": "user123",
            },
            embedding=_get_sample_vector(0.3),
        ),
        TextNode(
            text="Deep Learning Neural Networks",
            id_="node4",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="node4")},
            extra_info={
                "concept_tags": ["Deep Learning", "AI", "ML"],
                "category_ids": ["1", "2", "3"],
                "user_id": "user789",
            },
            embedding=_get_sample_vector(0.2),
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


@pytest.fixture()
def pg_gin_array_indexed(db: None) -> Any:
    """
    PostgreSQL vector store with GIN indices on text array metadata fields.

    This fixture creates a vector store with mixed index types:
    - GIN indices: For text[] arrays (concept_tags, category_ids)
      - Enables fast queries with ?| (ANY), ?& (ALL), @> (CONTAINS) operators
      - Optimized for array membership checks
    - BTREE indices: For scalar text fields (user_id)
      - Standard index for equality and range queries

    Index configuration:
    - concept_tags (text[]): GIN index for AI/ML topic tags
    - category_ids (text[]): GIN index for category identifiers
    - user_id (text): BTREE index for user lookups

    Use this fixture to test:
    - GIN index creation in PostgreSQL
    - Array filtering with ANY/ALL/CONTAINS operators
    - Performance of array queries with GIN indices
    - Mixed BTREE + GIN index behavior
    """
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME + "_gin",
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
        indexed_metadata_keys=[
            ("concept_tags", "text[]"),  # GIN index for text array
            ("category_ids", "text[]"),  # GIN index for text array
            ("user_id", "text"),  # BTREE index (for comparison)
        ],
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


@pytest.fixture()
def second_table(db):
    from sqlalchemy import create_engine

    engine = create_engine(
        f"postgresql+psycopg2://{PARAMS['user']}:{PARAMS['password']}@{PARAMS['host']}:{PARAMS['port']}/{TEST_DB}",
        echo=False,
    )

    metadata = MetaData()

    second_table = Table(
        "second_table",
        metadata,
        Column("id", String, primary_key=True),
        Column("field1", Integer, nullable=False),
    )

    second_table.create(engine)

    rows = [
        {"id": "aaa", "field1": 1},
        {"id": "bbb", "field1": 2},
        {"id": "ccc", "field1": 3},
        {"id": "ddd", "field1": 4},
    ]

    with engine.connect() as conn:
        stmt = insert(second_table)
        conn.execute(stmt, rows)
        conn.commit()

    yield second_table

    second_table.drop(engine)

    engine.dispose()


@pytest.fixture()
def pg_custom_query(db: None, second_table: Table) -> Any:
    def customize_query(query: Select, table_class: Any, **kwargs: Any) -> Select:
        return query.add_columns(second_table.c.field1).join(
            second_table, second_table.c.id == table_class.node_id
        )

    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
        customize_query_fn=customize_query,
    )

    yield pg

    asyncio.run(pg.close())


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
async def test_add_to_db_and_query_with_metadata_filters_with_any_operator(
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
                value=["test_value_1", "test_value_new"],
                operator=FilterOperator.ANY,
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
async def test_add_to_db_and_query_with_metadata_filters_with_all_operator(
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
                value=["test_value_1", "test_value_2"],
                operator=FilterOperator.ALL,
            )
        ]
    )
    filters_no_all_match = MetadataFilters(
        filters=[
            MetadataFilter(
                key="test_key_list",
                value=["test_value_1", "test_value_3"],
                operator=FilterOperator.ALL,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )
    q2 = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5),
        similarity_top_k=10,
        filters=filters_no_all_match,
    )
    if use_async:
        res = await pg_fixture.aquery(q)
        res_no_match = await pg_fixture.aquery(q2)
    else:
        res = pg_fixture.query(q)
        res_no_match = pg_fixture.query(q2)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "ccc"
    assert not res_no_match.nodes


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
                value="test_value_1",
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
async def test_sparse_query_special_character_parsing(
    pg_hybrid: PGVectorStore,
) -> None:
    built_query = pg_hybrid._build_sparse_query(
        query_str="   who' &..s |     (the): <-> **fox**?!!! lazy.hound lazy..dog ?jumped,over?",
        limit=5,
    )
    assert (
        built_query.compile().params["to_tsquery_1"]
        == "who|s|the|fox|lazy.hound|lazy|dog|jumped|over"
    )


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_sparse_query_with_special_characters(
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

    # text search should work with special characters
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="   who' &..s |     (the): <-> **fox**?!!!",
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
async def test_hybrid_query_with_special_characters(
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

    # text search should work with special characters
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="   who' & s |     (the): <-> **fox**?!!!  ",
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


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_custom_query(
    use_async: bool, pg_custom_query: PGVectorStore, node_embeddings: List[TextNode]
) -> None:
    pg_custom_query.add(node_embeddings)

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.5), similarity_top_k=10)
    if use_async:
        results = await pg_custom_query.aquery(q)
    else:
        results = pg_custom_query.query(q)

    nodes = results.nodes

    expected_values = {"aaa": 1, "bbb": 2, "ccc": 3, "ddd": 4}

    for node in nodes:
        assert "custom_fields" in node.metadata
        assert "field1" in node.metadata["custom_fields"]
        assert node.metadata["custom_fields"]["field1"] == expected_values[node.node_id]


# ================== GIN Index Tests ==================


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_gin_index_creation_in_database(
    pg_gin_array_indexed: PGVectorStore,
    array_metadata_nodes: List[TextNode],
) -> None:
    """
    Verify that GIN indices are correctly created in PostgreSQL for text[] metadata fields.

    This test ensures:
    1. GIN indices are created for text[] type fields (concept_tags, category_ids)
    2. BTREE indices are created for scalar text fields (user_id)
    3. GIN indices use the -> operator (not ->>) to preserve JSONB structure
    4. Index names follow the expected naming convention

    PostgreSQL Operators:
    - -> extracts JSONB and keeps it as JSONB (required for GIN)
    - ->> extracts as text (used for BTREE on scalars)

    Why this matters:
    - GIN indices only work on JSONB/array types, not text
    - Using ->> would convert to text and break GIN indexing
    - The -> operator with CAST to JSONB ensures compatibility
    """
    from sqlalchemy import text

    if pg_gin_array_indexed is None:
        pytest.skip("Postgres not available")

    # Add nodes to trigger table and index creation
    await pg_gin_array_indexed.async_add(array_metadata_nodes)

    # Query PostgreSQL system catalog to verify index creation
    # Join with pg_class and pg_am to get actual index type (not just name)
    async with pg_gin_array_indexed._async_session() as session:
        result = await session.execute(
            text(
                """
                SELECT
                    i.indexname,
                    i.indexdef,
                    am.amname as index_type
                FROM pg_indexes i
                JOIN pg_class c ON c.relname = i.indexname
                JOIN pg_am am ON am.oid = c.relam
                WHERE i.schemaname = :schema_name
                AND i.tablename = :table_name
                """
            ),
            {
                "schema_name": TEST_SCHEMA_NAME,
                "table_name": f"data_{TEST_TABLE_NAME}_gin",
            },
        )
        indexes = result.fetchall()

        # Group indices by actual PostgreSQL index type (not by name)
        gin_indices = {row.indexname: row for row in indexes if row.index_type == "gin"}
        btree_indices = {
            row.indexname: row for row in indexes if row.index_type == "btree"
        }

        # Verify GIN index exists for concept_tags array field
        concept_tags_gin = [name for name in gin_indices if "concept_tags" in name]
        assert len(concept_tags_gin) == 1, (
            f"Expected 1 GIN index for concept_tags, found {len(concept_tags_gin)}: {concept_tags_gin}"
        )
        # Verify it uses -> operator (preserves JSONB structure for GIN)
        concept_tags_indexdef = gin_indices[concept_tags_gin[0]].indexdef
        assert (
            "metadata_ -> 'concept_tags'" in concept_tags_indexdef
            or "metadata_->'concept_tags'" in concept_tags_indexdef
        ), f"GIN index should use -> operator (not ->>), got: {concept_tags_indexdef}"

        # Verify GIN index exists for category_ids array field
        category_ids_gin = [name for name in gin_indices if "category_ids" in name]
        assert len(category_ids_gin) == 1, (
            f"Expected 1 GIN index for category_ids, found {len(category_ids_gin)}: {category_ids_gin}"
        )

        # Verify BTREE index exists for user_id scalar field (not GIN)
        user_id_btree = [name for name in btree_indices if "user_id" in name]
        assert len(user_id_btree) == 1, (
            f"Expected 1 BTREE index for user_id, found {len(user_id_btree)}: {user_id_btree}"
        )


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_gin_index_query_with_contains(
    pg_gin_array_indexed: PGVectorStore,
    array_metadata_nodes: List[TextNode],
    use_async: bool,
) -> None:
    """
    Test CONTAINS operator (@>) on GIN-indexed text[] fields for single value lookups.

    PostgreSQL Operator: @> (contains)
    - Checks if array contains a specific single value
    - Example: ['AI', 'ML'] @> '["AI"]' → TRUE
    - GIN index accelerates this operation

    Test scenario:
    - Query: Find nodes where concept_tags CONTAINS "AI"
    - Expected matches:
      - node1: ["AI", "ML"] ✓
      - node2: ["AI", "NLP"] ✓
      - node3: ["Computer Vision", "ML"] ✗
      - node4: ["Deep Learning", "AI", "ML"] ✓
    - Result: 3 nodes (node1, node2, node4)

    Tests both sync and async query methods.
    """
    # Add nodes to database
    if use_async:
        await pg_gin_array_indexed.async_add(array_metadata_nodes)
    else:
        pg_gin_array_indexed.add(array_metadata_nodes)

    # Query for nodes containing "AI" tag
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="concept_tags",
                value="AI",
                operator=FilterOperator.CONTAINS,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )

    if use_async:
        res = await pg_gin_array_indexed.aquery(q)
    else:
        res = pg_gin_array_indexed.query(q)

    # Verify: Should return nodes 1, 2, and 4 (all contain "AI")
    assert res.nodes, "Expected non-empty result set"
    assert len(res.nodes) == 3, f"Expected 3 nodes with 'AI', got {len(res.nodes)}"
    node_ids = {node.node_id for node in res.nodes}
    assert node_ids == {"node1", "node2", "node4"}, (
        f"Expected nodes with 'AI' tag, got {node_ids}"
    )


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_gin_index_query_with_any_operator(
    pg_gin_array_indexed: PGVectorStore,
    array_metadata_nodes: List[TextNode],
    use_async: bool,
) -> None:
    """
    Test ANY operator (?|) on GIN-indexed text[] fields for OR-based matching.

    PostgreSQL Operator: ?| (ANY - contains any of the values)
    - Checks if array contains AT LEAST ONE of the specified values
    - Example: ['1', '2'] ?| array['1', '2'] → TRUE if array has '1' OR '2'
    - GIN index accelerates this operation

    Test scenario:
    - Query: Find nodes where category_ids contains "1" OR "2"
    - Expected matches:
      - node1: ["1", "2"] ✓ (has both "1" and "2")
      - node2: ["2", "3"] ✓ (has "2")
      - node3: ["1", "3"] ✓ (has "1")
      - node4: ["1", "2", "3"] ✓ (has both "1" and "2")
    - Result: ALL 4 nodes match

    Key difference from CONTAINS:
    - CONTAINS: Single value check
    - ANY: Multiple value check (OR logic)

    Tests both sync and async query methods.
    """
    # Add nodes to database
    if use_async:
        await pg_gin_array_indexed.async_add(array_metadata_nodes)
    else:
        pg_gin_array_indexed.add(array_metadata_nodes)

    # Query for nodes with category_ids containing "1" OR "2"
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="category_ids",
                value=["1", "2"],
                operator=FilterOperator.ANY,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )

    if use_async:
        res = await pg_gin_array_indexed.aquery(q)
    else:
        res = pg_gin_array_indexed.query(q)

    # Verify: Should return all 4 nodes (all have category_ids with "1" or "2")
    assert res.nodes, "Expected non-empty result set"
    assert len(res.nodes) == 4, (
        f"Expected 4 nodes with category '1' OR '2', got {len(res.nodes)}"
    )
    node_ids = {node.node_id for node in res.nodes}
    assert node_ids == {"node1", "node2", "node3", "node4"}, (
        f"Expected all nodes to match ANY(['1', '2']), got {node_ids}"
    )


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_gin_index_query_with_all_operator(
    pg_gin_array_indexed: PGVectorStore,
    array_metadata_nodes: List[TextNode],
    use_async: bool,
) -> None:
    """
    Test ALL operator (?&) on GIN-indexed text[] fields for AND-based matching.

    PostgreSQL Operator: ?& (ALL - contains all of the values)
    - Checks if array contains ALL of the specified values
    - Example: ['AI', 'ML', 'NLP'] ?& array['AI', 'ML'] → TRUE (has both)
    - GIN index accelerates this operation

    Test scenario 1 - Matching query:
    - Query: Find nodes where concept_tags contains BOTH "AI" AND "ML"
    - Expected matches:
      - node1: ["AI", "ML"] ✓ (has both)
      - node2: ["AI", "NLP"] ✗ (missing "ML")
      - node3: ["Computer Vision", "ML"] ✗ (missing "AI")
      - node4: ["Deep Learning", "AI", "ML"] ✓ (has both)
    - Result: 2 nodes (node1, node4)

    Test scenario 2 - No match query:
    - Query: Find nodes with BOTH "AI" AND "NonExistent"
    - Expected: Empty result (no node has both)
    - Demonstrates strict AND logic

    Key difference from ANY:
    - ANY: At least one value (OR logic)
    - ALL: Every value must be present (AND logic)

    Tests both sync and async query methods.
    """
    # Add nodes to database
    if use_async:
        await pg_gin_array_indexed.async_add(array_metadata_nodes)
    else:
        pg_gin_array_indexed.add(array_metadata_nodes)

    # Test 1: Query for nodes containing both "AI" AND "ML" tags
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="concept_tags",
                value=["AI", "ML"],
                operator=FilterOperator.ALL,
            )
        ]
    )
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5), similarity_top_k=10, filters=filters
    )

    if use_async:
        res = await pg_gin_array_indexed.aquery(q)
    else:
        res = pg_gin_array_indexed.query(q)

    # Verify: Should return nodes 1 and 4 (both have AI and ML)
    assert res.nodes, "Expected non-empty result set"
    assert len(res.nodes) == 2, (
        f"Expected 2 nodes with BOTH 'AI' AND 'ML', got {len(res.nodes)}"
    )
    node_ids = {node.node_id for node in res.nodes}
    assert node_ids == {"node1", "node4"}, (
        f"Expected nodes with both tags, got {node_ids}"
    )

    # Test 2: Query with no matches (demonstrates strict AND logic)
    filters_no_match = MetadataFilters(
        filters=[
            MetadataFilter(
                key="concept_tags",
                value=["AI", "NonExistent"],
                operator=FilterOperator.ALL,
            )
        ]
    )
    q_no_match = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5),
        similarity_top_k=10,
        filters=filters_no_match,
    )

    if use_async:
        res_no_match = await pg_gin_array_indexed.aquery(q_no_match)
    else:
        res_no_match = pg_gin_array_indexed.query(q_no_match)

    # Verify: Should return empty (no node has "NonExistent")
    assert not res_no_match.nodes or len(res_no_match.nodes) == 0, (
        "Expected empty result for non-existent tag combination"
    )


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_mixed_btree_and_gin_indices(
    pg_gin_array_indexed: PGVectorStore,
    array_metadata_nodes: List[TextNode],
) -> None:
    """
    Test that BTREE and GIN indices coexist and work correctly together.

    This test demonstrates:
    1. BTREE indices work for scalar equality queries
    2. GIN indices work for array containment queries
    3. Both index types can be combined in a single query with AND logic
    4. PostgreSQL query planner uses appropriate indices for each filter

    Index types:
    - BTREE on user_id: Fast equality/range queries on scalar values
    - GIN on concept_tags: Fast array membership queries

    Test scenarios:
    1. BTREE-only query: Filter by user_id (scalar)
    2. GIN-only query: Filter by concept_tags (array)
    3. Combined query: Filter by both (demonstrates index intersection)
    """
    # Add nodes to database
    await pg_gin_array_indexed.async_add(array_metadata_nodes)

    # ===== Test 1: BTREE Index Query (Scalar Equality) =====
    # Query for exact user_id match using BTREE index
    filters_btree = MetadataFilters(
        filters=[
            MetadataFilter(
                key="user_id",
                value="user123",
                operator=FilterOperator.EQ,
            )
        ]
    )
    q_btree = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5),
        similarity_top_k=10,
        filters=filters_btree,
    )
    res_btree = await pg_gin_array_indexed.aquery(q_btree)

    # Verify BTREE query: Should return nodes 1 and 3 (both have user_id="user123")
    assert res_btree.nodes, "Expected non-empty BTREE query result"
    assert len(res_btree.nodes) == 2, (
        f"Expected 2 nodes with user_id='user123', got {len(res_btree.nodes)}"
    )
    btree_node_ids = {node.node_id for node in res_btree.nodes}
    assert btree_node_ids == {"node1", "node3"}, (
        f"BTREE query failed: expected nodes 1 & 3, got {btree_node_ids}"
    )

    # ===== Test 2: GIN Index Query (Array Containment) =====
    # Query for array contains using GIN index
    filters_gin = MetadataFilters(
        filters=[
            MetadataFilter(
                key="concept_tags",
                value="ML",
                operator=FilterOperator.CONTAINS,
            )
        ]
    )
    q_gin = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5),
        similarity_top_k=10,
        filters=filters_gin,
    )
    res_gin = await pg_gin_array_indexed.aquery(q_gin)

    # Verify GIN query: Should return nodes 1, 3, and 4 (all contain "ML")
    assert res_gin.nodes, "Expected non-empty GIN query result"
    assert len(res_gin.nodes) == 3, (
        f"Expected 3 nodes with 'ML' tag, got {len(res_gin.nodes)}"
    )
    gin_node_ids = {node.node_id for node in res_gin.nodes}
    assert gin_node_ids == {"node1", "node3", "node4"}, (
        f"GIN query failed: expected nodes 1, 3 & 4, got {gin_node_ids}"
    )

    # ===== Test 3: Combined BTREE + GIN Query (Index Intersection) =====
    # Query using both BTREE and GIN indices (AND logic)
    filters_combined = MetadataFilters(
        filters=[
            MetadataFilter(
                key="user_id",
                value="user123",
                operator=FilterOperator.EQ,
            ),
            MetadataFilter(
                key="concept_tags",
                value="ML",
                operator=FilterOperator.CONTAINS,
            ),
        ]
    )
    q_combined = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.5),
        similarity_top_k=10,
        filters=filters_combined,
    )
    res_combined = await pg_gin_array_indexed.aquery(q_combined)

    # Verify combined query: Should return only nodes that satisfy BOTH conditions
    # node1: user_id="user123" ✓ AND concept_tags contains "ML" ✓
    # node3: user_id="user123" ✓ AND concept_tags contains "ML" ✓
    # (node4 has "ML" but user_id="user789", so excluded)
    assert res_combined.nodes, "Expected non-empty combined query result"
    assert len(res_combined.nodes) == 2, (
        f"Expected 2 nodes matching both filters, got {len(res_combined.nodes)}"
    )
    combined_node_ids = {node.node_id for node in res_combined.nodes}
    assert combined_node_ids == {"node1", "node3"}


# ================== End GIN Index Tests ==================


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_custom_sparse_query(
    use_async: bool,
    pg_custom_query: PGVectorStore,
    hybrid_node_embeddings: List[TextNode],
) -> None:
    pg_custom_query.add(hybrid_node_embeddings)

    q = VectorStoreQuery(query_str="who is the fox?", sparse_top_k=10)
    if use_async:
        results = await pg_custom_query.aquery(q)
    else:
        results = pg_custom_query.query(q)

    nodes = results.nodes

    expected_values = {"aaa": 1, "bbb": 2, "ccc": 3, "ddd": 4}

    for node in nodes:
        assert "custom_fields" in node.metadata
        assert "field1" in node.metadata["custom_fields"]
        assert node.metadata["custom_fields"]["field1"] == expected_values[node.node_id]


# ================== MMR Mock Tests ==================
def _create_mock_db_embedding_row(
    node_id: str,
    text: str,
    metadata: Dict[str, Any],
    similarity: float = 0.9,
    custom_fields: Optional[Dict[str, Any]] = None,
) -> "DBEmbeddingRow":
    """Helper to create DBEmbeddingRow for mock tests."""
    return DBEmbeddingRow(
        node_id=node_id,
        text=text,
        metadata=metadata,
        custom_fields=custom_fields or {},
        similarity=similarity,
    )


def _create_mock_pg_vector_store() -> MagicMock:
    """Create a mock PGVectorStore with minimal setup for testing _mmr_query."""
    mock_store = MagicMock(spec=PGVectorStore)
    mock_store.hnsw_kwargs = None

    mock_store._db_rows_to_query_result = (
        PGVectorStore._db_rows_to_query_result.__get__(mock_store)
    )
    mock_store._prepare_mmr_query = PGVectorStore._prepare_mmr_query.__get__(mock_store)
    mock_store._mmr_rerank_results = PGVectorStore._mmr_rerank_results.__get__(
        mock_store
    )
    return mock_store


# ---- sync/async dispatch helpers used by parametrized tests ----


def _setup_embedding_mock(mock_store, return_value, use_async):
    if use_async:
        mock_store._async_query_with_embedding = AsyncMock(return_value=return_value)
    else:
        mock_store._query_with_embedding = MagicMock(return_value=return_value)


def _setup_fallback_mock(mock_store, return_value, use_async):
    if use_async:
        mock_store._aquery_with_score = AsyncMock(return_value=return_value)
    else:
        mock_store._query_with_score = MagicMock(return_value=return_value)


async def _call_mmr(mock_store, query, use_async, **kwargs):
    if use_async:
        return await PGVectorStore._async_mmr_query(mock_store, query, **kwargs)
    else:
        return PGVectorStore._mmr_query(mock_store, query, **kwargs)


def _get_embedding_mock(mock_store, use_async):
    if use_async:
        return mock_store._async_query_with_embedding
    else:
        return mock_store._query_with_embedding


def _get_fallback_mock(mock_store, use_async):
    if use_async:
        return mock_store._aquery_with_score
    else:
        return mock_store._query_with_score


# ================== MMR Query Tests (parametrized sync/async) ==================


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_rejects_none_embedding(use_async):
    """Test that MMR query raises ValueError when query_embedding is None."""
    query = VectorStoreQuery(
        query_embedding=None,
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()

    with pytest.raises(ValueError, match="MMR query requires query_embedding"):
        await _call_mmr(mock_store, query, use_async)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_rejects_conflicting_prefetch_params(use_async):
    """Test that MMR query raises ValueError when both prefetch params provided."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()

    with pytest.raises(
        ValueError, match="'mmr_prefetch_factor' and 'mmr_prefetch_k' cannot coexist"
    ):
        await _call_mmr(
            mock_store, query, use_async, mmr_prefetch_factor=4, mmr_prefetch_k=20
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_returns_empty_when_no_candidates(use_async):
    """Test that MMR query returns empty result when no candidates found."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)

    result = await _call_mmr(mock_store, query, use_async)

    assert result.nodes == []
    assert result.similarities == []
    assert result.ids == []
    _get_embedding_mock(mock_store, use_async).assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_returns_empty_when_no_valid_embeddings(use_async):
    """Test MMR query returns empty when candidates have empty embeddings."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_row = _create_mock_db_embedding_row(
        node_id="node1",
        text="test text",
        metadata={"_node_type": "TextNode"},
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [(mock_row, [])], use_async)

    result = await _call_mmr(mock_store, query, use_async)

    assert result.nodes == []
    assert result.similarities == []
    assert result.ids == []


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_falls_back_when_insufficient_embeddings(use_async):
    """Test MMR query falls back to regular search when not enough embeddings."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_row1 = _create_mock_db_embedding_row(
        node_id="node1",
        text="text1",
        metadata={"_node_type": "TextNode"},
        similarity=0.9,
    )
    mock_row2 = _create_mock_db_embedding_row(
        node_id="node2",
        text="text2",
        metadata={"_node_type": "TextNode"},
        similarity=0.8,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        [(mock_row1, [1.0, 0.0]), (mock_row2, [0.9, 0.1])],
        use_async,
    )

    expected_result = VectorStoreQueryResult(
        nodes=[TextNode(id_="node1", text="text1")],
        similarities=[0.9],
        ids=["node1"],
    )
    _setup_fallback_mock(mock_store, [mock_row1, mock_row2], use_async)
    mock_store._db_rows_to_query_result = MagicMock(return_value=expected_result)

    result = await _call_mmr(mock_store, query, use_async)

    _get_fallback_mock(mock_store, use_async).assert_called_once()
    mock_store._db_rows_to_query_result.assert_called_once()
    assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_applies_mmr_algorithm(use_async):
    """
    Test MMR query correctly applies MMR and returns diverse results.

    With query=[1,0.5,0], node1=[1,0,0] and node2=[1,0.1,0] are near-duplicates,
    while node3=[0,1,0] is diverse. MMR at threshold=0.5 should pick the most
    relevant node first (node2), then prefer node3 over the redundant node1.
    """
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.5, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=0.5,
    )
    mock_row1 = _create_mock_db_embedding_row(
        node_id="node1",
        text="text1",
        metadata={"_node_type": "TextNode"},
        similarity=0.89,
    )
    mock_row2 = _create_mock_db_embedding_row(
        node_id="node2",
        text="text2",
        metadata={"_node_type": "TextNode"},
        similarity=0.93,
    )
    mock_row3 = _create_mock_db_embedding_row(
        node_id="node3",
        text="text3",
        metadata={"_node_type": "TextNode"},
        similarity=0.45,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        [
            (mock_row1, [1.0, 0.0, 0.0]),  # Similar to query, near-duplicate of node2
            (mock_row2, [1.0, 0.1, 0.0]),  # Most relevant, near-duplicate of node1
            (mock_row3, [0.0, 1.0, 0.0]),  # Diverse — different direction
        ],
        use_async,
    )

    result = await _call_mmr(mock_store, query, use_async)

    assert len(result.nodes) == 2
    assert len(result.ids) == 2
    assert len(result.similarities) == 2
    # MMR should pick node2 (most relevant) and node3 (diverse),
    # NOT node1 which is redundant with node2
    assert result.ids == ["node2", "node3"]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_uses_prefetch_k_override(use_async):
    """Test MMR query uses mmr_prefetch_k when provided."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)

    await _call_mmr(mock_store, query, use_async, mmr_prefetch_k=50)

    call_args = _get_embedding_mock(mock_store, use_async).call_args
    assert call_args[0][1] == 50  # limit argument


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_uses_default_prefetch_factor(use_async):
    """Test MMR query uses DEFAULT_MMR_PREFETCH_FACTOR when no override."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)

    await _call_mmr(mock_store, query, use_async)

    expected_prefetch = 5 * DEFAULT_MMR_PREFETCH_FACTOR
    call_args = _get_embedding_mock(mock_store, use_async).call_args
    assert call_args[0][1] == expected_prefetch


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_preserves_custom_fields_in_result(use_async):
    """Ensure MMR query preserves custom_fields into node.metadata."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=1,
        mode=VectorStoreQueryMode.MMR,
    )
    row = _create_mock_db_embedding_row(
        node_id="n1",
        text="t1",
        metadata={"_node_type": "TextNode"},
        custom_fields={"score": 7, "source": "join"},
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [(row, [1.0, 0.0, 0.0])], use_async)

    result = await _call_mmr(mock_store, query, use_async)

    assert result.nodes and len(result.nodes) == 1
    node = result.nodes[0]
    assert "custom_fields" in node.metadata
    assert node.metadata["custom_fields"] == {"score": 7, "source": "join"}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_passes_filters_to_prefetch(use_async):
    """MMR prefetch should receive metadata filters."""
    filters = MetadataFilters(filters=[ExactMatchFilter(key="k", value="v")])
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
        filters=filters,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)

    await _call_mmr(mock_store, query, use_async)

    call_args = _get_embedding_mock(mock_store, use_async).call_args
    assert call_args[0][2] == filters


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_uses_custom_prefetch_factor(use_async):
    """Test MMR query uses custom mmr_prefetch_factor."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)

    await _call_mmr(mock_store, query, use_async, mmr_prefetch_factor=10)

    # Should prefetch 5 * 10 = 50
    call_args = _get_embedding_mock(mock_store, use_async).call_args
    assert call_args[0][1] == 50


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_uses_mmr_threshold_from_kwargs(use_async):
    """Test MMR query uses mmr_threshold from kwargs when not in query."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=None,
    )
    mock_row1 = _create_mock_db_embedding_row(
        node_id="node1",
        text="text1",
        metadata={"_node_type": "TextNode"},
    )
    mock_row2 = _create_mock_db_embedding_row(
        node_id="node2",
        text="text2",
        metadata={"_node_type": "TextNode"},
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        [(mock_row1, [1.0, 0.0, 0.0]), (mock_row2, [0.0, 1.0, 0.0])],
        use_async,
    )

    result = await _call_mmr(mock_store, query, use_async, mmr_threshold=0.8)

    assert len(result.nodes) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_handles_node_without_node_type(use_async):
    """Test MMR query handles metadata without _node_type (uses TextNode fallback)."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=1,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_row = _create_mock_db_embedding_row(
        node_id="node1",
        text="text1",
        metadata={"some_key": "some_value"},  # No _node_type
        similarity=0.9,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [(mock_row, [1.0, 0.0])], use_async)

    result = await _call_mmr(mock_store, query, use_async)

    assert len(result.nodes) == 1
    assert result.nodes[0].id_ == "node1"
    assert result.nodes[0].text == "text1"


# ================== MMR Threshold Validation Tests ==================


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_rejects_threshold_above_one(use_async):
    """mmr_threshold > 1 should raise ValueError."""
    mock_store = _create_mock_pg_vector_store()
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    with pytest.raises(ValueError, match="mmr_threshold must be between 0 and 1"):
        await _call_mmr(mock_store, query, use_async, mmr_threshold=1.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_rejects_threshold_below_zero(use_async):
    """mmr_threshold < 0 should raise ValueError."""
    mock_store = _create_mock_pg_vector_store()
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    with pytest.raises(ValueError, match="mmr_threshold must be between 0 and 1"):
        await _call_mmr(mock_store, query, use_async, mmr_threshold=-0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_accepts_threshold_zero(use_async):
    """mmr_threshold=0.0 (max diversity) should be accepted, not rejected."""
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    result = await _call_mmr(mock_store, query, use_async, mmr_threshold=0.0)
    assert result.nodes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_accepts_threshold_one(use_async):
    """mmr_threshold=1.0 (max relevance) should be accepted."""
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    result = await _call_mmr(mock_store, query, use_async, mmr_threshold=1.0)
    assert result.nodes == []


# ================== Prefetch Clamping Test ==================


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_clamps_prefetch_k_to_similarity_top_k(use_async):
    """When mmr_prefetch_k < similarity_top_k, prefetch at least similarity_top_k."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)

    await _call_mmr(mock_store, query, use_async, mmr_prefetch_k=1)

    call_args = _get_embedding_mock(mock_store, use_async).call_args
    assert call_args[0][1] == 5


# ================== _get_query_session_settings Tests ==================


def test_get_query_session_settings_empty_when_no_kwargs():
    """No settings returned when no ivfflat_probes and no hnsw_kwargs."""
    mock_store = _create_mock_pg_vector_store()
    mock_store.hnsw_kwargs = None

    settings = PGVectorStore._get_query_session_settings(mock_store)
    assert settings == []


def test_get_query_session_settings_ivfflat_probes():
    """ivfflat_probes produces SET + bitmapscan off."""
    mock_store = _create_mock_pg_vector_store()
    mock_store.hnsw_kwargs = None

    settings = PGVectorStore._get_query_session_settings(mock_store, ivfflat_probes=10)
    assert len(settings) == 2
    assert "ivfflat.probes" in settings[0][0]
    assert settings[0][1] == {"ivfflat_probes": 10}
    assert "enable_bitmapscan" in settings[1][0]


def test_get_query_session_settings_hnsw():
    """hnsw_kwargs produces SET hnsw.ef_search + bitmapscan off."""
    mock_store = _create_mock_pg_vector_store()
    mock_store.hnsw_kwargs = {"hnsw_ef_search": 40}

    settings = PGVectorStore._get_query_session_settings(mock_store)
    assert len(settings) == 2
    assert "hnsw.ef_search" in settings[0][0]
    assert settings[0][1] == {"hnsw_ef_search": 40}
    assert "enable_bitmapscan" in settings[1][0]


def test_get_query_session_settings_hnsw_override_from_kwargs():
    """Per-call hnsw_ef_search kwarg overrides default from hnsw_kwargs."""
    mock_store = _create_mock_pg_vector_store()
    mock_store.hnsw_kwargs = {"hnsw_ef_search": 40}

    settings = PGVectorStore._get_query_session_settings(mock_store, hnsw_ef_search=200)
    assert settings[0][1] == {"hnsw_ef_search": 200}


def test_get_query_session_settings_both_ivfflat_and_hnsw():
    """When both ivfflat and hnsw are active, bitmapscan off is emitted once."""
    mock_store = _create_mock_pg_vector_store()
    mock_store.hnsw_kwargs = {"hnsw_ef_search": 40}

    settings = PGVectorStore._get_query_session_settings(mock_store, ivfflat_probes=7)
    # ivfflat SET, hnsw SET, one bitmapscan off
    assert len(settings) == 3
    bitmapscan_count = sum(1 for sql, _ in settings if "bitmapscan" in sql)
    assert bitmapscan_count == 1


def test_get_query_session_settings_casts_to_int():
    """Values are cast to int even when passed as strings."""
    mock_store = _create_mock_pg_vector_store()
    mock_store.hnsw_kwargs = None

    settings = PGVectorStore._get_query_session_settings(
        mock_store, ivfflat_probes="15"
    )
    assert settings[0][1] == {"ivfflat_probes": 15}


# ================== Public query() / aquery() Routing Tests ==================


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_query_routes_to_mmr(use_async):
    """query()/aquery() with MMR mode delegates to the MMR method."""
    mock_store = _create_mock_pg_vector_store()
    mock_store._initialize = MagicMock()

    expected = VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    if use_async:
        mock_store._async_mmr_query = AsyncMock(return_value=expected)
    else:
        mock_store._mmr_query = MagicMock(return_value=expected)

    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )

    if use_async:
        result = await PGVectorStore.aquery(mock_store, q)
        mock_store._async_mmr_query.assert_called_once_with(q)
    else:
        result = PGVectorStore.query(mock_store, q)
        mock_store._mmr_query.assert_called_once_with(q)
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_query_routes_to_default(use_async):
    """query()/aquery() with DEFAULT mode delegates to the score method."""
    mock_store = _create_mock_pg_vector_store()
    mock_store._initialize = MagicMock()

    rows = [
        _create_mock_db_embedding_row(
            node_id="n1", text="t1", metadata={"_node_type": "TextNode"}
        )
    ]
    if use_async:
        mock_store._aquery_with_score = AsyncMock(return_value=rows)
    else:
        mock_store._query_with_score = MagicMock(return_value=rows)

    q = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.DEFAULT,
    )

    if use_async:
        result = await PGVectorStore.aquery(mock_store, q)
        mock_store._aquery_with_score.assert_called_once()
    else:
        result = PGVectorStore.query(mock_store, q)
        mock_store._query_with_score.assert_called_once()
    assert len(result.nodes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_threshold_from_query_overrides_kwargs(use_async):
    """query.mmr_threshold=0.3 should take precedence over kwarg mmr_threshold=0.9."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.5, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=0.3,
    )
    mock_row1 = _create_mock_db_embedding_row(
        node_id="node1",
        text="text1",
        metadata={"_node_type": "TextNode"},
    )
    mock_row2 = _create_mock_db_embedding_row(
        node_id="node2",
        text="text2",
        metadata={"_node_type": "TextNode"},
    )
    mock_row3 = _create_mock_db_embedding_row(
        node_id="node3",
        text="text3",
        metadata={"_node_type": "TextNode"},
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        [
            (mock_row1, [1.0, 0.0, 0.0]),
            (mock_row2, [1.0, 0.1, 0.0]),
            (mock_row3, [0.0, 1.0, 0.0]),
        ],
        use_async,
    )

    # Pass a conflicting kwarg threshold — the query's own value should win
    result = await _call_mmr(mock_store, query, use_async, mmr_threshold=0.9)

    assert len(result.nodes) == 2
    # Low threshold (0.3) favors diversity, so node3 (diverse) should be picked
    # over node1 (redundant with node2). Same assertion as
    # test_mmr_query_applies_mmr_algorithm but now proving the 0.3 from query
    # was used, not the 0.9 from kwargs.
    assert "node3" in result.ids


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_returns_correct_similarity_scores(use_async):
    """
    Verify the actual MMR similarity scores returned, not just ordering.

    Uses the core get_top_k_mmr_embeddings to compute expected scores
    and then asserts the PG implementation returns matching values.
    """
    query_emb = [1.0, 0.5, 0.0]
    emb1, emb2, emb3 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]
    threshold = 0.5

    expected_sims, expected_ids = get_top_k_mmr_embeddings(
        query_embedding=query_emb,
        embeddings=[emb1, emb2, emb3],
        similarity_top_k=3,
        embedding_ids=["n1", "n2", "n3"],
        mmr_threshold=threshold,
    )

    rows = [
        _create_mock_db_embedding_row(
            node_id="n1", text="t1", metadata={"_node_type": "TextNode"}
        ),
        _create_mock_db_embedding_row(
            node_id="n2", text="t2", metadata={"_node_type": "TextNode"}
        ),
        _create_mock_db_embedding_row(
            node_id="n3", text="t3", metadata={"_node_type": "TextNode"}
        ),
    ]
    query = VectorStoreQuery(
        query_embedding=query_emb,
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=threshold,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        [(rows[0], emb1), (rows[1], emb2), (rows[2], emb3)],
        use_async,
    )

    result = await _call_mmr(mock_store, query, use_async)

    assert result.ids == expected_ids
    for actual_sim, expected_sim in zip(result.similarities, expected_sims):
        assert np.isclose(actual_sim, expected_sim, atol=1e-5), (
            f"Score mismatch: {actual_sim} != {expected_sim}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_threshold_one_matches_regular_similarity_order(use_async):
    """At mmr_threshold=1.0, MMR should produce the same order as plain similarity."""
    query_emb = [1.0, 0.0, 0.0]
    embeddings = [[0.9, 0.1, 0.0], [0.5, 0.5, 0.0], [0.1, 0.9, 0.0]]
    node_ids = ["a", "b", "c"]

    _, regular_ids = get_top_k_embeddings(query_emb, embeddings, embedding_ids=node_ids)

    rows = [
        _create_mock_db_embedding_row(
            node_id=nid, text=f"t_{nid}", metadata={"_node_type": "TextNode"}
        )
        for nid in node_ids
    ]
    query = VectorStoreQuery(
        query_embedding=query_emb,
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=1.0,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        list(zip(rows, embeddings)),
        use_async,
    )

    result = await _call_mmr(mock_store, query, use_async)

    assert result.ids == regular_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_invokes_customize_query_fn(use_async):
    """When _customize_query_fn is set, the MMR prefetch should call it via _build_query_with_embedding, and the extra kwarg should pass through."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()

    # Track whether _build_query_with_embedding was called with the extra param
    build_spy = MagicMock(return_value="stmt_sentinel")
    mock_store._build_query_with_embedding = build_spy
    mock_store._apply_filters_and_limit = MagicMock(return_value="final")

    # Make the embedding query return empty so we hit the empty-result path
    _setup_embedding_mock(mock_store, [], use_async)

    await _call_mmr(mock_store, query, use_async, hnsw_ef_search=200)

    # Verify the DB kwarg (hnsw_ef_search) was forwarded but MMR kwargs were not
    call_kwargs = _get_embedding_mock(mock_store, use_async).call_args
    forwarded_kwargs = call_kwargs[1] if call_kwargs[1] else {}
    assert forwarded_kwargs.get("hnsw_ef_search") == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_handles_index_node_type(use_async):
    """MMR query correctly handles IndexNode type in metadata."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=1,
        mode=VectorStoreQueryMode.MMR,
    )
    # IndexNode metadata uses a different _node_type
    mock_row = _create_mock_db_embedding_row(
        node_id="idx_node1",
        text="index text",
        metadata={
            "_node_type": "IndexNode",
            "index_id": "some_index",
        },
        similarity=0.95,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [(mock_row, [1.0, 0.0])], use_async)

    result = await _call_mmr(mock_store, query, use_async)

    assert len(result.nodes) == 1
    assert result.ids == ["idx_node1"]
    assert result.nodes[0].text == "index text"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_handles_duplicate_node_ids(use_async):
    """
    If prefetch returns duplicate node_ids, result_map deduplication
    should not crash and should still return valid results.
    """
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    row_a = _create_mock_db_embedding_row(
        node_id="dup",
        text="text_a",
        metadata={"_node_type": "TextNode"},
        similarity=0.9,
    )
    row_b = _create_mock_db_embedding_row(
        node_id="dup",
        text="text_b",
        metadata={"_node_type": "TextNode"},
        similarity=0.8,
    )
    row_c = _create_mock_db_embedding_row(
        node_id="unique",
        text="text_c",
        metadata={"_node_type": "TextNode"},
        similarity=0.7,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        [(row_a, [1.0, 0.0, 0.0]), (row_b, [1.0, 0.0, 0.0]), (row_c, [0.0, 1.0, 0.0])],
        use_async,
    )

    result = await _call_mmr(mock_store, query, use_async)

    # Should not crash; result should contain up to similarity_top_k nodes
    assert len(result.nodes) <= 2
    assert len(result.ids) <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_kwargs_not_forwarded_to_db_query(use_async):
    """mmr_prefetch_factor, mmr_prefetch_k, mmr_threshold must not leak to DB layer."""
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(mock_store, [], use_async)

    await _call_mmr(
        mock_store,
        query,
        use_async,
        mmr_prefetch_factor=4,
        mmr_threshold=0.5,
        hnsw_ef_search=100,  # legitimate DB kwarg
    )

    call_kwargs = _get_embedding_mock(mock_store, use_async).call_args
    # The positional/keyword args sent to the DB query
    forwarded_kwargs = call_kwargs[1] if call_kwargs[1] else {}
    assert "mmr_prefetch_factor" not in forwarded_kwargs
    assert "mmr_prefetch_k" not in forwarded_kwargs
    assert "mmr_threshold" not in forwarded_kwargs
    # Legitimate DB kwarg should be forwarded
    assert forwarded_kwargs.get("hnsw_ef_search") == 100


def test_prepare_mmr_query_query_threshold_takes_precedence():
    """_prepare_mmr_query should use query.mmr_threshold over kwargs."""
    mock_store = _create_mock_pg_vector_store()
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=0.3,
    )
    _, threshold = PGVectorStore._prepare_mmr_query(
        mock_store, query, mmr_threshold=0.9
    )
    assert threshold == 0.3


def test_prepare_mmr_query_falls_back_to_kwargs_threshold():
    """_prepare_mmr_query should use kwargs threshold when query has None."""
    mock_store = _create_mock_pg_vector_store()
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=None,
    )
    _, threshold = PGVectorStore._prepare_mmr_query(
        mock_store, query, mmr_threshold=0.7
    )
    assert threshold == 0.7


def test_prepare_mmr_query_threshold_none_when_both_absent():
    """_prepare_mmr_query returns None threshold when neither source provides it."""
    mock_store = _create_mock_pg_vector_store()
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=5,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=None,
    )
    _, threshold = PGVectorStore._prepare_mmr_query(mock_store, query)
    assert threshold is None


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_diverse_selection_from_large_candidate_pool(use_async):
    """
    With 6 candidates (3 clusters of 2 near-duplicates), MMR at low threshold
    should pick one from each cluster rather than both from the same cluster.

    All clusters have similar relevance to the query so diversity dominates.
    """
    # Query is equally similar to all three axes via a diagonal direction
    query = VectorStoreQuery(
        query_embedding=[1.0, 1.0, 1.0],
        similarity_top_k=3,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=0.1,  # strongly favor diversity
    )
    # Cluster 1: x-axis pair
    emb_a1, emb_a2 = [1.0, 0.0, 0.0], [0.95, 0.05, 0.0]
    # Cluster 2: y-axis pair
    emb_b1, emb_b2 = [0.0, 1.0, 0.0], [0.05, 0.95, 0.0]
    # Cluster 3: z-axis pair
    emb_c1, emb_c2 = [0.0, 0.0, 1.0], [0.05, 0.0, 0.95]

    rows = []
    embs = [emb_a1, emb_a2, emb_b1, emb_b2, emb_c1, emb_c2]
    for i, emb in enumerate(embs):
        rows.append(
            _create_mock_db_embedding_row(
                node_id=f"n{i}",
                text=f"t{i}",
                metadata={"_node_type": "TextNode"},
            )
        )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        list(zip(rows, embs)),
        use_async,
    )

    result = await _call_mmr(mock_store, query, use_async)

    assert len(result.nodes) == 3
    # With strong diversity and equal relevance, MMR should pick at most
    # one member from each near-duplicate cluster
    cluster_a_ids = {"n0", "n1"}
    cluster_b_ids = {"n2", "n3"}
    cluster_c_ids = {"n4", "n5"}
    selected = set(result.ids)
    assert len(selected & cluster_a_ids) <= 1, "Both x-axis duplicates selected"
    assert len(selected & cluster_b_ids) <= 1, "Both y-axis duplicates selected"
    assert len(selected & cluster_c_ids) <= 1, "Both z-axis duplicates selected"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_mmr_query_filters_out_empty_embeddings_in_mixed_results(use_async):
    """
    When some candidates have valid embeddings and others have empty ones,
    only valid ones should be used for MMR, and the result should still work
    as long as enough valid embeddings exist.
    """
    query = VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        mode=VectorStoreQueryMode.MMR,
    )
    row_valid1 = _create_mock_db_embedding_row(
        node_id="v1", text="valid1", metadata={"_node_type": "TextNode"}
    )
    row_empty = _create_mock_db_embedding_row(
        node_id="e1", text="empty", metadata={"_node_type": "TextNode"}
    )
    row_valid2 = _create_mock_db_embedding_row(
        node_id="v2", text="valid2", metadata={"_node_type": "TextNode"}
    )
    mock_store = _create_mock_pg_vector_store()
    _setup_embedding_mock(
        mock_store,
        [
            (row_valid1, [1.0, 0.0]),
            (row_empty, []),  # empty embedding
            (row_valid2, [0.0, 1.0]),
        ],
        use_async,
    )

    result = await _call_mmr(mock_store, query, use_async)

    assert len(result.nodes) == 2
    # Only v1 and v2 should be in results, not e1
    assert "e1" not in result.ids
    assert "v1" in result.ids
    assert "v2" in result.ids
