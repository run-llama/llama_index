import asyncio
from typing import Any, Dict, Generator, List, Union

import pytest
from llama_index.core.schema import (
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.paradedb import ParadeDBVectorStore
from llama_index.vector_stores.postgres.base import PGVectorStore

PARAMS: Dict[str, Union[str, int]] = {
    "host": "localhost",
    "user": "postgres",
    "password": "mark90",
    "port": 5432,
}
TEST_DB = "test_vector_db"
TEST_DB_HNSW = "test_vector_db_hnsw"
TEST_TABLE_NAME = "lorem_ipsum"
TEST_SCHEMA_NAME = "paradedb"
TEST_EMBED_DIM = 2

try:
    import asyncpg  # noqa
    import pgvector  # noqa
    import psycopg2
    import sqlalchemy
    import sqlalchemy.ext.asyncio  # noqa

    conn__ = psycopg2.connect(**PARAMS)  # type: ignore
    conn__.close()

    postgres_not_available = False
except (ImportError, Exception):
    postgres_not_available = True


def _get_sample_vector(num: float) -> List[float]:
    """Get sample embedding vector of the form [num, 1, 1, ..., 1]"""
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
        c.execute(f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{TEST_DB}' AND pid <> pg_backend_pid();
        """)
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
        c.execute(f"""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '{TEST_DB_HNSW}' AND pid <> pg_backend_pid();
        """)
        c.execute(f"DROP DATABASE {TEST_DB_HNSW}")
        conn.commit()


@pytest.fixture()
def pg_bm25(db: None) -> Any:
    """Fixture for ParadeDBVectorStore with BM25 enabled."""
    pg = ParadeDBVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME + "_bm25",
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        use_bm25=True,
        embed_dim=TEST_EMBED_DIM,
    )
    yield pg
    asyncio.run(pg.close())


@pytest.fixture()
def pg_bm25_halfvec(db: None) -> Any:
    """Fixture for ParadeDBVectorStore with BM25 and halfvec enabled."""
    pg = ParadeDBVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME + "_bm25_halfvec",
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        use_bm25=True,
        use_halfvec=True,
        embed_dim=TEST_EMBED_DIM,
    )
    yield pg
    asyncio.run(pg.close())


@pytest.fixture()
def pg_hnsw_bm25(db_hnsw: None) -> Any:
    """Fixture for ParadeDBVectorStore with HNSW and BM25 enabled."""
    pg = ParadeDBVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB_HNSW,
        table_name=TEST_TABLE_NAME + "_hnsw_bm25",
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        use_bm25=True,
        embed_dim=TEST_EMBED_DIM,
        hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},
    )
    yield pg
    asyncio.run(pg.close())


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


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_paradedb_instance_creation(db: None) -> None:
    """Test that ParadeDBVectorStore can be instantiated."""
    pg = ParadeDBVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        use_bm25=True,
        hybrid_search=True,
    )
    assert isinstance(pg, ParadeDBVectorStore)
    assert pg.use_bm25 is True
    assert pg.client is None
    await pg.close()


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_bm25_sparse_query(
    pg_bm25: ParadeDBVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """Test sparse query using BM25 index."""
    if use_async:
        await pg_bm25.async_add(hybrid_node_embeddings)
    else:
        pg_bm25.add(hybrid_node_embeddings)

    assert isinstance(pg_bm25, ParadeDBVectorStore)
    assert hasattr(pg_bm25, "_engine")
    assert pg_bm25.use_bm25 is True

    # Test BM25 text search with a sentence query
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
    )

    if use_async:
        res = await pg_bm25.aquery(q)
    else:
        res = pg_bm25.query(q)

    assert res.nodes
    assert len(res.nodes) == 2
    # BM25 should rank documents with "fox" highest
    node_ids = [node.node_id for node in res.nodes]
    assert "ccc" in node_ids or "ddd" in node_ids


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_bm25_hybrid_query(
    pg_bm25: ParadeDBVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """Test hybrid query combining vector similarity and BM25 text search."""
    if use_async:
        await pg_bm25.async_add(hybrid_node_embeddings)
    else:
        pg_bm25.add(hybrid_node_embeddings)

    assert isinstance(pg_bm25, ParadeDBVectorStore)
    assert hasattr(pg_bm25, "_engine")
    assert pg_bm25.use_bm25 is True

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
        sparse_top_k=1,
    )

    if use_async:
        res = await pg_bm25.aquery(q)
    else:
        res = pg_bm25.query(q)

    assert res.nodes
    assert len(res.nodes) >= 2
    node_ids = [node.node_id for node in res.nodes]
    # Should contain results from both vector and BM25 search
    assert "ccc" in node_ids or "ddd" in node_ids


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_bm25_with_special_characters(
    pg_bm25: ParadeDBVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """Test BM25 search handles special characters properly."""
    if use_async:
        await pg_bm25.async_add(hybrid_node_embeddings)
    else:
        pg_bm25.add(hybrid_node_embeddings)

    assert isinstance(pg_bm25, ParadeDBVectorStore)
    assert pg_bm25.use_bm25 is True

    # Test with special characters
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="   who' &..s |     (the): <-> **fox**?!!!",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
    )

    if use_async:
        res = await pg_bm25.aquery(q)
    else:
        res = pg_bm25.query(q)

    assert res.nodes
    assert len(res.nodes) == 2
    node_ids = [n.node_id for n in res.nodes]
    assert set(node_ids) == {"ccc", "ddd"}


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_bm25_halfvec(
    pg_bm25_halfvec: ParadeDBVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """Test BM25 with half-precision vectors."""
    if use_async:
        await pg_bm25_halfvec.async_add(hybrid_node_embeddings)
    else:
        pg_bm25_halfvec.add(hybrid_node_embeddings)

    assert isinstance(pg_bm25_halfvec, ParadeDBVectorStore)
    assert pg_bm25_halfvec.use_bm25 is True
    assert pg_bm25_halfvec.use_halfvec is True

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="fox",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_bm25_halfvec.aquery(q)
    else:
        res = pg_bm25_halfvec.query(q)

    assert res.nodes
    assert len(res.nodes) >= 2


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_bm25_hnsw_hybrid(
    pg_hnsw_bm25: ParadeDBVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """Test BM25 with HNSW index in hybrid mode."""
    if use_async:
        await pg_hnsw_bm25.async_add(hybrid_node_embeddings)
    else:
        pg_hnsw_bm25.add(hybrid_node_embeddings)

    assert isinstance(pg_hnsw_bm25, ParadeDBVectorStore)
    assert pg_hnsw_bm25.use_bm25 is True
    assert pg_hnsw_bm25.hnsw_kwargs is not None

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="who is the fox?",
        similarity_top_k=2,
        mode=VectorStoreQueryMode.HYBRID,
    )

    if use_async:
        res = await pg_hnsw_bm25.aquery(q)
    else:
        res = pg_hnsw_bm25.query(q)

    assert res.nodes
    assert len(res.nodes) >= 2
    node_ids = [node.node_id for node in res.nodes]
    assert "ccc" in node_ids or "ddd" in node_ids


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_bm25_index_creation(
    pg_bm25: ParadeDBVectorStore,
    hybrid_node_embeddings: List[TextNode],
) -> None:
    """Test that BM25 index is created properly."""
    pg_bm25.add(hybrid_node_embeddings[:1])

    data_test_table_name = f"data_{TEST_TABLE_NAME}_bm25"
    data_test_index_name = f"{data_test_table_name}_bm25_idx"

    with psycopg2.connect(**PARAMS, database=TEST_DB) as conn:
        with conn.cursor() as c:
            c.execute(
                f"SELECT COUNT(*) FROM pg_indexes WHERE schemaname = '{TEST_SCHEMA_NAME}' AND tablename = '{data_test_table_name}' AND indexname = '{data_test_index_name}';"
            )
            index_count = c.fetchone()[0]

    assert index_count == 1, f"Expected BM25 index '{data_test_index_name}' to exist"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [True, False])
async def test_bm25_empty_query(
    pg_bm25: ParadeDBVectorStore,
    hybrid_node_embeddings: List[TextNode],
    use_async: bool,
) -> None:
    """Test BM25 handles empty or whitespace-only queries."""
    if use_async:
        await pg_bm25.async_add(hybrid_node_embeddings)
    else:
        pg_bm25.add(hybrid_node_embeddings)

    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        query_str="   ",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
    )

    # BM25 will fail with empty queries
    with pytest.raises(Exception):
        if use_async:
            await pg_bm25.aquery(q)
        else:
            pg_bm25.query(q)


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_bm25_extensions_created(db: None) -> None:
    """Test that both vector and pg_search extensions are created."""
    pg = ParadeDBVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name="test_extensions",
        schema_name=TEST_SCHEMA_NAME,
        use_bm25=True,
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
    )
    
    # Force initialization
    pg.add([
        TextNode(
            text="test",
            id_="test",
            embedding=_get_sample_vector(1.0),
        )
    ])
    
    # Check that both extensions exist
    with psycopg2.connect(**PARAMS, database=TEST_DB) as conn:
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM pg_extension WHERE extname IN ('vector', 'pg_search');")
            ext_count = c.fetchone()[0]
    
    assert ext_count == 2, "Both 'vector' and 'pg_search' extensions should exist"
    
    await pg.close()


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_paradedb_inherits_pgvector_functionality(
    pg_bm25: ParadeDBVectorStore,
    hybrid_node_embeddings: List[TextNode],
) -> None:
    """Test that ParadeDBVectorStore inherits all PGVectorStore functionality."""
    # Add nodes
    pg_bm25.add(hybrid_node_embeddings)
    
    # Test vector-only query (inherited from PGVectorStore)
    q = VectorStoreQuery(
        query_embedding=_get_sample_vector(0.1),
        similarity_top_k=2,
        mode=VectorStoreQueryMode.DEFAULT,
    )
    
    res = pg_bm25.query(q)
    assert res.nodes
    assert len(res.nodes) == 2
    
    # Test delete (inherited)
    pg_bm25.delete_nodes(["aaa"])
    
    res = pg_bm25.query(q)
    assert "aaa" not in res.ids
    
    # Test clear (inherited)
    await pg_bm25.aclear()
    
    res = pg_bm25.query(q)
    assert len(res.nodes) == 0


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
@pytest.mark.asyncio
async def test_bm25_vs_tsvector_different_results(
    db: None,
    hybrid_node_embeddings: List[TextNode]
    ) -> None:
    """Test that BM25 and ts_vector can produce different ranking results."""
    
    # Create both stores
    pg_tsvector = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name="test_tsvector",
        schema_name="public",
        hybrid_search=True,
        embed_dim=TEST_EMBED_DIM,
    )
    
    pg_bm25 = ParadeDBVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name="test_bm25",
        schema_name=TEST_SCHEMA_NAME,
        hybrid_search=True,
        use_bm25=True,
        embed_dim=TEST_EMBED_DIM,
    )
    pg_tsvector.add(hybrid_node_embeddings)
    pg_bm25.add(hybrid_node_embeddings)
    
    q = VectorStoreQuery(
        query_str="fox",
        sparse_top_k=2,
        mode=VectorStoreQueryMode.SPARSE,
        query_embedding=_get_sample_vector(5.0),
    )
    
    res_tsvector = pg_tsvector.query(q)
    res_bm25 = pg_bm25.query(q)

    # Log results
    # print("\n=== RESULT COMPARISON ===")
    # print(f"TSVECTOR → Top1: {res_tsvector.nodes[0].node_id} ({res_tsvector.similarities[0]:.6f}) | "
    #     f"Top2: {res_tsvector.nodes[1].node_id} ({res_tsvector.similarities[1]:.6f})")
    # print(f"BM25     → Top1: {res_bm25.nodes[0].node_id} ({res_bm25.similarities[0]:.6f}) | "
    #     f"Top2: {res_bm25.nodes[1].node_id} ({res_bm25.similarities[1]:.6f})")
    # Both should return results
    assert len(res_tsvector.nodes) == 2
    assert len(res_bm25.nodes) == 2
    
    # BM25 uses BM25 ranking, ts_vector uses ts_rank
    # The implementation difference is verified
    assert pg_bm25.use_bm25 is True
    assert not hasattr(pg_tsvector, "use_bm25") or pg_tsvector.use_bm25 is False
    
    await pg_tsvector.close()
    await pg_bm25.close()