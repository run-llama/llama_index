from typing import List, Any, Dict, Union, Generator

import pytest

from llama_index.data_structs.node import DocumentRelationship, Node
from llama_index.vector_stores import PGVectorStore
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStoreQuery

# from testing find install here https://github.com/pgvector/pgvector#installation-notes


PARAMS: Dict[str, Union[str, int]] = dict(
    host="localhost", user="postgres", password="", port=5432
)
TEST_DB = "test_vector_db"
TEST_TABLE_NAME = "lorem_ipsum"


def connection_check() -> None:
    import sqlalchemy  # noqa: F401
    import pgvector  # noqa: F401
    import psycopg2  # noqa: F401

    conn = psycopg2.connect(**PARAMS)  # type: ignore
    conn.close()


try:

    import psycopg2

    connection_check()
    postgres_not_available = False
except (ImportError, Exception):
    postgres_not_available = True


@pytest.fixture(scope="session")
def conn() -> Any:
    import psycopg2

    conn_ = psycopg2.connect(**PARAMS)  # type: ignore
    return conn_


@pytest.fixture(scope="session")
def db(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute("CREATE DATABASE vector_db")
        conn.commit()
    yield
    with conn.cursor() as c:
        c.execute("DROP DATABASE vector_db")
        conn.commit()


@pytest.fixture(scope="session")
def node_embeddings() -> List[NodeWithEmbedding]:
    return [
        NodeWithEmbedding(
            embedding=[1.0] * 1536,
            node=Node(
                text="lorem ipsum",
                doc_id="aaa",
                relationships={DocumentRelationship.SOURCE: "test-0"},
            ),
        ),
        NodeWithEmbedding(
            embedding=[0.0] * 1536,
            node=Node(
                text="lorem ipsum",
                doc_id="bbb",
                relationships={DocumentRelationship.SOURCE: "test-1"},
            ),
        ),
    ]


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
def test_instance_creation(db: None) -> None:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
    )
    assert isinstance(pg, PGVectorStore)


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
def test_add_to_db_and_query(
    db: None, node_embeddings: List[NodeWithEmbedding]
) -> None:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
    )
    pg.add(node_embeddings)
    assert isinstance(pg, PGVectorStore)
    q = VectorStoreQuery(query_embedding=[1] * 1536, similarity_top_k=1)
    res = pg.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].doc_id == "aaa"


@pytest.mark.skipif(postgres_not_available, reason="postgres db is not available")
def test_query(db: None, node_embeddings: List[NodeWithEmbedding]) -> None:
    pg = PGVectorStore.from_params(
        **PARAMS,  # type: ignore
        database=TEST_DB,
        table_name=TEST_TABLE_NAME,
    )
    pg.add(node_embeddings)
    assert isinstance(pg, PGVectorStore)
