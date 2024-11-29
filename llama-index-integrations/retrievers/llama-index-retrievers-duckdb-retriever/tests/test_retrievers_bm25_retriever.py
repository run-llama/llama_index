import duckdb
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.duckdb_retriever.base import (
    DuckDBRetriever,
)


def test_class():
    names_of_base_classes = [b.__name__ for b in DuckDBRetriever.__mro__]
    assert BaseRetriever.__name__ in names_of_base_classes


def test_injection(tmp_path):
    db = tmp_path / "test.db"
    with duckdb.connect(db) as conn:
        conn.sql(
            "CREATE TABLE documents (node_id VARCHAR, text VARCHAR, author VARCHAR, doc_version INTEGER);"
        )
    r = DuckDBRetriever(database_name=db)
    nodes = r.retrieve(
        "life') AS score, node_id, text FROM documents UNION SELECT '1500', '!', concat('life', version()) UNION SELECT concat('0"
    )
    assert not nodes
