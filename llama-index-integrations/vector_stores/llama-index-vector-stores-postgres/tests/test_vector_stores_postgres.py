from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
)
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import column


def test_class():
    names_of_base_classes = [b.__name__ for b in PGVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_metadata_filter_key_is_parameterized() -> None:
    store = PGVectorStore.model_construct()
    store._table_class = type(
        "Table", (), {"metadata_": column("metadata_", JSONB)}
    )()
    malicious_key = "name' OR '1'='1"

    clause = store._build_filter_clause(
        MetadataFilter(key=malicious_key, value="alice", operator=FilterOperator.EQ)
    )
    compiled = str(clause.compile(dialect=postgresql.dialect()))

    assert malicious_key not in compiled
    assert "metadata_" in compiled
