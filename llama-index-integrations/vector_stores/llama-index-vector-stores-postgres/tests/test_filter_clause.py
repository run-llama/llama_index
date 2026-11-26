from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter
from llama_index.vector_stores.postgres import PGVectorStore


def _store() -> PGVectorStore:
    return PGVectorStore(
        connection_string="postgresql://unused:unused@localhost/unused",
        async_connection_string="postgresql+asyncpg://unused:unused@localhost/unused",
        table_name="test_filter_clause",
        embed_dim=2,
        perform_setup=False,
    )


def test_numeric_in_filter_uses_same_comparison_as_eq() -> None:
    store = _store()

    eq_clause = store._build_filter_clause(
        MetadataFilter(key="v", value=1, operator=FilterOperator.EQ)
    )
    in_clause = store._build_filter_clause(
        MetadataFilter(key="v", value=[1], operator=FilterOperator.IN)
    )
    nin_clause = store._build_filter_clause(
        MetadataFilter(key="v", value=[1], operator=FilterOperator.NIN)
    )

    assert str(eq_clause) == "(metadata_->>'v')::float = 1.0"
    assert str(in_clause) == "(metadata_->>'v')::float IN (1.0)"
    assert str(nin_clause) == "(metadata_->>'v')::float NOT IN (1.0)"


def test_string_in_and_nin_filters_keep_string_comparison() -> None:
    store = _store()

    in_clause = store._build_filter_clause(
        MetadataFilter(key="v", value=["one"], operator=FilterOperator.IN)
    )
    nin_clause = store._build_filter_clause(
        MetadataFilter(key="v", value=["one"], operator=FilterOperator.NIN)
    )

    assert str(in_clause) == "metadata_->>'v' IN ('one')"
    assert str(nin_clause) == "metadata_->>'v' NOT IN ('one')"
