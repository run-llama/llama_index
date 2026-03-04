"""VectorStore integration tests for Azure Database for PostgreSQL using LlamaIndex."""

import re
from contextlib import nullcontext
from typing import Any

import pytest
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import PositiveInt

from llama_index.core.schema import (
    TextNode,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.azure_postgres import AzurePGVectorStore
from llama_index.vector_stores.azure_postgres.common import DiskANN

from .conftest import Table

# SQL constants to be used in tests
_GET_TABLE_COLUMNS_AND_TYPES = sql.SQL(
    """
      select  a.attname as column_name,
              format_type(a.atttypid, a.atttypmod) as column_type
        from  pg_attribute a
              join pg_class c on a.attrelid = c.oid
              join pg_namespace n on c.relnamespace = n.oid
       where  a.attnum > 0
              and not a.attisdropped
              and n.nspname = %(schema_name)s
              and c.relname = %(table_name)s
    order by  a.attnum asc
    """
)


# Utility/assertion functions to be used in tests
def verify_table_created(table: Table, resultset: list[dict[str, Any]]) -> None:
    """Verify that the table has been created with the correct columns and types.

    :param table: Expected table to be created
    :type table: Table
    :param resultset: Actual result set from the database
    :type resultset: list[dict[str, Any]]
    """
    # Verify that the ID column has been created correctly
    result = next((r for r in resultset if r["column_name"] == table.id_column), None)
    assert result is not None, "ID column was not created in the table."
    assert result["column_type"] == "uuid", "ID column type is incorrect."

    # Verify that the content column has been created correctly
    result = next(
        (r for r in resultset if r["column_name"] == table.content_column), None
    )
    assert result is not None, "Content column was not created in the table."
    assert result["column_type"] == "text", "Content column type is incorrect."

    # Verify that the embedding column has been created correctly
    result = next(
        (r for r in resultset if r["column_name"] == table.embedding_column), None
    )
    assert result is not None, "Embedding column was not created in the table."
    embedding_column_type = result["column_type"]
    pattern = re.compile(r"(?P<type>\w+)(?:\((?P<dim>\d+)\))?")
    m = pattern.match(embedding_column_type if embedding_column_type else "")
    parsed_type: str | None = m.group("type") if m else None
    parsed_dim: PositiveInt | None = (
        PositiveInt(m.group("dim")) if m and m.group("dim") else None
    )
    assert parsed_type == table.embedding_type.value, (
        "Embedding column type is incorrect."
    )
    assert parsed_dim == table.embedding_dimension, (
        "Embedding column dimension is incorrect."
    )

    # Verify that metadata column have been created correctly
    result = next(
        (r for r in resultset if r["column_name"] == table.metadata_column), None
    )
    assert result is not None, (
        f"Metadata column '{table.metadata_column}' was not created in the table."
    )


class TestAzurePGVectorStore:
    """Integration tests for the AzurePGVectorStore implementation.

    Covers table creation, initialization via parameters, CRUD operations,
    and similarity queries against seeded data in the test database.
    """

    def test_table_creation_success(
        self, vectorstore: AzurePGVectorStore, table: Table
    ):
        """Verify the database table is created with the expected columns."""
        with (
            vectorstore.connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                _GET_TABLE_COLUMNS_AND_TYPES,
                {
                    "schema_name": table.schema_name,
                    "table_name": table.table_name,
                },
            )
            resultset = cursor.fetchall()
        verify_table_created(table, resultset)

    def test_vectorstore_initialization_from_params(
        self,
        connection_pool: ConnectionPool,
        schema: str,
    ):
        """Create a store using class factory `from_params` and assert type."""
        table_name = "vs_init_from_params"
        embedding_dimension = 3

        diskann = DiskANN(
            op_class="vector_cosine_ops",
            max_neighbors=32,
            l_value_ib=100,
            l_value_is=100,
        )

        vectorstore = AzurePGVectorStore.from_params(
            connection_pool=connection_pool,
            schema_name=schema,
            table_name=table_name,
            embed_dim=embedding_dimension,
            embedding_index=diskann,
        )
        assert isinstance(vectorstore, AzurePGVectorStore)

    def test_get_nodes(
        self,
        vectorstore: AzurePGVectorStore,
    ):
        """Retrieve all nodes and assert expected seeded node count."""
        in_nodes = vectorstore.get_nodes()
        assert len(in_nodes) == 4, "Retrieved node count does not match expected"

    @pytest.mark.parametrize(
        ["node_tuple", "expected"],
        [
            ("node-success", nullcontext(AzurePGVectorStore)),
            ("node-not-found", pytest.raises(IndexError)),
        ],
        indirect=["node_tuple"],
        ids=[
            "success",
            "not-found",
        ],
    )
    def test_get_nodes_with_ids(
        self,
        vectorstore: AzurePGVectorStore,
        node_tuple: tuple[TextNode, str | None],
        expected: nullcontext[AzurePGVectorStore] | pytest.RaisesExc,
    ):
        """Retrieve nodes by ID and validate returned node matches expected."""
        node, expected_node_id = node_tuple
        in_nodes = vectorstore.get_nodes([node.node_id])

        with expected:
            assert expected_node_id == in_nodes[0].node_id, (
                "Retrieved node ID does not match expected"
            )

    @pytest.mark.parametrize(
        ["node_tuple", "expected"],
        [
            ("node-success", nullcontext(AzurePGVectorStore)),
            # ("node-failure", pytest.raises(AssertionError)),
        ],
        indirect=["node_tuple"],
        ids=[
            "success",
            # "failure",
        ],
    )
    def test_add(
        self,
        vectorstore: AzurePGVectorStore,
        node_tuple: tuple[TextNode, str | None],
        expected: nullcontext[AzurePGVectorStore] | pytest.RaisesExc,
    ):
        """Add a node to the store and assert the returned ID matches."""
        node, expected_node_id = node_tuple
        with expected:
            assert node.node_id is not None, "Node ID must be provided for this test"
            returned_ids = vectorstore.add([node])
            assert returned_ids[0] == expected_node_id, "Inserted text IDs do not match"

    @pytest.mark.parametrize(
        ["doc_id"],
        [
            ("1",),
            ("10",),
        ],
        ids=["existing", "non-existing"],
    )
    def test_delete(
        self,
        vectorstore: AzurePGVectorStore,
        doc_id: str,
    ):
        """Delete a node by reference doc id and assert it was removed."""
        vectorstore.delete(doc_id)

        with (
            vectorstore.connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                sql.SQL(
                    """
                    select  {metadata} ->> 'doc_id' as doc_id
                      from  {table_name}
                    """
                ).format(
                    metadata=sql.Identifier(vectorstore.metadata_columns),
                    table_name=sql.Identifier(
                        vectorstore.schema_name, vectorstore.table_name
                    ),
                )
            )
            resultset = cursor.fetchall()

        remaining_set = set(str(r["doc_id"]) for r in resultset)

        assert doc_id not in remaining_set, (
            "Deleted document IDs should not exist in the remaining set"
        )

    @pytest.mark.parametrize(
        ["node_tuple"],
        [
            ("node-success",),
            ("node-not-found",),
        ],
        indirect=["node_tuple"],
        ids=[
            "success",
            "not-found",
        ],
    )
    def test_delete_nodes(
        self,
        vectorstore: AzurePGVectorStore,
        node_tuple: tuple[TextNode, str | None],
    ):
        """Delete a list of node IDs and assert they are removed from the table."""
        node, expected_node_id = node_tuple
        vectorstore.delete_nodes([node.node_id])

        with (
            vectorstore.connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                sql.SQL(
                    """
                    select  {id_column} as node_id
                      from  {table_name}
                    """
                ).format(
                    id_column=sql.Identifier(vectorstore.id_column),
                    table_name=sql.Identifier(
                        vectorstore.schema_name, vectorstore.table_name
                    ),
                )
            )
            resultset = cursor.fetchall()

        remaining_set = set(str(r["node_id"]) for r in resultset)

        assert expected_node_id not in remaining_set, (
            "Deleted document IDs should not exist in the remaining set"
        )

    def test_clear(
        self,
        vectorstore: AzurePGVectorStore,
    ):
        """Clear all nodes from the underlying table and verify none remain."""
        vectorstore.clear()

        with (
            vectorstore.connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                sql.SQL(
                    """
                    select  {id_column} as node_id
                      from  {table_name}
                    """
                ).format(
                    id_column=sql.Identifier(vectorstore.id_column),
                    table_name=sql.Identifier(
                        vectorstore.schema_name, vectorstore.table_name
                    ),
                )
            )
            resultset = cursor.fetchall()

        remaining_set = set(str(r["node_id"]) for r in resultset)

        assert not remaining_set, "All document IDs should have been deleted"

    @pytest.mark.parametrize(
        ["query", "embedding", "k", "filters", "mode"],
        [
            ("query about cats", [0.99] * 1536, 2, None, None),
            ("query about cats", [0.99] * 1536, 2, None, "hybrid"),
            ("query about animals", [0.5] * 1536, 3, None, None),
            ("query about cats", [0.99] * 1536, 2, "filter1", None),
            ("query about cats", [0.99] * 1536, 2, "filter2", None),
        ],
        indirect=["filters"],
        ids=[
            "search-cats",
            "search-cats-hybrid",
            "search-animals",
            "search-cats-filtered",
            "search-cats-multifiltered",
        ],
    )
    def test_query(
        self,
        vectorstore: AzurePGVectorStore,
        query: str,
        embedding: list[float],
        k: int,
        filters: MetadataFilters | None,
        mode: str | None,
    ):
        """Run a similarity query and assert returned documents match expectations.

        Tests multiple query types (cats/animals) and optional metadata
        filters to ensure the vector search returns relevant documents and
        that filtering works as intended.
        """
        vsquery = VectorStoreQuery(
            query_str=query,
            query_embedding=embedding,
            similarity_top_k=k,
            filters=filters,
            mode=(
                VectorStoreQueryMode.HYBRID
                if mode == "hybrid"
                else VectorStoreQueryMode.DEFAULT
            ),
        )
        results = vectorstore.query(query=vsquery)

        results = results.nodes
        contents = [row.get_content() for row in results]

        if ("cats" in query) or ("animals" in query):
            assert len(results) == k, f"Expected {k} results"
            assert any("cats" in c for c in contents) or any(
                "tigers" in c for c in contents
            ), (
                f"Expected 'cats' or 'tigers' in retrieved documents' contents for query: {query}"
            )

        if "cats" in query:
            assert all("dogs" not in c for c in contents), (
                f"Expected 'dogs' not to be in retrieved documents' contents for query: {query}"
            )
        elif "animals" in query:
            assert any("dogs" in c for c in contents), (
                f"Expected 'dogs' to be in retrieved documents' contents for query: {query}"
            )

        assert all("plants" not in c for c in contents), (
            f"Expected 'plants' not to be in retrieved documents' contents for query: {query}"
        )
