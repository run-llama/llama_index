"""Pytest fixtures and Pydantic models used for Azure PostgreSQL vector store integration tests."""

from collections.abc import Generator
from typing import Any

import pytest
from psycopg import sql
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, PositiveInt

from llama_index.core.schema import Node
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.azure_postgres import (
    AzurePGVectorStore,
)
from llama_index.vector_stores.azure_postgres.common import (
    Algorithm,
    DiskANN,
    VectorType,
)

_FIXTURE_PARAMS_TABLE: dict[str, Any] = {
    "scope": "class",
    "params": [
        {
            "existing": True,
            "table_name": "existing_table",
            "id_column": "id",
            "content_column": "content",
            "embedding_column": "embedding",
            "embedding_type": VectorType.vector,
            "embedding_dimension": 1_536,
            "embedding_index": None,
            "metadata_column": "metadata",
        },
    ],
    "ids": [
        # "non-existing-table-metadata-str",
        "existing-table-metadata-str",
    ],
}


@pytest.fixture(
    params=[
        "node-success",
        "node-not-found",
    ]
)
def node_tuple(
    request: pytest.FixtureRequest,
) -> tuple[Node, str | None]:
    """Fixture to provide a parametrized node configuration for tests.

    This fixture yields a `Node` model with the specified parameters.

    :param request: The pytest request object providing parametrization.
    :type request: pytest.FixtureRequest
    :return: A generator yielding a `Node` configuration.
    :rtype: Generator[Node, Any, None]
    """
    assert isinstance(request.param, str), "Request param must be a str"

    if request.param == "node-success":
        n = Node()
        n.node_id = "00000000-0000-0000-0000-000000000001"
        n.set_content("Text 1 about cats")
        n.embedding = [1.0] * 1536
        n.metadata = {"metadata_column1": "text1", "metadata_column2": 1}
    elif request.param == "node-not-found":
        n = Node()
        n.node_id = "00000000-0000-0000-0000-000000000010"
        n.set_content("Text 10 about cats")
        n.embedding = [10.0] * 1536
        n.metadata = {"metadata_column1": "text1", "metadata_column2": 10}
    else:
        raise ValueError(f"Unknown node parameter: {request.param}")

    return (n, n.node_id)


class Table(BaseModel):
    """Table configuration for test parameterization.

    :param existing: Whether the table should be created before running a test.
    :param schema_name: Schema where the table resides.
    :param table_name: Name of the table.
    :param id_column: Primary key column name (uuid).
    :param content_column: Text content column name.
    :param embedding_column: Vector/embedding column name.
    :param embedding_type: Embedding type (e.g., "vector").
    :param embedding_dimension: Embedding dimension length.
    :param metadata_column: List of metadata column names or (name, type) tuples.
    """

    existing: bool
    schema_name: str
    table_name: str
    id_column: str
    content_column: str
    embedding_column: str
    embedding_type: VectorType
    embedding_dimension: PositiveInt
    embedding_index: Algorithm | None
    metadata_column: str


@pytest.fixture(**_FIXTURE_PARAMS_TABLE)
def table(
    connection_pool: ConnectionPool,
    schema: str,
    request: pytest.FixtureRequest,
) -> Generator[Table, Any, None]:
    """Fixture to provide a parametrized table configuration for synchronous tests.

    This fixture yields a `Table` model with normalized metadata columns. When
    the parameter `existing` is `True`, it creates the table in the provided
    schema before yielding and drops it after the test class completes.

    :param connection_pool: The synchronous connection pool to use for DDL.
    :type connection_pool: ConnectionPool
    :param schema: The schema name where the table should be created.
    :type schema: str
    :param request: The pytest request object providing parametrization.
    :type request: pytest.FixtureRequest
    :return: A generator yielding a `Table` configuration.
    :rtype: Generator[Table, Any, None]
    """
    assert isinstance(request.param, dict), "Request param must be a dictionary"

    table = Table(
        existing=request.param.get("existing", None),
        schema_name=schema,
        table_name=request.param.get("table_name", "llamaindex"),
        id_column=request.param.get("id_column", "id"),
        content_column=request.param.get("content_column", "content"),
        embedding_column=request.param.get("embedding_column", "embedding"),
        embedding_type=request.param.get("embedding_type", "vector"),
        embedding_dimension=request.param.get("embedding_dimension", 1_536),
        embedding_index=request.param.get("embedding_index", None),
        metadata_column=request.param.get("metadata_column", "metadata"),
    )

    if table.existing:
        with connection_pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    create table {table_name} (
                        {id_column} uuid primary key,
                        {content_column} text,
                        {embedding_column} {embedding_type}({embedding_dimension}),
                        {metadata_column} jsonb
                    )
                    """
                ).format(
                    table_name=sql.Identifier(schema, table.table_name),
                    id_column=sql.Identifier(table.id_column),
                    content_column=sql.Identifier(table.content_column),
                    embedding_column=sql.Identifier(table.embedding_column),
                    embedding_type=sql.Identifier(table.embedding_type),
                    embedding_dimension=sql.Literal(table.embedding_dimension),
                    metadata_column=sql.Identifier(table.metadata_column),
                )
            )

    yield table

    with connection_pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("drop table {table} cascade").format(
                table=sql.Identifier(schema, table.table_name)
            )
        )


@pytest.fixture(
    params=[
        "filter1",
        "filter2",
    ]
)
def filters(
    request: pytest.FixtureRequest,
) -> MetadataFilters | None:
    """Define filters for various queries."""
    if request.param == "filter1":
        vsfilters = MetadataFilters(
            filters=[MetadataFilter(key="metadata_column2", value="3", operator="!=")],
            condition="and",
        )
    elif request.param == "filter2":
        vsfilters = MetadataFilters(
            filters=[
                MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="metadata_column1", value="not-text", operator="!="
                        ),
                    ],
                    condition="or",
                ),
                MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="metadata_column2", value="3", operator="!="
                        ),
                    ],
                    condition="and",
                ),
            ],
            condition="and",
        )
    else:
        return None
    return vsfilters


@pytest.fixture
def vectorstore(connection_pool: ConnectionPool, table: Table) -> AzurePGVectorStore:
    """Define vectorstore with DiskANN."""
    diskann = DiskANN(
        op_class="vector_cosine_ops", max_neighbors=32, l_value_ib=100, l_value_is=100
    )
    vector_store = AzurePGVectorStore.from_params(
        connection_pool=connection_pool,
        schema_name=table.schema_name,
        table_name=table.table_name,
        embed_dim=table.embedding_dimension,
        embedding_index=diskann,
    )

    # add several documents with deterministic embeddings for testing similarity
    dim = int(table.embedding_dimension)

    nodes = []

    n1 = Node()
    n1.node_id = "00000000-0000-0000-0000-000000000001"
    n1.set_content("Text 1 about cats")
    n1.embedding = [1.0] * dim
    n1.metadata = {"metadata_column1": "text1", "metadata_column2": 1}
    nodes.append(n1)

    n2 = Node()
    n2.node_id = "00000000-0000-0000-0000-000000000002"
    n2.set_content("Text 2 about tigers")
    # tigers should be close to cats
    n2.embedding = [0.95] * dim
    n2.metadata = {"metadata_column1": "text2", "metadata_column2": 2}
    nodes.append(n2)

    n3 = Node()
    n3.node_id = "00000000-0000-0000-0000-000000000003"
    n3.set_content("Text 3 about dogs")
    n3.embedding = [0.3] * dim
    n3.metadata = {"metadata_column1": "text3", "metadata_column2": 3}
    nodes.append(n3)

    n4 = Node()
    n4.node_id = "00000000-0000-0000-0000-000000000004"
    n4.set_content("Text 4 about plants")
    n4.embedding = [-1.0] * dim
    n4.metadata = {"metadata_column1": "text4", "metadata_column2": 4}
    nodes.append(n4)

    vector_store.add(nodes)

    return vector_store
