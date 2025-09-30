"""Base VectorStore integration for Azure Database for PostgreSQL."""

import logging
import re
import sys
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
from pgvector.psycopg import (  # type: ignore[import-untyped]
    register_vector,
    register_vector_async,
)
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from ._connection import AzurePGConnectionPool
from ._shared import (
    HNSW,
    Algorithm,
    DiskANN,
    IVFFlat,
    VectorOpClass,
    VectorType,
    run_coroutine_in_sync,
)
from .aio._connection import AsyncAzurePGConnectionPool

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

_logger = logging.getLogger(__name__)


def metadata_filters_to_sql(filters: Optional[MetadataFilters]) -> sql.SQL:
    """Convert LlamaIndex MetadataFilters to a SQL WHERE clause.

    Args:
        filters: Optional MetadataFilters object.

    Returns:
        sql.SQL: SQL WHERE clause representing the filters.
    """
    if not filters or not filters.filters:
        return sql.SQL("TRUE")

    def _filter_to_sql(filter_item: Union[MetadataFilter, MetadataFilters]) -> sql.SQL:
        """Recursively convert MetadataFilter or MetadataFilters to SQL."""
        if isinstance(filter_item, MetadataFilters):
            # Handle nested MetadataFilters
            if not filter_item.filters:
                return sql.SQL("TRUE")

            filter_sqls = [_filter_to_sql(f) for f in filter_item.filters]
            if filter_item.condition.lower() == "and":
                return sql.SQL("({})").format(sql.SQL(" AND ").join(filter_sqls))
            elif filter_item.condition.lower() == "or":
                return sql.SQL("({})").format(sql.SQL(" OR ").join(filter_sqls))
            else:  # NOT
                if len(filter_sqls) == 1:
                    return sql.SQL("NOT ({})").format(filter_sqls[0])
                else:
                    # For multiple filters with NOT, apply NOT to the AND of all filters
                    return sql.SQL("NOT ({})").format(
                        sql.SQL(" AND ").join(filter_sqls)
                    )

        elif isinstance(filter_item, MetadataFilter):
            # Handle individual MetadataFilter
            key = filter_item.key
            value = filter_item.value
            operator = filter_item.operator

            # Use JSONB path for metadata column
            column_ref = sql.SQL("metadata ->> {}").format(sql.Literal(key))

            if operator == FilterOperator.EQ:
                return sql.SQL("{} = {}").format(column_ref, sql.Literal(str(value)))
            elif operator == FilterOperator.NE:
                return sql.SQL("{} != {}").format(column_ref, sql.Literal(str(value)))
            elif operator == FilterOperator.GT:
                return sql.SQL("({}) > {}").format(column_ref, sql.Literal(value))
            elif operator == FilterOperator.LT:
                return sql.SQL("({}) < {}").format(column_ref, sql.Literal(value))
            elif operator == FilterOperator.GTE:
                return sql.SQL("({}) >= {}").format(column_ref, sql.Literal(value))
            elif operator == FilterOperator.LTE:
                return sql.SQL("({}) <= {}").format(column_ref, sql.Literal(value))
            elif operator == FilterOperator.IN:
                if isinstance(value, list):
                    values = sql.SQL(", ").join([sql.Literal(str(v)) for v in value])
                    return sql.SQL("{} IN ({})").format(column_ref, values)
                else:
                    return sql.SQL("{} = {}").format(
                        column_ref, sql.Literal(str(value))
                    )
            elif operator == FilterOperator.NIN:
                if isinstance(value, list):
                    values = sql.SQL(", ").join([sql.Literal(str(v)) for v in value])
                    return sql.SQL("{} NOT IN ({})").format(column_ref, values)
                else:
                    return sql.SQL("{} != {}").format(
                        column_ref, sql.Literal(str(value))
                    )
            elif operator == FilterOperator.CONTAINS:
                # For JSONB array contains
                return sql.SQL("metadata -> {} ? {}").format(
                    sql.Literal(key), sql.Literal(str(value))
                )
            elif operator == FilterOperator.TEXT_MATCH:
                return sql.SQL("{} LIKE {}").format(
                    column_ref, sql.Literal(f"%{value}%")
                )
            elif operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
                return sql.SQL("{} ILIKE {}").format(
                    column_ref, sql.Literal(f"%{value}%")
                )
            elif operator == FilterOperator.IS_EMPTY:
                return sql.SQL("({} IS NULL OR {} = '')").format(column_ref, column_ref)
            else:
                # Default to equality for unsupported operators
                return sql.SQL("{} = {}").format(column_ref, sql.Literal(str(value)))

        return sql.SQL("TRUE")

    filter_sqls = [_filter_to_sql(f) for f in filters.filters]

    if filters.condition.lower() == "and":
        return sql.SQL(" AND ").join(filter_sqls)
    elif filters.condition.lower() == "or":
        return sql.SQL(" OR ").join(filter_sqls)
    else:  # NOT
        if len(filter_sqls) == 1:
            return sql.SQL("NOT ({})").format(filter_sqls[0])
        else:
            return sql.SQL("NOT ({})").format(sql.SQL(" AND ").join(filter_sqls))


def _table_row_to_node(row: dict[str, Any]) -> BaseNode:
    """Convert a table row dictionary to a BaseNode object."""
    metadata = row.get("metadata")
    if metadata is None:
        raise ValueError("Metadata not found in row data.")

    node = metadata_dict_to_node(metadata, text=row.get("content"))
    # Convert UUID to string if needed
    node_id = row.get("id")
    if node_id is not None:
        node.node_id = str(node_id)
    embedding = row.get("embedding")

    if isinstance(embedding, str):
        embedding = row.get("embedding").strip("[]").split(",")
        node.embedding = list(map(float, embedding))
    elif embedding is not None:
        node.embedding = embedding
    else:
        raise ValueError("Missing embedding value")

    return node


class BaseAzurePGVectorStore(BaseModel):
    """Base Pydantic model for an Azure PostgreSQL-backed vector store.

    This class encapsulates configuration (connection pool, table/column
    names, embedding type/dimension, index configuration and metadata
    column) and performs runtime verification that the target table
    exists with expected columns and index configuration. If the table
    does not exist, ``verify_and_init_store`` will create it.
    """

    connection_pool: AzurePGConnectionPool
    schema_name: str = "public"
    table_name: str = "vector_store"
    id_column: str = "id"
    content_column: str = "content"
    embedding_column: str = "embedding"
    embedding_type: VectorType | None = None
    embedding_dimension: PositiveInt | None = None
    embedding_index: Algorithm | None = None
    metadata_column: str | None = "metadata"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow arbitrary types like Embeddings and AzurePGConnectionPool
    )

    @model_validator(mode="after")
    def verify_and_init_store(self) -> Self:
        """Validate the store configuration and initialize DB schema and index.

        This validator runs after Pydantic model initialization. It queries
        the database to detect an existing table and its columns/indexes,
        performs type and dimension checks for the embedding column, and
        sets inferred properties (like embedding_type and embedding_dimension)
        when they are not explicitly provided. If the table does not exist,
        it will create the table with sensible defaults.

        Returns:
            Self: The same model instance, possibly updated with inferred values.
        """
        # verify that metadata_column is not empty if provided
        if self.metadata_column is not None and len(self.metadata_column) == 0:
            raise ValueError("'metadata_column' cannot be empty if provided.")

        _logger.debug(
            "checking if table '%s.%s' exists with the required columns",
            self.schema_name,
            self.table_name,
        )

        with (
            self.connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                sql.SQL(
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
                ),
                {"schema_name": self.schema_name, "table_name": self.table_name},
            )
            resultset = cursor.fetchall()
            existing_columns: dict[str, str] = {
                row["column_name"]: row["column_type"] for row in resultset
            }

        # if table exists, verify that required columns exist and have correct types
        if len(existing_columns) > 0:
            _logger.debug(
                "table '%s.%s' exists with the following column mapping: %s",
                self.schema_name,
                self.table_name,
                existing_columns,
            )

            id_column_type = existing_columns.get(self.id_column)
            if id_column_type != "uuid":
                raise ValueError(
                    f"Table '{self.schema_name}.{self.table_name}' must have a column '{self.id_column}' of type 'uuid'."
                )

            content_column_type = existing_columns.get(self.content_column)
            if content_column_type is None or (
                content_column_type != "text"
                and not content_column_type.startswith("varchar")
            ):
                raise ValueError(
                    f"Table '{self.schema_name}.{self.table_name}' must have a column '{self.content_column}' of type 'text' or 'varchar'."
                )

            embedding_column_type = existing_columns.get(self.embedding_column)
            pattern = re.compile(r"(?P<type>\w+)(?:\((?P<dim>\d+)\))?")
            m = pattern.match(embedding_column_type if embedding_column_type else "")
            parsed_type: str | None = m.group("type") if m else None
            parsed_dim: PositiveInt | None = (
                PositiveInt(m.group("dim")) if m and m.group("dim") else None
            )

            vector_types = [t.value for t in VectorType.__members__.values()]
            if parsed_type not in vector_types:
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' must be one of the following types: {vector_types}."
                )
            elif (
                self.embedding_type is not None
                and parsed_type != self.embedding_type.value
            ):
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' has type '{parsed_type}', but the specified embedding_type is '{self.embedding_type.value}'. They must match."
                )
            elif self.embedding_type is None:
                _logger.info(
                    "embedding_type is not specified, but the column '%s' in table '%s.%s' has type '%s'. Overriding embedding_type accordingly.",
                    self.embedding_column,
                    self.schema_name,
                    self.table_name,
                    parsed_type,
                )
                self.embedding_type = VectorType(parsed_type)

            if parsed_dim is not None and self.embedding_dimension is None:
                _logger.info(
                    "embedding_dimension is not specified, but the column '%s' in table '%s.%s' has a dimension of %d. Overriding embedding_dimension accordingly.",
                    self.embedding_column,
                    self.schema_name,
                    self.table_name,
                    parsed_dim,
                )
                self.embedding_dimension = parsed_dim
            elif (
                parsed_dim is not None
                and self.embedding_dimension is not None
                and parsed_dim != self.embedding_dimension
            ):
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' has a dimension of {parsed_dim}, but the specified embedding_dimension is {self.embedding_dimension}. They must match."
                )

            if self.metadata_column is not None:
                existing_type = existing_columns.get(self.metadata_column)
                if existing_type is None:
                    raise ValueError(
                        f"Column '{self.metadata_column}' does not exist in table '{self.schema_name}.{self.table_name}'."
                    )

            with (
                self.connection_pool.connection() as conn,
                conn.cursor(row_factory=dict_row) as cursor,
            ):
                _logger.debug(
                    "checking if table '%s.%s' has a vector index on column '%s'",
                    self.schema_name,
                    self.table_name,
                    self.embedding_column,
                )
                cursor.execute(
                    sql.SQL(
                        """
                        with cte as (
                          select  n.nspname as schema_name,
                                  ct.relname as table_name,
                                  ci.relname as index_name,
                                  a.amname as index_type,
                                  pg_get_indexdef(
                                    ci.oid, -- index OID
                                    generate_series(1, array_length(ii.indkey, 1)), -- column no
                                    true -- pretty print
                                  ) as index_column,
                                  o.opcname as index_opclass,
                                  ci.reloptions as index_opts
                            from  pg_class ci
                                  join pg_index ii on ii.indexrelid = ci.oid
                                  join pg_am a on a.oid = ci.relam
                                  join pg_class ct on ct.oid = ii.indrelid
                                  join pg_namespace n on n.oid = ci.relnamespace
                                  join pg_opclass o on o.oid = any(ii.indclass)
                           where  ci.relkind = 'i'
                                  and ct.relkind = 'r'
                                  and ii.indisvalid
                                  and ii.indisready
                        ) select  schema_name, table_name, index_name, index_type,
                                  index_column, index_opclass, index_opts
                            from  cte
                           where  schema_name = %(schema_name)s
                                  and table_name = %(table_name)s
                                  and index_column like %(embedding_column)s
                                  and (
                                      index_opclass like '%%vector%%'
                                      or index_opclass like '%%halfvec%%'
                                      or index_opclass like '%%sparsevec%%'
                                      or index_opclass like '%%bit%%'
                                  )
                        order by  schema_name, table_name, index_name
                        """
                    ),
                    {
                        "schema_name": self.schema_name,
                        "table_name": self.table_name,
                        "embedding_column": f"%{self.embedding_column}%",
                    },
                )
                resultset = cursor.fetchall()

            if len(resultset) > 0:
                _logger.debug(
                    "table '%s.%s' has %d vector index(es): %s",
                    self.schema_name,
                    self.table_name,
                    len(resultset),
                    resultset,
                )

                if self.embedding_index is None:
                    _logger.info(
                        "embedding_index is not specified, using the first found index: %s",
                        resultset[0],
                    )

                    index_type = resultset[0]["index_type"]
                    index_opclass = VectorOpClass(resultset[0]["index_opclass"])
                    index_opts = {
                        opts.split("=")[0]: opts.split("=")[1]
                        for opts in resultset[0]["index_opts"]
                    }

                    index = (
                        DiskANN(op_class=index_opclass, **index_opts)
                        if index_type == "diskann"
                        else (
                            HNSW(op_class=index_opclass, **index_opts)
                            if index_type == "hnsw"
                            else IVFFlat(op_class=index_opclass, **index_opts)
                        )
                    )

                    self.embedding_index = index
                else:
                    _logger.info(
                        "embedding_index is specified as '%s'; will try to find a matching index.",
                        self.embedding_index,
                    )
                    print(resultset)
                    index_opclass = self.embedding_index.op_class.value  # type: ignore[assignment]
                    if isinstance(self.embedding_index, DiskANN):
                        index_type = "diskann"
                    elif isinstance(self.embedding_index, HNSW):
                        index_type = "hnsw"
                    else:
                        index_type = "ivfflat"

                    for row in resultset:
                        if (
                            row["index_type"] == index_type
                            and row["index_opclass"] == index_opclass
                        ):
                            _logger.info(
                                "found a matching index: %s. overriding embedding_index.",
                                row,
                            )
                            index_opts = {
                                opts.split("=")[0]: opts.split("=")[1]
                                for opts in row["index_opts"]
                            }
                            index = (
                                DiskANN(op_class=index_opclass, **index_opts)
                                if index_type == "diskann"
                                else (
                                    HNSW(op_class=index_opclass, **index_opts)
                                    if index_type == "hnsw"
                                    else IVFFlat(op_class=index_opclass, **index_opts)
                                )
                            )
                            self.embedding_index = index
                            break
            elif self.embedding_index is None:
                _logger.info(
                    "embedding_index is not specified, and no vector index found in table '%s.%s'. defaulting to 'DiskANN' with 'vector_cosine_ops' opclass.",
                    self.schema_name,
                    self.table_name,
                )
                self.embedding_index = DiskANN(op_class=VectorOpClass.vector_cosine_ops)

        # if table does not exist, create it
        else:
            _logger.debug(
                "table '%s.%s' does not exist, creating it with the required columns",
                self.schema_name,
                self.table_name,
            )

            if self.embedding_type is None:
                _logger.warning(
                    "Embedding type is not specified, defaulting to 'vector'."
                )
                self.embedding_type = VectorType.vector

            if self.embedding_dimension is None:
                _logger.warning(
                    "Embedding dimension is not specified, defaulting to 1536."
                )
                self.embedding_dimension = PositiveInt(1_536)

            if self.embedding_index is None:
                _logger.warning(
                    "Embedding index is not specified, defaulting to 'DiskANN' with 'vector_cosine_ops' opclass."
                )
                self.embedding_index = DiskANN(op_class=VectorOpClass.vector_cosine_ops)

            with self.connection_pool.connection() as conn, conn.cursor() as cursor:
                cursor.execute(
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
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        id_column=sql.Identifier(self.id_column),
                        content_column=sql.Identifier(self.content_column),
                        embedding_column=sql.Identifier(self.embedding_column),
                        embedding_type=sql.Identifier(self.embedding_type.value),
                        embedding_dimension=sql.Literal(self.embedding_dimension),
                        metadata_column=sql.Identifier(self.metadata_column),
                    )
                )

        return self

    def _delete_rows_from_table(
        self, ids: list[str] | None = None, **kwargs: Any
    ) -> bool | None:
        """Delete rows from the table by their IDs or truncate the table.

        Args:
            ids (list[str] | None): List of IDs to delete. If None, truncates the table.
            **kwargs: Additional options, such as 'restart' and 'cascade' for truncation.

        Returns:
            bool | None: True if successful, False if an exception occurred, None otherwise.
        """
        with self.connection_pool.connection() as conn:
            conn.autocommit = False
            try:
                with conn.transaction() as _tx, conn.cursor() as cursor:
                    if ids is None:
                        restart = bool(kwargs.pop("restart", None))
                        cascade = bool(kwargs.pop("cascade", None))
                        cursor.execute(
                            sql.SQL(
                                """
                                truncate table {table_name} {restart} {cascade}
                                """
                            ).format(
                                table_name=sql.Identifier(
                                    self.schema_name, self.table_name
                                ),
                                restart=sql.SQL(
                                    "restart identity"
                                    if restart
                                    else "continue identity"
                                ),
                                cascade=sql.SQL("cascade" if cascade else "restrict"),
                            )
                        )
                    else:
                        cursor.execute(
                            sql.SQL(
                                """
                                delete from {table_name}
                                      where {id_column} = any(%(id)s)
                                """
                            ).format(
                                table_name=sql.Identifier(
                                    self.schema_name, self.table_name
                                ),
                                id_column=sql.Identifier(self.id_column),
                            ),
                            {"id": ids},
                        )
            except Exception:
                return False
            else:
                return True

    def _similarity_search_by_vector_with_distance(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[dict, float, np.ndarray | None]]:
        """Perform a similarity search using a vector embedding and return results with distances.

        Args:
            embedding (list[float]): The query embedding vector.
            k (int): Number of top results to return.
            **kwargs: Additional options such as 'return_embeddings', 'top_m', and 'filter_expression'.

        Returns:
            list[tuple[dict, float, np.ndarray | None]]: List of tuples containing document dict, distance, and optionally the embedding.
        """
        assert self.embedding_index is not None, (
            "embedding_index should have already been set"
        )
        return_embeddings = bool(kwargs.pop("return_embeddings", None))
        top_m = int(kwargs.pop("top_m", 5 * k))
        filter_expression: sql.SQL = kwargs.pop("filter_expression", sql.SQL("true"))
        with self.connection_pool.connection() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cursor:
                metadata_column: list[str]
                if isinstance(self.metadata_column, list):
                    metadata_column = [
                        col if isinstance(col, str) else col[0]
                        for col in self.metadata_column
                    ]
                elif isinstance(self.metadata_column, str):
                    metadata_column = [self.metadata_column]
                else:
                    metadata_column = []

                # do reranking for the following cases:
                #   - binary or scalar quantizations (for HNSW and IVFFlat), or
                #   - product quantization (for DiskANN)
                if (
                    self.embedding_index.op_class == VectorOpClass.bit_hamming_ops
                    or self.embedding_index.op_class == VectorOpClass.bit_jaccard_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_cosine_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_ip_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_l1_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_l2_ops
                    or (
                        isinstance(self.embedding_index, DiskANN)
                        and self.embedding_index.product_quantized
                    )
                ):
                    sql_query = sql.SQL(
                        """
                          select  {outer_columns},
                                  {embedding_column} {op} %(query)s as distance,
                                  {maybe_embedding_column}
                            from  (
                                     select {inner_columns}
                                       from {table_name}
                                      where {filter_expression}
                                   order by {expression} asc
                                      limit %(top_m)s
                                  ) i
                        order by  {embedding_column} {op} %(query)s asc
                           limit  %(top_k)s
                        """
                    ).format(
                        outer_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    *metadata_column,
                                ],
                            )
                        ),
                        embedding_column=sql.Identifier(self.embedding_column),
                        op=(
                            sql.SQL(
                                VectorOpClass.vector_cosine_ops.to_operator()
                            )  # TODO(arda): Think of getting this from outside
                            if (
                                self.embedding_index.op_class
                                in (
                                    VectorOpClass.bit_hamming_ops,
                                    VectorOpClass.bit_jaccard_ops,
                                )
                            )
                            else sql.SQL(self.embedding_index.op_class.to_operator())
                        ),
                        maybe_embedding_column=(
                            sql.Identifier(self.embedding_column)
                            if return_embeddings
                            else sql.SQL(" as ").join(
                                (sql.NULL, sql.Identifier(self.embedding_column))
                            )
                        ),
                        inner_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    self.embedding_column,
                                    *metadata_column,
                                ],
                            )
                        ),
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        filter_expression=filter_expression,
                        expression=(
                            sql.SQL(
                                "binary_quantize({embedding_column})::bit({embedding_dim}) {op} binary_quantize({query})"
                            ).format(
                                embedding_column=sql.Identifier(self.embedding_column),
                                embedding_dim=sql.Literal(self.embedding_dimension),
                                op=sql.SQL(self.embedding_index.op_class.to_operator()),
                                query=sql.Placeholder("query"),
                            )
                            if self.embedding_index.op_class
                            in (
                                VectorOpClass.bit_hamming_ops,
                                VectorOpClass.bit_jaccard_ops,
                            )
                            else (
                                sql.SQL(
                                    "{embedding_column}::halfvec({embedding_dim}) {op} {query}::halfvec({embedding_dim})"
                                ).format(
                                    embedding_column=sql.Identifier(
                                        self.embedding_column
                                    ),
                                    embedding_dim=sql.Literal(self.embedding_dimension),
                                    op=sql.SQL(
                                        self.embedding_index.op_class.to_operator()
                                    ),
                                    query=sql.Placeholder("query"),
                                )
                                if self.embedding_index.op_class
                                in (
                                    VectorOpClass.halfvec_cosine_ops,
                                    VectorOpClass.halfvec_ip_ops,
                                    VectorOpClass.halfvec_l1_ops,
                                    VectorOpClass.halfvec_l2_ops,
                                )
                                else sql.SQL("{embedding_column} {op} {query}").format(
                                    embedding_column=sql.Identifier(
                                        self.embedding_column
                                    ),
                                    op=sql.SQL(
                                        self.embedding_index.op_class.to_operator()
                                    ),
                                    query=sql.Placeholder("query"),
                                )
                            )
                        ),
                    )
                # otherwise (i.e., no quantization), do not do reranking
                else:
                    sql_query = sql.SQL(
                        """
                          select  {outer_columns},
                                  {embedding_column} {op} %(query)s as distance,
                                  {maybe_embedding_column}
                            from  {table_name}
                           where  {filter_expression}
                        order by  {embedding_column} {op} %(query)s asc
                           limit  %(top_k)s
                        """
                    ).format(
                        outer_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    *metadata_column,
                                ],
                            )
                        ),
                        embedding_column=sql.Identifier(self.embedding_column),
                        op=sql.SQL(self.embedding_index.op_class.to_operator()),
                        maybe_embedding_column=(
                            sql.Identifier(self.embedding_column)
                            if return_embeddings
                            else sql.SQL(" as ").join(
                                (sql.NULL, sql.Identifier(self.embedding_column))
                            )
                        ),
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        filter_expression=filter_expression,
                    )
                cursor.execute(
                    sql_query,
                    {
                        "query": np.array(embedding, dtype=np.float32),
                        "top_m": top_m,
                        "top_k": k,
                    },
                )
                resultset = cursor.fetchall()
        return [
            (
                {
                    "id": result[self.id_column],
                    "content": result[self.content_column],
                    "metadata": (
                        result[metadata_column[0]]
                        if isinstance(self.metadata_column, str)
                        else {col: result[col] for col in metadata_column}
                    ),
                },
                result["distance"],
                result.get(self.embedding_column),  # type: ignore[return-value]
            )
            for result in resultset
        ]

    def _get_by_ids(self, ids: Sequence[str], /) -> list[dict[str, Any]]:
        """Retrieve documents from the table by their IDs.

        Args:
            ids (Sequence[str]): List of IDs to retrieve.

        Returns:
            list[dict[str, Any]]: List of document dictionaries with id, content, embedding, and metadata.
        """
        with (
            self.connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            metadata_column: list[str]
            if isinstance(self.metadata_column, list):
                metadata_column = [
                    col if isinstance(col, str) else col[0]
                    for col in self.metadata_column
                ]
            elif isinstance(self.metadata_column, str):
                metadata_column = [self.metadata_column]
            else:
                metadata_column = []

            if ids is not None:
                where_clause = sql.SQL(" where {id_column} = any(%(id)s)").format(
                    id_column=sql.Identifier(self.id_column)
                )
            else:
                where_clause = sql.SQL("")

            get_sql = sql.SQL(
                """
                select {columns}
                    from {table_name}
                {where_clause}
                """
            ).format(
                columns=sql.SQL(", ").join(
                    map(
                        sql.Identifier,
                        [
                            self.id_column,
                            self.content_column,
                            self.embedding_column,
                            *metadata_column,
                        ],
                    )
                ),
                table_name=sql.Identifier(self.schema_name, self.table_name),
                where_clause=where_clause,
            )

            if ids is not None:
                cursor.execute(get_sql, {"id": ids})
            else:
                cursor.execute(get_sql)
            resultset = cursor.fetchall()
            documents = [
                {
                    "id": result[self.id_column],
                    "content": result[self.content_column],
                    "embedding": result[self.embedding_column],
                    "metadata": (
                        result[metadata_column[0]]
                        if isinstance(self.metadata_column, str)
                        else {col: result[col] for col in metadata_column}
                    ),
                }
                for result in resultset
            ]
            return documents


class AsyncBaseAzurePGVectorStore(BaseModel):
    connection_pool: AsyncAzurePGConnectionPool
    schema_name: str = "public"
    table_name: str = "vector_store"
    id_column: str = "id"
    content_column: str = "content"
    embedding_column: str = "embedding"
    embedding_type: VectorType | None = None
    embedding_dimension: PositiveInt | None = None
    embedding_index: Algorithm | None = None
    metadata_column: str | None = "metadata"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow arbitrary types like Embeddings and AzurePGConnectionPool
    )

    @model_validator(mode="after")
    def verify_and_init_store(self) -> Self:
        # verify that metadata_columns is not empty if provided
        if self.metadata_columns is not None and len(self.metadata_columns) == 0:
            raise ValueError("'metadata_columns' cannot be empty if provided.")

        _logger.debug(
            "checking if table '%s.%s' exists with the required columns",
            self.schema_name,
            self.table_name,
        )

        coroutine = self._ensure_table_verified()
        run_coroutine_in_sync(coroutine)

        return self

    async def _ensure_table_verified(self) -> None:
        async with (
            self.connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                sql.SQL(
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
                ),
                {"schema_name": self.schema_name, "table_name": self.table_name},
            )
            resultset = await cursor.fetchall()
            existing_columns: dict[str, str] = {
                row["column_name"]: row["column_type"] for row in resultset
            }

        # if table exists, verify that required columns exist and have correct types
        if len(existing_columns) > 0:
            _logger.debug(
                "table '%s.%s' exists with the following column mapping: %s",
                self.schema_name,
                self.table_name,
                existing_columns,
            )

            id_column_type = existing_columns.get(self.id_column)
            if id_column_type != "uuid":
                raise ValueError(
                    f"Table '{self.schema_name}.{self.table_name}' must have a column '{self.id_column}' of type 'uuid'."
                )

            content_column_type = existing_columns.get(self.content_column)
            if content_column_type is None or (
                content_column_type != "text"
                and not content_column_type.startswith("varchar")
            ):
                raise ValueError(
                    f"Table '{self.schema_name}.{self.table_name}' must have a column '{self.content_column}' of type 'text' or 'varchar'."
                )

            embedding_column_type = existing_columns.get(self.embedding_column)
            pattern = re.compile(r"(?P<type>\w+)(?:\((?P<dim>\d+)\))?")
            m = pattern.match(embedding_column_type if embedding_column_type else "")
            parsed_type: str | None = m.group("type") if m else None
            parsed_dim: PositiveInt | None = (
                PositiveInt(m.group("dim")) if m and m.group("dim") else None
            )

            vector_types = [t.value for t in VectorType.__members__.values()]
            if parsed_type not in vector_types:
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' must be one of the following types: {vector_types}."
                )
            elif (
                self.embedding_type is not None
                and parsed_type != self.embedding_type.value
            ):
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' has type '{parsed_type}', but the specified embedding_type is '{self.embedding_type.value}'. They must match."
                )
            elif self.embedding_type is None:
                _logger.info(
                    "embedding_type is not specified, but the column '%s' in table '%s.%s' has type '%s'. Overriding embedding_type accordingly.",
                    self.embedding_column,
                    self.schema_name,
                    self.table_name,
                    parsed_type,
                )
                self.embedding_type = VectorType(parsed_type)

            if parsed_dim is not None and self.embedding_dimension is None:
                _logger.info(
                    "embedding_dimension is not specified, but the column '%s' in table '%s.%s' has a dimension of %d. Overriding embedding_dimension accordingly.",
                    self.embedding_column,
                    self.schema_name,
                    self.table_name,
                    parsed_dim,
                )
                self.embedding_dimension = parsed_dim
            elif (
                parsed_dim is not None
                and self.embedding_dimension is not None
                and parsed_dim != self.embedding_dimension
            ):
                raise ValueError(
                    f"Column '{self.embedding_column}' in table '{self.schema_name}.{self.table_name}' has a dimension of {parsed_dim}, but the specified embedding_dimension is {self.embedding_dimension}. They must match."
                )

            if self.metadata_column is not None:
                existing_type = existing_columns.get(self.metadata_column)
                if existing_type is None:
                    raise ValueError(
                        f"Column '{self.metadata_column}' does not exist in table '{self.schema_name}.{self.table_name}'."
                    )

            async with (
                self.connection_pool.connection() as conn,
                conn.cursor(row_factory=dict_row) as cursor,
            ):
                _logger.debug(
                    "checking if table '%s.%s' has a vector index on column '%s'",
                    self.schema_name,
                    self.table_name,
                    self.embedding_column,
                )
                await cursor.execute(
                    sql.SQL(
                        """
                        with cte as (
                          select  n.nspname as schema_name,
                                  ct.relname as table_name,
                                  ci.relname as index_name,
                                  a.amname as index_type,
                                  pg_get_indexdef(
                                    ci.oid, -- index OID
                                    generate_series(1, array_length(ii.indkey, 1)), -- column no
                                    true -- pretty print
                                  ) as index_column,
                                  o.opcname as index_opclass,
                                  ci.reloptions as index_opts
                            from  pg_class ci
                                  join pg_index ii on ii.indexrelid = ci.oid
                                  join pg_am a on a.oid = ci.relam
                                  join pg_class ct on ct.oid = ii.indrelid
                                  join pg_namespace n on n.oid = ci.relnamespace
                                  join pg_opclass o on o.oid = any(ii.indclass)
                           where  ci.relkind = 'i'
                                  and ct.relkind = 'r'
                                  and ii.indisvalid
                                  and ii.indisready
                        ) select  schema_name, table_name, index_name, index_type,
                                  index_column, index_opclass, index_opts
                            from  cte
                           where  schema_name = %(schema_name)s
                                  and table_name = %(table_name)s
                                  and index_column like %(embedding_column)s
                                  and (
                                      index_opclass like '%%vector%%'
                                      or index_opclass like '%%halfvec%%'
                                      or index_opclass like '%%sparsevec%%'
                                      or index_opclass like '%%bit%%'
                                  )
                        order by  schema_name, table_name, index_name
                        """
                    ),
                    {
                        "schema_name": self.schema_name,
                        "table_name": self.table_name,
                        "embedding_column": f"%{self.embedding_column}%",
                    },
                )
                resultset = await cursor.fetchall()

            if len(resultset) > 0:
                _logger.debug(
                    "table '%s.%s' has %d vector index(es): %s",
                    self.schema_name,
                    self.table_name,
                    len(resultset),
                    resultset,
                )

                if self.embedding_index is None:
                    _logger.info(
                        "embedding_index is not specified, using the first found index: %s",
                        resultset[0],
                    )

                    index_type = resultset[0]["index_type"]
                    index_opclass = VectorOpClass(resultset[0]["index_opclass"])
                    index_opts = {
                        opts.split("=")[0]: opts.split("=")[1]
                        for opts in resultset[0]["index_opts"]
                    }

                    index = (
                        DiskANN(op_class=index_opclass, **index_opts)
                        if index_type == "diskann"
                        else HNSW(op_class=index_opclass, **index_opts)
                        if index_type == "hnsw"
                        else IVFFlat(op_class=index_opclass, **index_opts)
                    )

                    self.embedding_index = index
                else:
                    _logger.info(
                        "embedding_index is specified as '%s'; will try to find a matching index.",
                        self.embedding_index,
                    )

                    index_opclass = self.embedding_index.op_class.value  # type: ignore[assignment]
                    if isinstance(self.embedding_index, DiskANN):
                        index_type = "diskann"
                    elif isinstance(self.embedding_index, HNSW):
                        index_type = "hnsw"
                    else:
                        index_type = "ivfflat"

                    for row in resultset:
                        if (
                            row["index_type"] == index_type
                            and row["index_opclass"] == index_opclass
                        ):
                            _logger.info(
                                "found a matching index: %s. overriding embedding_index.",
                                row,
                            )
                            index_opts = {
                                opts.split("=")[0]: opts.split("=")[1]
                                for opts in row["index_opts"]
                            }
                            index = (
                                DiskANN(op_class=index_opclass, **index_opts)
                                if index_type == "diskann"
                                else HNSW(op_class=index_opclass, **index_opts)
                                if index_type == "hnsw"
                                else IVFFlat(op_class=index_opclass, **index_opts)
                            )
                            self.embedding_index = index
                            break
            elif self.embedding_index is None:
                _logger.info(
                    "embedding_index is not specified, and no vector index found in table '%s.%s'. defaulting to 'DiskANN' with 'vector_cosine_ops' opclass.",
                    self.schema_name,
                    self.table_name,
                )
                self.embedding_index = DiskANN(op_class=VectorOpClass.vector_cosine_ops)

        # if table does not exist, create it
        else:
            _logger.debug(
                "table '%s.%s' does not exist, creating it with the required columns",
                self.schema_name,
                self.table_name,
            )

            metadata_columns: list[tuple[str, str]] = []  # type: ignore[no-redef]
            if self.metadata_columns is None:
                _logger.warning(
                    "Metadata columns are not specified, defaulting to 'metadata' of type 'jsonb'."
                )
                metadata_columns = [("metadata", "jsonb")]
            elif isinstance(self.metadata_columns, str):
                _logger.warning(
                    "Metadata columns are specified as a string, defaulting to 'jsonb' type."
                )
                metadata_columns = [(self.metadata_columns, "jsonb")]
            elif isinstance(self.metadata_columns, list):
                _logger.warning(
                    "Metadata columns are specified as a list; defaulting to 'text' when type is not defined."
                )
                metadata_columns = [
                    (col[0], col[1]) if isinstance(col, tuple) else (col, "text")
                    for col in self.metadata_columns
                ]

            if self.embedding_type is None:
                _logger.warning(
                    "Embedding type is not specified, defaulting to 'vector'."
                )
                self.embedding_type = VectorType.vector

            if self.embedding_dimension is None:
                _logger.warning(
                    "Embedding dimension is not specified, defaulting to 1536."
                )
                self.embedding_dimension = PositiveInt(1_536)

            if self.embedding_index is None:
                _logger.warning(
                    "Embedding index is not specified, defaulting to 'DiskANN' with 'vector_cosine_ops' opclass."
                )
                self.embedding_index = DiskANN(op_class=VectorOpClass.vector_cosine_ops)

            async with (
                self.connection_pool.connection() as conn,
                conn.cursor() as cursor,
            ):
                await cursor.execute(
                    sql.SQL(
                        """
                        create table {table_name} (
                            {id_column} uuid primary key,
                            {content_column} text,
                            {embedding_column} {embedding_type}({embedding_dimension}),
                            {metadata_columns}
                        )
                        """
                    ).format(
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        id_column=sql.Identifier(self.id_column),
                        content_column=sql.Identifier(self.content_column),
                        embedding_column=sql.Identifier(self.embedding_column),
                        embedding_type=sql.Identifier(self.embedding_type.value),
                        embedding_dimension=sql.Literal(self.embedding_dimension),
                        metadata_columns=sql.SQL(", ").join(
                            sql.SQL("{col} {type}").format(
                                col=sql.Identifier(col),
                                type=sql.SQL(type),  # type: ignore[arg-type]
                            )
                            for col, type in metadata_columns
                        ),
                    )
                )

    async def _delete_rows_from_table(
        self, ids: list[str] | None = None, **kwargs: Any
    ) -> bool | None:
        """Delete rows from the table by their IDs or truncate the table.

        Args:
            ids (list[str] | None): List of IDs to delete. If None, truncates the table.
            **kwargs: Additional options, such as 'restart' and 'cascade' for truncation.

        Returns:
            bool | None: True if successful, False if an exception occurred, None otherwise.
        """
        async with self.connection_pool.connection() as conn:
            try:
                async with conn.transaction() as _tx, conn.cursor() as cursor:
                    if ids is None:
                        restart = bool(kwargs.pop("restart", None))
                        cascade = bool(kwargs.pop("cascade", None))
                        await cursor.execute(
                            sql.SQL(
                                """
                                truncate table {table_name} {restart} {cascade}
                                """
                            ).format(
                                table_name=sql.Identifier(
                                    self.schema_name, self.table_name
                                ),
                                restart=sql.SQL(
                                    "restart identity"
                                    if restart
                                    else "continue identity"
                                ),
                                cascade=sql.SQL("cascade" if cascade else "restrict"),
                            )
                        )
                    else:
                        await cursor.execute(
                            sql.SQL(
                                """
                                delete from {table_name}
                                      where {id_column} = any(%(id)s)
                                """
                            ).format(
                                table_name=sql.Identifier(
                                    self.schema_name, self.table_name
                                ),
                                id_column=sql.Identifier(self.id_column),
                            ),
                            {"id": ids},
                        )
            except Exception:
                return False
            else:
                return True

    async def _similarity_search_by_vector_with_distance(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[dict, float, np.ndarray | None]]:
        """Perform a similarity search using a vector embedding and return results with distances.

        Args:
            embedding (list[float]): The query embedding vector.
            k (int): Number of top results to return.
            **kwargs: Additional options such as 'return_embeddings', 'top_m', and 'filter_expression'.

        Returns:
            list[tuple[dict, float, np.ndarray | None]]: List of tuples containing document dict, distance, and optionally the embedding.
        """
        assert self.embedding_index is not None, (
            "embedding_index should have already been set"
        )
        return_embeddings = bool(kwargs.pop("return_embeddings", None))
        top_m = int(kwargs.pop("top_m", 5 * k))
        filter_expression: sql.SQL = kwargs.pop("filter_expression", sql.SQL("true"))
        async with self.connection_pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cursor:
                metadata_column: list[str]
                if isinstance(self.metadata_column, list):
                    metadata_column = [
                        col if isinstance(col, str) else col[0]
                        for col in self.metadata_column
                    ]
                elif isinstance(self.metadata_column, str):
                    metadata_column = [self.metadata_column]
                else:
                    metadata_column = []

                # do reranking for the following cases:
                #   - binary or scalar quantizations (for HNSW and IVFFlat), or
                #   - product quantization (for DiskANN)
                if (
                    self.embedding_index.op_class == VectorOpClass.bit_hamming_ops
                    or self.embedding_index.op_class == VectorOpClass.bit_jaccard_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_cosine_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_ip_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_l1_ops
                    or self.embedding_index.op_class == VectorOpClass.halfvec_l2_ops
                    or (
                        isinstance(self.embedding_index, DiskANN)
                        and self.embedding_index.product_quantized
                    )
                ):
                    sql_query = sql.SQL(
                        """
                          select  {outer_columns},
                                  {embedding_column} {op} %(query)s as distance,
                                  {maybe_embedding_column}
                            from  (
                                     select {inner_columns}
                                       from {table_name}
                                      where {filter_expression}
                                   order by {expression} asc
                                      limit %(top_m)s
                                  ) i
                        order by  {embedding_column} {op} %(query)s asc
                           limit  %(top_k)s
                        """
                    ).format(
                        outer_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    *metadata_column,
                                ],
                            )
                        ),
                        embedding_column=sql.Identifier(self.embedding_column),
                        op=(
                            sql.SQL(
                                VectorOpClass.vector_cosine_ops.to_operator()
                            )  # TODO(arda): Think of getting this from outside
                            if (
                                self.embedding_index.op_class
                                in (
                                    VectorOpClass.bit_hamming_ops,
                                    VectorOpClass.bit_jaccard_ops,
                                )
                            )
                            else sql.SQL(self.embedding_index.op_class.to_operator())
                        ),
                        maybe_embedding_column=(
                            sql.Identifier(self.embedding_column)
                            if return_embeddings
                            else sql.SQL(" as ").join(
                                (sql.NULL, sql.Identifier(self.embedding_column))
                            )
                        ),
                        inner_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    self.embedding_column,
                                    *metadata_column,
                                ],
                            )
                        ),
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        filter_expression=filter_expression,
                        expression=(
                            sql.SQL(
                                "binary_quantize({embedding_column})::bit({embedding_dim}) {op} binary_quantize({query})"
                            ).format(
                                embedding_column=sql.Identifier(self.embedding_column),
                                embedding_dim=sql.Literal(self.embedding_dimension),
                                op=sql.SQL(self.embedding_index.op_class.to_operator()),
                                query=sql.Placeholder("query"),
                            )
                            if self.embedding_index.op_class
                            in (
                                VectorOpClass.bit_hamming_ops,
                                VectorOpClass.bit_jaccard_ops,
                            )
                            else (
                                sql.SQL(
                                    "{embedding_column}::halfvec({embedding_dim}) {op} {query}::halfvec({embedding_dim})"
                                ).format(
                                    embedding_column=sql.Identifier(
                                        self.embedding_column
                                    ),
                                    embedding_dim=sql.Literal(self.embedding_dimension),
                                    op=sql.SQL(
                                        self.embedding_index.op_class.to_operator()
                                    ),
                                    query=sql.Placeholder("query"),
                                )
                                if self.embedding_index.op_class
                                in (
                                    VectorOpClass.halfvec_cosine_ops,
                                    VectorOpClass.halfvec_ip_ops,
                                    VectorOpClass.halfvec_l1_ops,
                                    VectorOpClass.halfvec_l2_ops,
                                )
                                else sql.SQL("{embedding_column} {op} {query}").format(
                                    embedding_column=sql.Identifier(
                                        self.embedding_column
                                    ),
                                    op=sql.SQL(
                                        self.embedding_index.op_class.to_operator()
                                    ),
                                    query=sql.Placeholder("query"),
                                )
                            )
                        ),
                    )
                # otherwise (i.e., no quantization), do not do reranking
                else:
                    sql_query = sql.SQL(
                        """
                          select  {outer_columns},
                                  {embedding_column} {op} %(query)s as distance,
                                  {maybe_embedding_column}
                            from  {table_name}
                           where  {filter_expression}
                        order by  {embedding_column} {op} %(query)s asc
                           limit  %(top_k)s
                        """
                    ).format(
                        outer_columns=sql.SQL(", ").join(
                            map(
                                sql.Identifier,
                                [
                                    self.id_column,
                                    self.content_column,
                                    *metadata_column,
                                ],
                            )
                        ),
                        embedding_column=sql.Identifier(self.embedding_column),
                        op=sql.SQL(self.embedding_index.op_class.to_operator()),
                        maybe_embedding_column=(
                            sql.Identifier(self.embedding_column)
                            if return_embeddings
                            else sql.SQL(" as ").join(
                                (sql.NULL, sql.Identifier(self.embedding_column))
                            )
                        ),
                        table_name=sql.Identifier(self.schema_name, self.table_name),
                        filter_expression=filter_expression,
                    )
                await cursor.execute(
                    sql_query,
                    {
                        "query": np.array(embedding, dtype=np.float32),
                        "top_m": top_m,
                        "top_k": k,
                    },
                )
                resultset = await cursor.fetchall()
        return [
            (
                {
                    "id": result[self.id_column],
                    "content": result[self.content_column],
                    "metadata": (
                        result[metadata_column[0]]
                        if isinstance(self.metadata_column, str)
                        else {col: result[col] for col in metadata_column}
                    ),
                },
                result["distance"],
                result.get(self.embedding_column),  # type: ignore[return-value]
            )
            for result in resultset
        ]

    async def _get_by_ids(self, ids: Sequence[str], /) -> list[dict[str, Any]]:
        """Retrieve documents from the table by their IDs.

        Args:
            ids (Sequence[str]): List of IDs to retrieve.

        Returns:
            list[dict[str, Any]]: List of document dictionaries with id, content, embedding, and metadata.
        """
        async with (
            self.connection_pool.connection() as conn,
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            metadata_column: list[str]
            if isinstance(self.metadata_column, list):
                metadata_column = [
                    col if isinstance(col, str) else col[0]
                    for col in self.metadata_column
                ]
            elif isinstance(self.metadata_column, str):
                metadata_column = [self.metadata_column]
            else:
                metadata_column = []

            if ids is not None:
                where_clause = sql.SQL(" where {id_column} = any(%(id)s)").format(
                    id_column=sql.Identifier(self.id_column)
                )
            else:
                where_clause = sql.SQL("")

            get_sql = sql.SQL(
                """
                select {columns}
                    from {table_name}
                {where_clause}
                """
            ).format(
                columns=sql.SQL(", ").join(
                    map(
                        sql.Identifier,
                        [
                            self.id_column,
                            self.content_column,
                            self.embedding_column,
                            *metadata_column,
                        ],
                    )
                ),
                table_name=sql.Identifier(self.schema_name, self.table_name),
                where_clause=where_clause,
            )

            if ids is not None:
                await cursor.execute(get_sql, {"id": ids})
            else:
                await cursor.execute(get_sql)
            resultset = await cursor.fetchall()
            documents = [
                {
                    "id": result[self.id_column],
                    "content": result[self.content_column],
                    "embedding": result[self.embedding_column],
                    "metadata": (
                        result[metadata_column[0]]
                        if isinstance(self.metadata_column, str)
                        else {col: result[col] for col in metadata_column}
                    ),
                }
                for result in resultset
            ]
            return documents
