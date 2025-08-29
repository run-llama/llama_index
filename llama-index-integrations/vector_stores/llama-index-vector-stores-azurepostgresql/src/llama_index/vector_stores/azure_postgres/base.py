"""VectorStore integration for Azure Database for PostgreSQL using LlamaIndex."""

from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np
from pgvector.psycopg import register_vector  # type: ignore[import-untyped]
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from .common import (
    AzurePGConnectionPool,
    BaseAzurePGVectorStore,
)


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


class AzurePGVectorStore(BasePydanticVectorStore, BaseAzurePGVectorStore):
    """Azure PostgreSQL vector store for LlamaIndex."""

    stores_text: bool = True

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for this vector store."""
        return "AzurePGVectorStore"

    @property
    def client(self) -> None:
        """Return the client property (not used for AzurePGVectorStore)."""
        return

    @classmethod
    def from_params(
        cls,
        connection_pool: AzurePGConnectionPool,
        schema_name: str = "public",
        table_name: str = "llamaindex_vectors",
        embed_dim: int = 1536,
        pg_diskann_kwargs: dict | None = None,
        hnsw_kwargs: dict | None = None,
        ivfflat_kwargs: dict | None = None,
    ) -> "AzurePGVectorStore":
        """Create an AzurePGVectorStore from connection and configuration parameters."""
        return cls(
            connection_pool=connection_pool,
            schema_name=schema_name,
            table_name=table_name,
            embed_dim=embed_dim,
            pg_diskann_kwargs=pg_diskann_kwargs or {},
            hnsw_kwargs=hnsw_kwargs or {},
            ivfflat_kwargs=ivfflat_kwargs or {},
        )

    def _node_to_table_row(self, node: BaseNode) -> Any:
        """Convert a BaseNode to a table row dictionary for insertion."""
        return {
            "id": node.node_id,
            "content": node.get_content(metadata_mode=MetadataMode.NONE),
            "embedding": node.get_embedding(),
            "metadata": node_to_metadata_dict(
                node,
                remove_text=True,
                flat_metadata=self.flat_metadata,
            ),
        }

    def _table_row_to_node(self, row: dict[str, Any]) -> BaseNode:
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

    def add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        """Add a list of BaseNode objects to the vector store.

        Args:
            nodes: List of BaseNode objects to add.
            **add_kwargs: Additional keyword arguments.

        Returns:
            List of node IDs added.
        """
        ids = []
        insert_sql = sql.SQL(
            """
            INSERT INTO {schema}.{table} ({id_col}, {content_col}, {embedding_col}, {metadata_col})
            VALUES (%(id)s, %(content)s, %(embedding)s, %(metadata)s)
            ON CONFLICT ({id_col}) DO UPDATE SET
                {content_col} = EXCLUDED.{content_col},
                {embedding_col} = EXCLUDED.{embedding_col},
                {metadata_col} = EXCLUDED.{metadata_col}
        """
        ).format(
            schema=sql.Identifier(self.schema_name),
            table=sql.Identifier(self.table_name),
            id_col=sql.Identifier(self.id_column),
            content_col=sql.Identifier(self.content_column),
            embedding_col=sql.Identifier(self.embedding_column),
            metadata_col=sql.Identifier(self.metadata_columns),
        )

        with self.connection_pool.connection() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cursor:
                for node in nodes:
                    print(node.node_id)
                    ids.append(node.node_id)
                    cursor.execute(
                        insert_sql,
                        {
                            "id": node.node_id,
                            "content": node.get_content(
                                metadata_mode=MetadataMode.NONE
                            ),
                            "embedding": np.array(
                                node.get_embedding(), dtype=np.float32
                            ),
                            "metadata": Jsonb(node_to_metadata_dict(node)),
                        },
                    )
        return ids

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Perform a similarity search using the provided query.

        Args:
            query: VectorStoreQuery object containing the query embedding and parameters.
            **kwargs: Additional keyword arguments.

        Returns:
            VectorStoreQueryResult containing the search results.
        """
        results = self._similarity_search_by_vector_with_distance(
            embedding=query.query_embedding,
            k=query.similarity_top_k,
            filter_expression=metadata_filters_to_sql(query.filters),
            **kwargs,
        )
        nodes = []
        similarities = []
        ids = []
        for row in results:
            node = metadata_dict_to_node(row[0]["metadata"], text=row[0]["content"])
            nodes.append(node)
            similarities.append(row[1])
            ids.append(row[0]["id"])

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a node from the vector store by reference document ID.

        Args:
            ref_doc_id: The reference document ID to delete.
            **delete_kwargs: Additional keyword arguments.
        """
        with self.connection_pool.connection() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cursor:
                delete_sql = sql.SQL(
                    "DELETE FROM {schema}.{table} WHERE metadata ->> 'doc_id' = %s"
                ).format(
                    schema=sql.Identifier(self.schema_name),
                    table=sql.Identifier(self.table_name),
                )
                cursor.execute(delete_sql, (ref_doc_id,))

    def delete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes from the vector store by node IDs or filters.

        Args:
            node_ids: Optional list of node IDs to delete.
            filters: Optional MetadataFilters to filter nodes for deletion.
            **delete_kwargs: Additional keyword arguments.
        """
        if not node_ids and not filters:
            return

        self._delete_rows_from_table(
            ids=node_ids, filters=metadata_filters_to_sql(filters), **delete_kwargs
        )

    def clear(self) -> None:
        """Clear all data from the vector store table."""
        with self.connection_pool.connection() as conn:
            register_vector(conn)
            with conn.cursor(row_factory=dict_row) as cursor:
                stmt = sql.SQL("TRUNCATE TABLE {schema}.{table}").format(
                    schema=sql.Identifier(self.schema_name),
                    table=sql.Identifier(self.table_name),
                )
                cursor.execute(stmt)
                conn.commit()

    def get_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Retrieve nodes by IDs or filters.

        Args:
            node_ids: Optional list of node IDs to retrieve.
            filters: Optional MetadataFilters to filter nodes.
            **kwargs: Additional keyword arguments.

        Returns:
            List of BaseNode objects matching the criteria.
        """
        # TODO: Implement filter handling
        documents = self._get_by_ids(node_ids)
        nodes = []
        for doc in documents:
            node = self._table_row_to_node(doc)
            nodes.append(node)

        return nodes
