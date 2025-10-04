"""VectorStore integration for Azure Database for PostgreSQL using LlamaIndex."""

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
from pgvector.psycopg import register_vector_async  # type: ignore[import-untyped]
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

from .common import (
    Algorithm,
    AsyncAzurePGConnectionPool,
    AsyncBaseAzurePGVectorStore,
    _table_row_to_node,
    metadata_filters_to_sql,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class AsyncAzurePGVectorStore(BasePydanticVectorStore, AsyncBaseAzurePGVectorStore):
    """Azure PostgreSQL vector store for LlamaIndex."""

    stores_text: bool = True
    metadata_columns: str | None = "metadata"

    @classmethod
    def class_name(cls) -> str:
        """Return the class name for this vector store."""
        return "AzurePGVectorStore"

    @property
    def client(self) -> None:
        """Return the client property (not used for AzurePGVectorStore)."""
        return

    @asynccontextmanager
    async def _connection(self) -> AsyncGenerator[AsyncConnection, None]:
        async with self.connection_pool.connection() as conn:
            yield conn

    @override
    @classmethod
    def from_params(
        cls,
        connection_pool: AsyncAzurePGConnectionPool,
        schema_name: str = "public",
        table_name: str = "llamaindex_vectors",
        embed_dim: int = 1536,
        embedding_index: Algorithm | None = None,
    ) -> "AsyncAzurePGVectorStore":
        """Create an AsyncAzurePGVectorStore from connection and configuration parameters."""
        return cls(
            connection_pool=connection_pool,
            schema_name=schema_name,
            table_name=table_name,
            embed_dim=embed_dim,
            embedding_index=embedding_index,
        )

    def _get_insert_sql_dict(
        self, node: BaseNode, on_conflict_update: bool
    ) -> tuple[sql.SQL, dict[str, Any]]:
        """Get the SQL command and dictionary for inserting a node."""
        if on_conflict_update:
            update = sql.SQL(
                """
                UPDATE SET
                    {content_col} = EXCLUDED.{content_col},
                    {embedding_col} = EXCLUDED.{embedding_col},
                    {metadata_col} = EXCLUDED.{metadata_col}
                """
            ).format(
                id_col=sql.Identifier(self.id_column),
                content_col=sql.Identifier(self.content_column),
                embedding_col=sql.Identifier(self.embedding_column),
                metadata_col=sql.Identifier(self.metadata_columns),
            )
        else:
            update = sql.SQL("nothing")
        insert_sql = sql.SQL(
            """
            INSERT INTO {schema}.{table} ({id_col}, {content_col}, {embedding_col}, {metadata_col})
            VALUES (%(id)s, %(content)s, %(embedding)s, %(metadata)s)
            ON CONFLICT ({id_col}) DO {update}
        """
        ).format(
            schema=sql.Identifier(self.schema_name),
            table=sql.Identifier(self.table_name),
            id_col=sql.Identifier(self.id_column),
            content_col=sql.Identifier(self.content_column),
            embedding_col=sql.Identifier(self.embedding_column),
            metadata_col=sql.Identifier(self.metadata_columns),
            update=update,
        )

        return (
            insert_sql,
            {
                "id": node.node_id,
                "content": node.get_content(metadata_mode=MetadataMode.NONE),
                "embedding": np.array(node.get_embedding(), dtype=np.float32),
                "metadata": Jsonb(node_to_metadata_dict(node)),
            },
        )

    @override
    async def async_add(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously add nodes to vector store."""
        on_conflict_update = bool(kwargs.pop("on_conflict_update", None))
        ids = []
        async with self._connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cursor:
                for node in nodes:
                    ids.append(node.node_id)
                    insert_sql, insert_dict = self._get_insert_sql_dict(
                        node, on_conflict_update=on_conflict_update
                    )
                    await cursor.execute(insert_sql, insert_dict)
        return ids

    @override
    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Asynchronously query the vector store."""
        results = await self._similarity_search_by_vector_with_distance(
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

    @override
    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a node from the vector store by reference document ID.

        Args:
            ref_doc_id: The reference document ID to delete.
            **delete_kwargs: Additional keyword arguments.
        """
        async with self.connection_pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cursor:
                delete_sql = sql.SQL(
                    "DELETE FROM {table} WHERE metadata ->> 'doc_id' = %s"
                ).format(table=sql.Identifier(self.schema_name, self.table_name))
                await cursor.execute(delete_sql, (ref_doc_id,))

    @override
    async def adelete_nodes(
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
        if not node_ids:
            return

        await self._delete_rows_from_table(
            ids=node_ids, filters=metadata_filters_to_sql(filters), **delete_kwargs
        )

    @override
    async def aclear(self) -> None:
        """Clear all data from the vector store table."""
        async with self.connection_pool.connection() as conn:
            await register_vector_async(conn)
            async with conn.cursor(row_factory=dict_row) as cursor:
                stmt = sql.SQL("TRUNCATE TABLE {schema}.{table}").format(
                    schema=sql.Identifier(self.schema_name),
                    table=sql.Identifier(self.table_name),
                )
                await cursor.execute(stmt)
                await conn.commit()

    @override
    async def aget_nodes(
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
        documents = await self._get_by_ids(node_ids)
        nodes = []
        for doc in documents:
            node = _table_row_to_node(doc)
            nodes.append(node)

        return nodes

    @override
    def add(
        self,
        nodes: list[BaseNode],
        **kwargs: Any,
    ) -> list[str]:
        """Not implemented for AsyncAzurePGVectorStore; use AzurePGVectorStore instead."""
        raise NotImplementedError(
            "Add interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Not implemented for AsyncAzurePGVectorStore; use AzurePGVectorStore instead."""
        raise NotImplementedError(
            "Delete interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Not implemented for AsyncAzurePGVectorStore; use AzurePGVectorStore instead."""
        raise NotImplementedError(
            "Query interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def get_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> list[BaseNode]:
        """Not implemented for AsyncAzurePGVectorStore; use AzurePGVectorStore instead."""
        raise NotImplementedError(
            "get_nodes interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def clear(self) -> None:
        """Not implemented for AsyncAzurePGVectorStore; use AzurePGVectorStore instead."""
        raise NotImplementedError(
            "clear interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )

    @override
    def delete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Not implemented for AsyncAzurePGVectorStore; use AzurePGVectorStore instead."""
        raise NotImplementedError(
            "delete_nodes interface is not implemented for AsyncAzurePGVectorStore: use AzurePGVectorStore, instead."
        )
