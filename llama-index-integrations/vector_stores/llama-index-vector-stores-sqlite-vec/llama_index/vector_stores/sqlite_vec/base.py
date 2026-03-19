"""sqlite-vec vector store."""

import asyncio
import json
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Any, Optional
from collections.abc import Sequence

from typing_extensions import override

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

logger = logging.getLogger(__name__)

OVERSAMPLE_FACTOR = 10


def _serialize_f32(vector: list[float]) -> bytes:
    """Serialize a list of floats into a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


class SqliteVecVectorStore(BasePydanticVectorStore):
    """
    sqlite-vec vector store.

    In this vector store, embeddings are stored within a SQLite database
    using the sqlite-vec extension.

    During query time, the index uses sqlite-vec to query for the top
    k most similar nodes.

    Examples:
        `pip install llama-index-vector-stores-sqlite-vec`

        ```python
        from llama_index.vector_stores.sqlite_vec import SqliteVecVectorStore

        # in-memory
        vector_store = SqliteVecVectorStore(embed_dim=1536)

        # persist to disk
        vector_store = SqliteVecVectorStore(
            database_path="./vectors.db",
            embed_dim=1536,
        )
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = True

    database_path: str
    table_name: str
    embed_dim: int
    distance_metric: str

    _connection: Optional[sqlite3.Connection] = PrivateAttr(default=None)

    def __init__(
        self,
        database_path: str = ":memory:",
        table_name: str = "documents",
        embed_dim: int = 1536,
        distance_metric: str = "cosine",
        connection: Optional[sqlite3.Connection] = None,
        **kwargs: Any,
    ) -> None:
        """
        Init params.

        Args:
            database_path: Path to the SQLite database file, or ":memory:" for in-memory.
            table_name: Name of the table to store documents in.
            embed_dim: Dimension of the embedding vectors.
            distance_metric: Distance metric to use. One of "cosine" or "l2".
            connection: Optional pre-existing sqlite3 connection.

        """
        super().__init__(
            stores_text=True,
            database_path=database_path,
            table_name=table_name,
            embed_dim=embed_dim,
            distance_metric=distance_metric,
        )

        if connection is not None:
            self._connection = connection
        else:
            self._connection = self._connect(database_path)

        self._initialize_db()

    @classmethod
    def from_local(
        cls,
        database_path: str,
        table_name: str = "documents",
        embed_dim: int = 1536,
        distance_metric: str = "cosine",
        **kwargs: Any,
    ) -> "SqliteVecVectorStore":
        """Load a sqlite-vec vector store from a local file."""
        return cls(
            database_path=database_path,
            table_name=table_name,
            embed_dim=embed_dim,
            distance_metric=distance_metric,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SqliteVecVectorStore"

    @property
    def client(self) -> sqlite3.Connection:
        """Return the SQLite connection."""
        if self._connection is None:
            self._connection = self._connect(self.database_path)
            self._initialize_db()
        return self._connection

    @classmethod
    def _connect(cls, database_path: str) -> sqlite3.Connection:
        """Create a connection to the SQLite database."""
        if database_path != ":memory:":
            db_path = Path(database_path)
            if not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)

        return sqlite3.connect(database_path, check_same_thread=False)

    def _initialize_db(self) -> None:
        """Initialize the database: load sqlite-vec extension and create tables."""
        import sqlite_vec

        conn = self.client
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        distance_clause = ""
        if self.distance_metric == "cosine":
            distance_clause = " distance_metric=cosine"

        # vec0 virtual table for vector search
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}_vec USING vec0(
                embedding float[{self.embed_dim}]{distance_clause}
            )
        """)

        # Regular table for metadata and text
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                node_id TEXT PRIMARY KEY,
                text TEXT,
                metadata TEXT,
                vec_rowid INTEGER UNIQUE
            )
        """)

        conn.commit()

    @override
    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """Add nodes to the vector store."""
        conn = self.client
        ids = []

        for node in nodes:
            embedding = node.get_embedding()
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            metadata = node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            )
            metadata_str = json.dumps(metadata)

            # Insert embedding into vec0 table
            cursor = conn.execute(
                f"INSERT INTO {self.table_name}_vec(embedding) VALUES (?)",
                [_serialize_f32(embedding)],
            )
            vec_rowid = cursor.lastrowid

            # Insert metadata into regular table
            conn.execute(
                f"""INSERT OR REPLACE INTO {self.table_name}
                    (node_id, text, metadata, vec_rowid) VALUES (?, ?, ?, ?)""",
                [node.node_id, text, metadata_str, vec_rowid],
            )

            ids.append(node.node_id)

        conn.commit()
        return ids

    @override
    async def async_add(
        self, nodes: Sequence[BaseNode], **add_kwargs: Any
    ) -> list[str]:
        return await asyncio.to_thread(self.add, nodes, **add_kwargs)

    @override
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query the vector store for top k most similar nodes."""
        conn = self.client
        query_embedding = _serialize_f32(query.query_embedding)

        has_filters = query.filters is not None and len(query.filters.filters) > 0

        if has_filters:
            # Over-fetch from vec0 then post-filter with metadata
            fetch_k = query.similarity_top_k * OVERSAMPLE_FACTOR

            rows = conn.execute(
                f"""
                SELECT
                    m.node_id,
                    m.text,
                    m.metadata,
                    v.distance
                FROM {self.table_name}_vec v
                INNER JOIN {self.table_name} m ON m.vec_rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                """,
                [query_embedding, fetch_k],
            ).fetchall()

            # Apply metadata filters in Python
            filtered_rows = []
            for row in rows:
                metadata = json.loads(row[2])
                if self._matches_filters(metadata, query.filters):
                    filtered_rows.append(row)
                    if len(filtered_rows) >= query.similarity_top_k:
                        break

            rows = filtered_rows
        else:
            rows = conn.execute(
                f"""
                SELECT
                    m.node_id,
                    m.text,
                    m.metadata,
                    v.distance
                FROM {self.table_name}_vec v
                INNER JOIN {self.table_name} m ON m.vec_rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                """,
                [query_embedding, query.similarity_top_k],
            ).fetchall()

        return self._rows_to_query_result(rows)

    @override
    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        return await asyncio.to_thread(self.query, query, **kwargs)

    @override
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete nodes using ref_doc_id."""
        conn = self.client

        rows = conn.execute(
            f"SELECT vec_rowid FROM {self.table_name} WHERE json_extract(metadata, '$.ref_doc_id') = ?",
            [ref_doc_id],
        ).fetchall()

        for (vec_rowid,) in rows:
            conn.execute(
                f"DELETE FROM {self.table_name}_vec WHERE rowid = ?", [vec_rowid]
            )

        conn.execute(
            f"DELETE FROM {self.table_name} WHERE json_extract(metadata, '$.ref_doc_id') = ?",
            [ref_doc_id],
        )
        conn.commit()

    @override
    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        return await asyncio.to_thread(self.delete, ref_doc_id, **delete_kwargs)

    @override
    def delete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes using node_ids and/or filters."""
        conn = self.client

        if node_ids is not None:
            placeholders = ",".join("?" for _ in node_ids)
            rows = conn.execute(
                f"SELECT vec_rowid FROM {self.table_name} WHERE node_id IN ({placeholders})",
                node_ids,
            ).fetchall()

            for (vec_rowid,) in rows:
                conn.execute(
                    f"DELETE FROM {self.table_name}_vec WHERE rowid = ?", [vec_rowid]
                )

            conn.execute(
                f"DELETE FROM {self.table_name} WHERE node_id IN ({placeholders})",
                node_ids,
            )

        if filters is not None:
            # Get all rows and filter in Python
            all_rows = conn.execute(
                f"SELECT node_id, metadata, vec_rowid FROM {self.table_name}"
            ).fetchall()

            for node_id, metadata_str, vec_rowid in all_rows:
                metadata = json.loads(metadata_str)
                if self._matches_filters(metadata, filters):
                    conn.execute(
                        f"DELETE FROM {self.table_name}_vec WHERE rowid = ?",
                        [vec_rowid],
                    )
                    conn.execute(
                        f"DELETE FROM {self.table_name} WHERE node_id = ?", [node_id]
                    )

        conn.commit()

    @override
    async def adelete_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        return await asyncio.to_thread(
            self.delete_nodes, node_ids, filters, **delete_kwargs
        )

    @override
    def get_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **get_kwargs: Any,
    ) -> list[BaseNode]:
        """Get nodes using node_ids and/or filters."""
        conn = self.client

        if node_ids is not None:
            placeholders = ",".join("?" for _ in node_ids)
            sql = f"""
                SELECT m.node_id, m.text, m.metadata, v.embedding
                FROM {self.table_name} m
                LEFT JOIN {self.table_name}_vec v ON v.rowid = m.vec_rowid
                WHERE m.node_id IN ({placeholders})
            """
            rows = conn.execute(sql, node_ids).fetchall()
        else:
            sql = f"""
                SELECT m.node_id, m.text, m.metadata, v.embedding
                FROM {self.table_name} m
                LEFT JOIN {self.table_name}_vec v ON v.rowid = m.vec_rowid
            """
            rows = conn.execute(sql).fetchall()

        nodes = []
        for node_id, text, metadata_str, embedding_bytes in rows:
            metadata = json.loads(metadata_str)

            if filters is not None and not self._matches_filters(metadata, filters):
                continue

            node = metadata_dict_to_node(metadata=metadata, text=text)
            if embedding_bytes is not None:
                num_floats = len(embedding_bytes) // 4
                node.embedding = list(struct.unpack(f"{num_floats}f", embedding_bytes))
            nodes.append(node)

        return nodes

    @override
    async def aget_nodes(
        self,
        node_ids: Optional[list[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **get_kwargs: Any,
    ) -> list[BaseNode]:
        return await asyncio.to_thread(self.get_nodes, node_ids, filters, **get_kwargs)

    @override
    def clear(self, **clear_kwargs: Any) -> None:
        """Clear the vector store."""
        conn = self.client
        conn.execute(f"DELETE FROM {self.table_name}")
        conn.execute(f"DELETE FROM {self.table_name}_vec")
        conn.commit()

    @override
    async def aclear(self, **clear_kwargs: Any) -> None:
        return await asyncio.to_thread(self.clear, **clear_kwargs)

    def _rows_to_query_result(self, rows: list[tuple]) -> VectorStoreQueryResult:
        """Convert query result rows to VectorStoreQueryResult."""
        nodes = []
        similarities = []
        ids = []

        for node_id, text, metadata_str, distance in rows:
            metadata = json.loads(metadata_str)
            node = metadata_dict_to_node(metadata=metadata, text=text)

            nodes.append(node)
            ids.append(node_id)

            # Convert distance to similarity
            if self.distance_metric == "cosine":
                similarities.append(1.0 - distance)
            else:
                # L2 distance
                similarities.append(-distance)

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _matches_filters(self, metadata: dict, filters: MetadataFilters) -> bool:
        """
        Check if metadata matches the given filters.

        Uses SQL NULL semantics for NOT conditions: if a key is missing,
        the comparison is NULL and NOT(NULL) -> NULL -> excluded.
        """
        if not filters.filters:
            return True

        if filters.condition == FilterCondition.NOT:
            # Use tri-state evaluation to match SQL NULL semantics
            results = []
            for f in filters.filters:
                if isinstance(f, MetadataFilters):
                    results.append(self._matches_filters(metadata, f))
                elif isinstance(f, MetadataFilter):
                    results.append(self._matches_single_filter_nullable(metadata, f))

            # SQL AND with NULLs: False if any False, NULL if any None, else True
            combined = True
            for r in results:
                if r is False:
                    combined = False
                    break
                if r is None:
                    combined = None

            # NOT(NULL) -> NULL -> excluded; NOT(True) -> False; NOT(False) -> True
            if combined is None:
                return False
            return not combined

        results = []
        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                results.append(self._matches_filters(metadata, f))
            elif isinstance(f, MetadataFilter):
                results.append(self._matches_single_filter(metadata, f))

        if filters.condition == FilterCondition.AND:
            return all(results)
        elif filters.condition == FilterCondition.OR:
            return any(results)

        return True

    def _matches_single_filter_nullable(
        self, metadata: dict, f: MetadataFilter
    ) -> Optional[bool]:
        """Like _matches_single_filter but returns None for missing keys (SQL NULL)."""
        if f.key not in metadata and f.operator != FilterOperator.IS_EMPTY:
            return None
        return self._matches_single_filter(metadata, f)

    def _matches_single_filter(self, metadata: dict, f: MetadataFilter) -> bool:
        """
        Check if metadata matches a single filter.

        Follows SQL NULL semantics: comparisons with a missing key return False
        (like SQL NULL != 'x' -> NULL -> falsy).
        """
        key_exists = f.key in metadata
        value = metadata.get(f.key)

        if f.operator == FilterOperator.IS_EMPTY:
            return not key_exists or value is None or value == [] or value == ""

        # For all comparison operators, missing key -> False (SQL NULL semantics)
        if not key_exists:
            return False

        if f.operator == FilterOperator.EQ:
            return value == f.value
        elif f.operator == FilterOperator.NE:
            return value is not None and value != f.value
        elif f.operator == FilterOperator.GT:
            return value is not None and value > f.value
        elif f.operator == FilterOperator.GTE:
            return value is not None and value >= f.value
        elif f.operator == FilterOperator.LT:
            return value is not None and value < f.value
        elif f.operator == FilterOperator.LTE:
            return value is not None and value <= f.value
        elif f.operator == FilterOperator.IN:
            return value in f.value
        elif f.operator == FilterOperator.NIN:
            return value is not None and value not in f.value
        elif f.operator == FilterOperator.TEXT_MATCH:
            return value is not None and f.value in value
        elif f.operator == FilterOperator.TEXT_MATCH_INSENSITIVE:
            return value is not None and f.value.lower() in value.lower()
        elif f.operator == FilterOperator.CONTAINS:
            return isinstance(value, list) and f.value in value
        elif f.operator == FilterOperator.ANY:
            return isinstance(value, list) and any(v in f.value for v in value)
        elif f.operator == FilterOperator.ALL:
            return isinstance(value, list) and all(v in value for v in f.value)

        return False
