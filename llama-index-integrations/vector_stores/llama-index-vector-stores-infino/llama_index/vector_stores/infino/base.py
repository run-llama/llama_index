"""The :class:`InfinoVectorStore` LlamaIndex vector store.

Infino is a retrieval engine that keeps your data as Apache Parquet on object
storage (local disk, S3, GCS, Azure Blob) and runs SQL, full-text (BM25), and
vector search over it from a single embedded library — no server, no cluster.

This store maps the LlamaIndex ``BasePydanticVectorStore`` contract onto one
Infino table. A single table holds the node id, the node text, the embedding,
any promoted (filterable) metadata columns, and a JSON catch-all for the rest.
Because ``stores_text`` is ``True``, that one table replaces both the vector
store and the docstore: ``VectorStoreIndex.from_vector_store(store)`` rebuilds
full nodes with no separate document store.

The same class connects to three back-ends, chosen from the ``uri`` scheme:

* ``"/path/to/dir"`` — local embedded (a directory on disk).
* ``"s3://bucket/prefix"`` / ``"gs://…"`` / ``"az://…"`` — embedded straight
  over object storage (credentials via ``storage_options``).
* ``"https://…"`` — a hosted Infino endpoint (auth via ``api_key``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

import infino

# Mirror the engine's accepted metric names so the type flows through unchanged.
Metric = Literal["cosine", "l2sq", "l2", "negdot", "dot"]

DEFAULT_TABLE_NAME = "llamaindex"
DEFAULT_METRIC: Metric = "cosine"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_ID_COLUMN = "doc_id"
DEFAULT_VECTOR_COLUMN = "embedding"
DEFAULT_REF_DOC_ID_COLUMN = "ref_doc_id"

# Column holding metadata not promoted to a declared scalar column, as JSON.
METADATA_JSON_COLUMN = "_metadata_json"
# Trailing relevance column every Infino search TVF / method appends.
SCORE_COLUMN = "score"
# The engine's internal 16-byte row id (distinct from the user id column).
ENGINE_ID_COLUMN = "_id"

# A structured metadata filter is applied as a SQL WHERE *after* the vector TVF
# ranks, so over-fetch to refill the top-k before the WHERE trims. A very
# selective filter may still under-return.
FILTER_OVERSAMPLE = 10

# Object-storage URI schemes that route to embedded-over-bucket mode.
_BUCKET_SCHEMES = (
    "s3://",
    "gs://",
    "gcs://",
    "az://",
    "azure://",
    "abfs://",
    "abfss://",
)
# URI schemes that route to the hosted (remote) transport.
_REMOTE_SCHEMES = ("http://", "https://")

# LlamaIndex filter operators → SQL comparison operators.
_SCALAR_OPERATORS = {
    FilterOperator.EQ: "=",
    FilterOperator.NE: "!=",
    FilterOperator.GT: ">",
    FilterOperator.GTE: ">=",
    FilterOperator.LT: "<",
    FilterOperator.LTE: "<=",
}
# Set-membership operators.
_IN_OPERATORS = {FilterOperator.IN: "IN", FilterOperator.NIN: "NOT IN"}

# FilterCondition → SQL boolean joiner.
_CONDITION_JOINERS = {
    FilterCondition.AND: " AND ",
    FilterCondition.OR: " OR ",
}


def _sql_str_literal(value: str) -> str:
    """Quote a string as a SQL literal, escaping embedded single quotes."""
    return "'" + value.replace("'", "''") + "'"


def _sql_value_literal(value: Any) -> str:
    """Render a filter value as a SQL literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return _sql_str_literal(str(value))


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    """Unit-normalize for the cosine metric.

    The engine's cosine contract expects unit-ish inputs: the stored rerank
    payload lives on a fixed [-1, 1] grid, so an unnormalized component clamps
    and distorts served distances. Cosine is scale-invariant, so normalizing
    changes nothing semantically — it only keeps components on the grid.
    """
    norm = sum(x * x for x in vector) ** 0.5
    if norm <= 0.0:
        return list(vector)
    return [x / norm for x in vector]


def _distance_to_similarity(metric: str, distance: float) -> float:
    """Map an Infino distance (lower = nearer) to a similarity (higher = nearer).

    LlamaIndex ranks by ``similarities``; Infino returns raw distances. The map
    is monotonic decreasing in the distance so the engine's ranking is
    preserved regardless of the exact form.
    """
    if metric == "cosine":
        # Cosine distance is 1 - cosine_similarity, in [0, 2]; clamp to [0, 1].
        return max(0.0, min(1.0, 1.0 - distance))
    if metric in ("l2", "l2sq"):
        # Squared / plain L2 is unbounded above; map monotonically into (0, 1].
        return 1.0 / (1.0 + max(0.0, distance))
    # dot / negdot: preserve ordering with a monotone-decreasing transform.
    return -distance


class InfinoVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by a single Infino table.

    Args:
        uri: Where the data lives, and how to reach it. A local directory path
            for embedded mode; an ``s3://`` / ``gs://`` / ``az://`` URI for
            embedded-over-object-storage; an ``http(s)://`` URL for a hosted
            Infino endpoint.
        table_name: The Infino table to use (created lazily on first ``add``).
        embed_dim: Embedding dimension. If omitted, it is inferred from the
            first node's embedding when the table is created. Must lie in the
            engine's supported range [16, 4096].
        metric: Distance metric to index with — ``"cosine"`` (default),
            ``"l2sq"`` / ``"l2"``, ``"negdot"`` / ``"dot"``.
        text_column / id_column / vector_column: Column names.
        metadata_columns: Metadata keys to promote to real, filterable scalar
            columns. Each entry is either a column name (``str``, stored as a
            string column) or a ``pyarrow.Field`` for an explicit type. Any
            metadata not promoted is kept in a JSON catch-all column and is
            returned on read but is not filterable. Fixed at table creation.
        api_key: Bearer key for the hosted endpoint (``http(s)://`` uris).
        api_key_file: Path to a file holding the key (read if ``api_key`` is
            not given).
        storage_options: Object-store credentials/config for bucket uris,
            passed through to the engine (e.g. ``{"aws_region": "us-east-1"}``).
        connection: A pre-built :class:`infino.Connection` to use instead of
            connecting from ``uri``.
        create_if_not_exists: Create the table on first ``add`` when it is
            missing. When ``False``, the table must already exist.
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    uri: str
    table_name: str = DEFAULT_TABLE_NAME
    metric: str = DEFAULT_METRIC
    text_column: str = DEFAULT_TEXT_COLUMN
    id_column: str = DEFAULT_ID_COLUMN
    vector_column: str = DEFAULT_VECTOR_COLUMN
    ref_doc_id_column: str = DEFAULT_REF_DOC_ID_COLUMN
    embed_dim: int | None = None
    create_if_not_exists: bool = True

    _connection: Any = PrivateAttr()
    _table: Any = PrivateAttr(default=None)
    _metadata_fields: list[pa.Field] = PrivateAttr(default_factory=list)
    _metadata_names: list[str] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        uri: str,
        table_name: str = DEFAULT_TABLE_NAME,
        *,
        embed_dim: int | None = None,
        metric: Metric = DEFAULT_METRIC,
        text_column: str = DEFAULT_TEXT_COLUMN,
        id_column: str = DEFAULT_ID_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        ref_doc_id_column: str = DEFAULT_REF_DOC_ID_COLUMN,
        metadata_columns: Sequence[str | pa.Field] = (),
        api_key: str | None = None,
        api_key_file: str | None = None,
        storage_options: dict | None = None,
        connection: Any | None = None,
        create_if_not_exists: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            uri=uri,
            table_name=table_name,
            metric=metric,
            text_column=text_column,
            id_column=id_column,
            vector_column=vector_column,
            ref_doc_id_column=ref_doc_id_column,
            embed_dim=embed_dim,
            create_if_not_exists=create_if_not_exists,
        )

        # Normalize promoted metadata columns to a list of pyarrow.Field.
        fields: list[pa.Field] = []
        for col in metadata_columns:
            if isinstance(col, pa.Field):
                fields.append(col)
            else:
                fields.append(pa.field(str(col), pa.large_utf8(), nullable=True))
        self._metadata_fields = fields
        self._metadata_names = [f.name for f in fields]

        self._connection = connection or self._connect(
            uri,
            api_key=api_key,
            api_key_file=api_key_file,
            storage_options=storage_options,
        )
        self._table = None

    @staticmethod
    def _connect(
        uri: str,
        *,
        api_key: str | None,
        api_key_file: str | None,
        storage_options: dict | None,
    ) -> Any:
        """Open the right connection for the ``uri`` scheme (the differentiator).

        The same store class reaches a local directory, an object-storage
        bucket, or a hosted endpoint — the scheme decides which.
        """
        lowered = uri.lower()
        if lowered.startswith(_REMOTE_SCHEMES):
            key = api_key
            if key is None and api_key_file is not None:
                key = Path(api_key_file).read_text().strip()
            return infino.connect(uri, api_key=key)
        if lowered.startswith(_BUCKET_SCHEMES):
            return infino.connect(uri, storage_options=storage_options)
        # Local embedded: a plain filesystem path.
        return infino.connect(uri)

    @property
    def client(self) -> Any:
        """The underlying :class:`infino.Connection`."""
        return self._connection

    # ------------------------------------------------------------------ #
    # Table lifecycle
    # ------------------------------------------------------------------ #

    def _build_schema(self, dim: int) -> pa.Schema:
        """The declared table schema: id, text, embedding, *promoted, ref, JSON."""
        return pa.schema(
            [
                pa.field(self.id_column, pa.large_utf8(), nullable=False),
                pa.field(self.text_column, pa.large_utf8(), nullable=False),
                pa.field(
                    self.vector_column, pa.list_(pa.float32(), dim), nullable=False
                ),
                *self._metadata_fields,
                pa.field(self.ref_doc_id_column, pa.large_utf8(), nullable=True),
                pa.field(METADATA_JSON_COLUMN, pa.large_utf8(), nullable=False),
            ]
        )

    def _ensure_table_for_add(self, nodes: list[BaseNode]) -> None:
        if self._table is not None:
            return
        if self.table_name in self._connection.list_tables():
            self._table = self._connection.open_table(self.table_name)
            self._adopt_existing_schema()
            return
        if not self.create_if_not_exists:
            raise ValueError(
                f"table {self.table_name!r} does not exist and "
                "create_if_not_exists=False"
            )
        dim = self.embed_dim
        if dim is None:
            embedding = nodes[0].get_embedding()
            dim = len(embedding)
        schema = self._build_schema(dim)
        indexes = (
            infino.IndexSpec()
            .fts(self.id_column)
            .fts(self.text_column)
            .vector(self.vector_column, dim, self.metric)
        )
        self._table = self._connection.create_table(self.table_name, schema, indexes)

    def _resolve_table(self) -> Any:
        if self._table is not None:
            return self._table
        if self.table_name in self._connection.list_tables():
            self._table = self._connection.open_table(self.table_name)
            self._adopt_existing_schema()
            return self._table
        raise ValueError(
            f"table {self.table_name!r} has no data yet; add nodes before querying"
        )

    def _adopt_existing_schema(self) -> None:
        """Derive promoted metadata columns from an existing table's schema.

        The stored table schema is the source of truth for which metadata keys
        are promoted scalar columns. Reading them back here means a caller that
        reopens the store on an existing table does not have to re-declare
        ``metadata_columns`` — promoted metadata still round-trips and stays
        filterable after a restart.
        """
        schema = self._table.schema()
        reserved = {
            self.id_column,
            self.text_column,
            self.vector_column,
            self.ref_doc_id_column,
            METADATA_JSON_COLUMN,
        }
        fields = [schema.field(name) for name in schema.names if name not in reserved]
        self._metadata_fields = fields
        self._metadata_names = [f.name for f in fields]

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def add(self, nodes: list[BaseNode], **add_kwargs: Any) -> list[str]:
        """Add nodes (each carrying an embedding) and return their ids."""
        if not nodes:
            return []
        self._ensure_table_for_add(nodes)

        promoted = set(self._metadata_names)
        rows: list[dict] = []
        ids: list[str] = []
        for node in nodes:
            node_id = node.node_id
            ids.append(node_id)
            embedding = node.get_embedding()
            if self.metric == "cosine":
                embedding = _l2_normalize(embedding)
            metadata = node_to_metadata_dict(node, remove_text=True)
            row: dict = {
                self.id_column: node_id,
                self.text_column: node.get_content(metadata_mode=MetadataMode.NONE)
                or "",
                self.vector_column: [float(x) for x in embedding],
                self.ref_doc_id_column: node.ref_doc_id,
                METADATA_JSON_COLUMN: json.dumps(
                    {k: v for k, v in metadata.items() if k not in promoted},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
            for name in self._metadata_names:
                row[name] = metadata.get(name)
            rows.append(row)

        self._table.append(rows)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete every node whose source document is ``ref_doc_id``."""
        table = self._resolve_table()
        table.delete(f"{self.ref_doc_id_column} = {_sql_str_literal(ref_doc_id)}")

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by id and/or by a metadata filter over promoted columns."""
        table = self._resolve_table()
        predicates: list[str] = []
        if node_ids:
            id_list = ", ".join(_sql_str_literal(i) for i in node_ids)
            predicates.append(f"{self.id_column} IN ({id_list})")
        if filters is not None:
            predicates.append(self._compile_filters(filters))
        if not predicates:
            return
        table.delete(" AND ".join(f"({p})" for p in predicates))

    def clear(self) -> None:
        """Drop the whole table."""
        if self.table_name in self._connection.list_tables():
            self._connection.drop_table(self.table_name)
        self._table = None

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def _projection(self) -> list[str]:
        return [
            self.id_column,
            self.text_column,
            *self._metadata_names,
            METADATA_JSON_COLUMN,
            SCORE_COLUMN,
        ]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Run a query, dispatching on ``query.mode``.

        DEFAULT (and MMR) → vector search; HYBRID / SEMANTIC_HYBRID →
        BM25+vector fusion in one engine call; TEXT_SEARCH / SPARSE → BM25.
        """
        table = self._resolve_table()
        k = query.similarity_top_k
        projection = self._projection()
        mode = query.mode

        # Metadata filtering is wired through the vector-search path only. The
        # BM25 / hybrid engine calls take no filter argument, so rather than
        # silently drop a filter, reject it explicitly.
        if query.filters is not None and mode in (
            VectorStoreQueryMode.HYBRID,
            VectorStoreQueryMode.SEMANTIC_HYBRID,
            VectorStoreQueryMode.TEXT_SEARCH,
            VectorStoreQueryMode.SPARSE,
        ):
            raise NotImplementedError(
                f"metadata filters are not supported with mode={mode}; "
                "filters currently apply to the default vector-search mode only"
            )

        if mode in (VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.SEMANTIC_HYBRID):
            if query.query_str is None or query.query_embedding is None:
                raise ValueError(
                    "hybrid query requires both query_str and query_embedding"
                )
            embedding = self._prepare_embedding(query.query_embedding)
            result = table.hybrid_search(
                self.text_column,
                query.query_str,
                self.vector_column,
                embedding,
                k,
                projection=projection,
            )
        elif mode in (VectorStoreQueryMode.TEXT_SEARCH, VectorStoreQueryMode.SPARSE):
            if query.query_str is None:
                raise ValueError("text/sparse query requires query_str")
            result = table.bm25_search(
                self.text_column, query.query_str, k, projection=projection
            )
        else:
            if query.query_embedding is None:
                raise ValueError("vector query requires query_embedding")
            embedding = self._prepare_embedding(query.query_embedding)
            if query.filters is not None:
                result = self._filtered_vector_search(
                    embedding, k, query.filters, projection
                )
            else:
                result = table.vector_search(
                    self.vector_column, embedding, k, projection=projection
                )

        return self._result_to_query_result(result)

    def _prepare_embedding(self, embedding: Sequence[float]) -> list[float]:
        vector = [float(x) for x in embedding]
        if self.metric == "cosine":
            vector = _l2_normalize(vector)
        return vector

    def _filtered_vector_search(
        self,
        embedding: list[float],
        k: int,
        filters: MetadataFilters,
        projection: list[str],
    ) -> Any:
        """Vector search with a structured metadata filter.

        The filter is a post-rank SQL ``WHERE`` over promoted columns, so it is
        expressed through the ``vector_search`` table-valued function: over-fetch,
        filter, then trim to ``k``.
        """
        where = self._compile_filters(filters)
        columns = ", ".join(projection)
        vector_literal = ",".join(str(float(x)) for x in embedding)
        table_lit = _sql_str_literal(self.table_name)
        column_lit = _sql_str_literal(self.vector_column)
        query_lit = _sql_str_literal(vector_literal)
        sql = (
            f"SELECT {columns} FROM vector_search("
            f"{table_lit}, {column_lit}, {query_lit}, {k * FILTER_OVERSAMPLE}) "
            f"WHERE {where} ORDER BY {SCORE_COLUMN} ASC LIMIT {k}"
        )
        return self._connection.query_sql(sql)

    def _compile_filters(self, filters: MetadataFilters) -> str:
        """Compile a ``MetadataFilters`` tree into a SQL ``WHERE`` clause."""
        condition = filters.condition or FilterCondition.AND
        parts: list[str] = []
        for f in filters.filters:
            if isinstance(f, MetadataFilters):
                parts.append(f"({self._compile_filters(f)})")
            else:
                parts.append(self._compile_filter(f))
        if condition == FilterCondition.NOT:
            joined = " AND ".join(parts)
            return f"NOT ({joined})"
        joiner = _CONDITION_JOINERS.get(condition, " AND ")
        return joiner.join(parts)

    def _compile_filter(self, f: MetadataFilter) -> str:
        if f.key not in self._metadata_names:
            raise ValueError(
                f"cannot filter on {f.key!r}: not a promoted metadata column "
                f"(promoted: {sorted(self._metadata_names)}). Declare it in "
                "metadata_columns at construction to make it filterable."
            )
        op = f.operator
        if op in _SCALAR_OPERATORS:
            return f"{f.key} {_SCALAR_OPERATORS[op]} {_sql_value_literal(f.value)}"
        if op in _IN_OPERATORS:
            items = ", ".join(_sql_value_literal(v) for v in f.value)
            return f"{f.key} {_IN_OPERATORS[op]} ({items})"
        raise ValueError(f"unsupported filter operator {op!r}")

    def _result_to_query_result(self, table: pa.Table) -> VectorStoreQueryResult:
        n = table.num_rows
        if n == 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        ids = columns.get(self.id_column, [None] * n)
        texts = columns.get(self.text_column, [""] * n)
        metadata_json = columns.get(METADATA_JSON_COLUMN, [None] * n)
        scores = columns.get(SCORE_COLUMN, [None] * n)

        reserved = {
            self.id_column,
            self.text_column,
            self.ref_doc_id_column,
            METADATA_JSON_COLUMN,
            SCORE_COLUMN,
            ENGINE_ID_COLUMN,
        }
        promoted_present = [c for c in columns if c not in reserved]

        nodes: list[BaseNode] = []
        similarities: list[float] = []
        out_ids: list[str] = []
        for i in range(n):
            raw = metadata_json[i]
            metadata = json.loads(raw) if raw else {}
            for name in promoted_present:
                value = columns[name][i]
                if value is not None:
                    metadata[name] = value
            text = texts[i] or ""
            try:
                node = metadata_dict_to_node(metadata, text=text)
            except Exception:
                node = TextNode(id_=ids[i], text=text, metadata=metadata)
            nodes.append(node)
            out_ids.append(ids[i])
            score = scores[i]
            similarities.append(
                _distance_to_similarity(self.metric, score)
                if score is not None
                else 0.0
            )
        return VectorStoreQueryResult(
            nodes=nodes, similarities=similarities, ids=out_ids
        )

    # ------------------------------------------------------------------ #
    # Async — delegate to the sync engine calls.
    # ------------------------------------------------------------------ #

    async def async_add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        return self.add(nodes, **kwargs)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self.delete(ref_doc_id, **delete_kwargs)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        return self.query(query, **kwargs)
