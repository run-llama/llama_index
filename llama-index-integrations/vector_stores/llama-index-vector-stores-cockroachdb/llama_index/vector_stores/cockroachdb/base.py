"""CockroachDB vector store backed by native VECTOR + C-SPANN index."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Literal,
    NamedTuple,
)

import asyncpg  # noqa: F401  (asyncpg engine driver, ensures dep is installed)
import psycopg2  # noqa: F401
import sqlalchemy
import sqlalchemy.ext.asyncio
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
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
from sqlalchemy import select, text
from sqlalchemy.sql.selectable import Select

DEFAULT_MMR_PREFETCH_FACTOR = 4.0

CRDBType = Literal[
    "text",
    "int",
    "integer",
    "numeric",
    "float",
    "double precision",
    "boolean",
    "date",
    "timestamp",
    "uuid",
]


class DBEmbeddingRow(NamedTuple):
    node_id: str
    text: str
    metadata: dict
    custom_fields: dict
    similarity: float


_logger = logging.getLogger(__name__)


def _normalize_distance_metric(metric: str) -> str:
    metric = metric.lower().strip()
    if metric in {"cosine", "<=>"}:
        return "<=>"
    if metric in {"l2", "euclidean", "<->"}:
        return "<->"
    if metric in {"ip", "inner_product", "negative_inner_product", "<#>"}:
        return "<#>"
    raise ValueError(f"Unknown distance metric {metric!r}. Use 'cosine', 'l2', or 'inner_product'.")


def _get_data_model(
    base: type,
    table_name: str,
    schema_name: str,
    embed_dim: int,
    indexed_metadata_keys: set[tuple[str, CRDBType]] | None,
) -> Any:
    """Build a dynamic SQLAlchemy model with a CockroachDB native VECTOR column.

    Note: SQLAlchemy has no built-in VECTOR type, so we use a UserDefinedType
    that emits ``VECTOR(dim)`` in DDL and forwards Python lists/tuples as
    pgwire parameters cast with ``::vector`` at bind time.
    """
    from sqlalchemy import (
        Boolean,
        Column,
        Date,
        DateTime,
        Float,
        Integer,
        Numeric,
        String,
    )
    from sqlalchemy import (
        cast as sql_cast,
    )
    from sqlalchemy.dialects.postgresql import JSONB, UUID, VARCHAR
    from sqlalchemy.schema import Index
    from sqlalchemy.types import UserDefinedType

    pg_type_map = {
        "text": String,
        "int": Integer,
        "integer": Integer,
        "numeric": Numeric,
        "float": Float,
        "double precision": Float,
        "boolean": Boolean,
        "date": Date,
        "timestamp": DateTime,
        "uuid": UUID,
    }

    class CRDBVector(UserDefinedType):
        """CockroachDB native VECTOR(dim) column type."""

        cache_ok = True

        def __init__(self, dim: int) -> None:
            self.dim = dim

        def get_col_spec(self, **kw: Any) -> str:
            return f"VECTOR({self.dim})"

        def bind_processor(self, dialect: Any) -> Callable[[Any], Any]:
            def process(value: Any) -> str | None:
                if value is None:
                    return None
                if hasattr(value, "tolist"):
                    value = value.tolist()
                return "[" + ",".join(repr(float(x)) for x in value) + "]"

            return process

        def result_processor(self, dialect: Any, coltype: Any) -> Callable[[Any], Any]:
            def process(value: Any) -> list[float] | None:
                if value is None:
                    return None
                if isinstance(value, (list, tuple)):
                    return [float(x) for x in value]
                return [float(x) for x in value.strip("[]").split(",") if x]

            return process

        def bind_expression(self, bindvalue: Any) -> Any:
            """Force an inline ::VECTOR(n) cast on every parameter.

            Required for asyncpg, which prepares statements eagerly and
            cannot infer placeholder types from outer SELECT casts.
            """
            return sql_cast(bindvalue, self)

        class comparator_factory(UserDefinedType.Comparator):  # type: ignore[misc]
            def cosine_distance(self, other: Any) -> Any:
                return self.op("<=>", return_type=Float())(other)

            def l2_distance(self, other: Any) -> Any:
                return self.op("<->", return_type=Float())(other)

            def negative_inner_product(self, other: Any) -> Any:
                return self.op("<#>", return_type=Float())(other)

    indexed_metadata_keys = indexed_metadata_keys or set()
    for key, crdb_type in indexed_metadata_keys:
        if crdb_type not in pg_type_map:
            raise ValueError(
                f"Invalid type {crdb_type} for key {key}. "
                f"Must be one of {sorted(pg_type_map.keys())}"
            )

    tablename = f"data_{table_name}"
    class_name = f"Data{table_name}"
    indexname = f"{table_name}_idx"

    from sqlalchemy import cast, column
    from sqlalchemy.dialects.postgresql import BIGINT

    btree_indices = [
        Index(
            f"{indexname}_{key}_{crdb_type.replace(' ', '_')}",
            cast(column("metadata_").op("->>")(key), pg_type_map[crdb_type]),
            postgresql_using="btree",
        )
        for key, crdb_type in indexed_metadata_keys
    ]

    class AbstractData(base):  # type: ignore[misc, valid-type]
        __abstract__ = True
        id = Column(BIGINT, primary_key=True, autoincrement=True)
        text = Column(VARCHAR, nullable=False)
        metadata_ = Column(JSONB)
        node_id = Column(VARCHAR)
        embedding = Column(CRDBVector(embed_dim))

    return type(
        class_name,
        (AbstractData,),
        {
            "__tablename__": tablename,
            "__table_args__": (*btree_indices, {"schema": schema_name}),
        },
    )


class CockroachDBVectorStore(BasePydanticVectorStore):
    """CockroachDB-backed vector store using native VECTOR + C-SPANN index.

    Examples:
        ```python
        from llama_index.vector_stores.cockroachdb import CockroachDBVectorStore

        store = CockroachDBVectorStore.from_params(
            host="localhost",
            port=26257,
            database="defaultdb",
            user="root",
            table_name="my_index",
            embed_dim=1536,
            distance_metric="cosine",
            cspann_kwargs={"min_partition_size": 16, "max_partition_size": 128},
        )
        ```
    """

    stores_text: bool = True
    flat_metadata: bool = False
    is_embedding_query: bool = True

    connection_string: str
    async_connection_string: str
    table_name: str
    schema_name: str
    embed_dim: int
    distance_metric: str
    perform_setup: bool
    debug: bool
    create_engine_kwargs: dict
    cspann_kwargs: dict[str, Any] | None
    vector_search_beam_size: int | None
    indexed_metadata_keys: set[tuple[str, CRDBType]] | None
    enable_feature_setting: bool
    initialization_fail_on_error: bool

    _base: Any = PrivateAttr()
    _table_class: Any = PrivateAttr()
    _engine: sqlalchemy.engine.Engine | None = PrivateAttr(default=None)
    _session: Any = PrivateAttr(default=None)
    _async_engine: sqlalchemy.ext.asyncio.AsyncEngine | None = PrivateAttr(default=None)
    _async_session: Any = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)
    _customize_query_fn: Callable[[Select, Any, Any], Select] | None = PrivateAttr(default=None)
    _distance_op: str = PrivateAttr(default="<=>")

    def __init__(
        self,
        connection_string: str | sqlalchemy.engine.URL,
        async_connection_string: str | sqlalchemy.engine.URL,
        table_name: str = "llamaindex",
        schema_name: str = "public",
        embed_dim: int = 1536,
        distance_metric: str = "cosine",
        perform_setup: bool = True,
        debug: bool = False,
        create_engine_kwargs: dict[str, Any] | None = None,
        cspann_kwargs: dict[str, Any] | None = None,
        vector_search_beam_size: int | None = None,
        indexed_metadata_keys: set[tuple[str, CRDBType]] | None = None,
        enable_feature_setting: bool = True,
        initialization_fail_on_error: bool = False,
        engine: sqlalchemy.engine.Engine | None = None,
        async_engine: sqlalchemy.ext.asyncio.AsyncEngine | None = None,
        customize_query_fn: Callable[[Select, Any, Any], Select] | None = None,
    ) -> None:
        table_name = (table_name or "llamaindex").lower()
        schema_name = (schema_name or "public").lower()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
            raise ValueError(f"Invalid table_name: {table_name}")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", schema_name):
            raise ValueError(f"Invalid schema_name: {schema_name}")

        from sqlalchemy.orm import declarative_base

        super().__init__(
            connection_string=str(connection_string),
            async_connection_string=str(async_connection_string),
            table_name=table_name,
            schema_name=schema_name,
            embed_dim=embed_dim,
            distance_metric=distance_metric,
            perform_setup=perform_setup,
            debug=debug,
            create_engine_kwargs=create_engine_kwargs or {},
            cspann_kwargs=cspann_kwargs,
            vector_search_beam_size=vector_search_beam_size,
            indexed_metadata_keys=indexed_metadata_keys,
            enable_feature_setting=enable_feature_setting,
            initialization_fail_on_error=initialization_fail_on_error,
        )
        self._distance_op = _normalize_distance_metric(distance_metric)
        self._base = declarative_base()
        self._table_class = _get_data_model(
            self._base,
            table_name=table_name,
            schema_name=schema_name,
            embed_dim=embed_dim,
            indexed_metadata_keys=indexed_metadata_keys,
        )

        if (engine is None) != (async_engine is None):
            raise ValueError("engine and async_engine must both be provided or both be None")
        if engine is not None:
            self._engine = engine
            self._async_engine = async_engine
        self._customize_query_fn = customize_query_fn

    @classmethod
    def class_name(cls) -> str:
        return "CockroachDBVectorStore"

    @classmethod
    def from_params(
        cls,
        host: str | None = None,
        port: int | str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        sslmode: str = "verify-full",
        sslrootcert: str | None = None,
        table_name: str = "llamaindex",
        schema_name: str = "public",
        connection_string: str | sqlalchemy.engine.URL | None = None,
        async_connection_string: str | sqlalchemy.engine.URL | None = None,
        embed_dim: int = 1536,
        distance_metric: str = "cosine",
        perform_setup: bool = True,
        debug: bool = False,
        create_engine_kwargs: dict[str, Any] | None = None,
        cspann_kwargs: dict[str, Any] | None = None,
        vector_search_beam_size: int | None = None,
        indexed_metadata_keys: set[tuple[str, CRDBType]] | None = None,
        enable_feature_setting: bool = True,
        customize_query_fn: Callable[[Select, Any, Any], Select] | None = None,
    ) -> CockroachDBVectorStore:
        """Construct a CockroachDBVectorStore from individual connection params.

        Uses the ``cockroachdb+psycopg2`` / ``cockroachdb+asyncpg`` dialects
        provided by ``sqlalchemy-cockroachdb`` so transaction retries on
        SERIALIZATION_FAILURE work transparently.
        """
        if connection_string is None:
            qs = ""
            params: list[str] = []
            if sslmode:
                params.append(f"sslmode={sslmode}")
            if sslrootcert:
                params.append(f"sslrootcert={sslrootcert}")
            if params:
                qs = "?" + "&".join(params)
            connection_string = (
                f"cockroachdb+psycopg2://{user}:{password}@{host}:{port}/{database}{qs}"
            )
        if async_connection_string is None:
            qs = ""
            params = []
            if sslmode and sslmode != "disable":
                params.append("ssl=true")
            if params:
                qs = "?" + "&".join(params)
            async_connection_string = (
                f"cockroachdb+asyncpg://{user}:{password}@{host}:{port}/{database}{qs}"
            )

        return cls(
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=table_name,
            schema_name=schema_name,
            embed_dim=embed_dim,
            distance_metric=distance_metric,
            perform_setup=perform_setup,
            debug=debug,
            create_engine_kwargs=create_engine_kwargs,
            cspann_kwargs=cspann_kwargs,
            vector_search_beam_size=vector_search_beam_size,
            indexed_metadata_keys=indexed_metadata_keys,
            enable_feature_setting=enable_feature_setting,
            customize_query_fn=customize_query_fn,
        )

    @property
    def client(self) -> Any:
        return self._engine if self._is_initialized else None

    def _connect(self) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        if self._engine is None:
            self._engine = create_engine(
                self.connection_string, echo=self.debug, **self.create_engine_kwargs
            )
        self._session = sessionmaker(self._engine, expire_on_commit=False)
        if self._async_engine is None:
            self._async_engine = create_async_engine(
                self.async_connection_string, **self.create_engine_kwargs
            )
        self._async_session = sessionmaker(
            self._async_engine, class_=AsyncSession, expire_on_commit=False
        )

    def _create_schema_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}"))

    def _create_tables_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            self._table_class.__table__.create(session.connection(), checkfirst=True)

    def _enable_vector_feature(self) -> None:
        """Required on CRDB v25.2+ before the first VECTOR INDEX is created.

        Cluster settings must run outside a transaction, so we use a fresh
        AUTOCOMMIT connection rather than the session-bound transaction.
        """
        with self._engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT").execute(
                text("SET CLUSTER SETTING feature.vector_index.enabled = true")
            )

    def _create_cspann_index(self) -> None:
        """Create the C-SPANN vector index. Idempotent via IF NOT EXISTS."""
        if not self.cspann_kwargs:
            return
        index_name = f"{self._table_class.__tablename__}_embedding_cspann_idx"
        with_clauses: list[str] = []
        if "min_partition_size" in self.cspann_kwargs:
            with_clauses.append(
                f"min_partition_size = {int(self.cspann_kwargs['min_partition_size'])}"
            )
        if "max_partition_size" in self.cspann_kwargs:
            with_clauses.append(
                f"max_partition_size = {int(self.cspann_kwargs['max_partition_size'])}"
            )
        with_sql = f" WITH ({', '.join(with_clauses)})" if with_clauses else ""
        op_class = {
            "<=>": "vector_cosine_ops",
            "<->": "vector_l2_ops",
            "<#>": "vector_ip_ops",
        }[self._distance_op]
        stmt = (
            f"CREATE VECTOR INDEX IF NOT EXISTS {index_name} "
            f"ON {self.schema_name}.{self._table_class.__tablename__} "
            f"(embedding {op_class}){with_sql}"
        )
        with self._session() as session, session.begin():
            session.execute(text(stmt))

    def _initialize(self) -> None:
        if self._is_initialized:
            return
        self._connect()
        if self.perform_setup:
            fail = self.initialization_fail_on_error
            try:
                self._create_schema_if_not_exists()
            except Exception as e:
                _logger.warning(f"CRDB setup: schema: {e}")
                if fail:
                    raise
            try:
                self._create_tables_if_not_exists()
            except Exception as e:
                _logger.warning(f"CRDB setup: table: {e}")
                if fail:
                    raise
            if self.enable_feature_setting and self.cspann_kwargs:
                try:
                    self._enable_vector_feature()
                except Exception as e:
                    _logger.warning(f"CRDB setup: feature flag: {e}")
                    if fail:
                        raise
            try:
                self._create_cspann_index()
            except Exception as e:
                _logger.warning(f"CRDB setup: vector index: {e}")
                if fail:
                    raise
        self._is_initialized = True

    def _node_to_row(self, node: BaseNode) -> Any:
        return self._table_class(
            node_id=node.node_id,
            embedding=node.get_embedding(),
            text=node.get_content(metadata_mode=MetadataMode.NONE),
            metadata_=node_to_metadata_dict(
                node, remove_text=True, flat_metadata=self.flat_metadata
            ),
        )

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        self._initialize()
        ids: list[str] = []
        with self._session() as session, session.begin():
            for node in nodes:
                session.add(self._node_to_row(node))
                ids.append(node.node_id)
        return ids

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        self._initialize()
        ids: list[str] = []
        async with self._async_session() as session, session.begin():
            for node in nodes:
                session.add(self._node_to_row(node))
                ids.append(node.node_id)
        return ids

    def _to_sql_operator(self, op: FilterOperator) -> str:
        return {
            FilterOperator.EQ: "=",
            FilterOperator.GT: ">",
            FilterOperator.LT: "<",
            FilterOperator.NE: "!=",
            FilterOperator.GTE: ">=",
            FilterOperator.LTE: "<=",
            FilterOperator.IN: "IN",
            FilterOperator.NIN: "NOT IN",
            FilterOperator.CONTAINS: "@>",
            FilterOperator.TEXT_MATCH: "LIKE",
            FilterOperator.TEXT_MATCH_INSENSITIVE: "ILIKE",
            FilterOperator.IS_EMPTY: "IS NULL",
        }.get(op, "=")

    def _build_filter_clause(self, f: MetadataFilter) -> Any:
        op = f.operator
        if op in (FilterOperator.IN, FilterOperator.NIN):
            joined = ", ".join(f"'{v}'" for v in f.value)
            return text(f"metadata_->>'{f.key}' {self._to_sql_operator(op)} ({joined})")
        if op == FilterOperator.CONTAINS:
            return text(f"metadata_->'{f.key}' @> '[\"{f.value}\"]'::jsonb")
        if op in (FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE):
            return text(f"metadata_->>'{f.key}' {self._to_sql_operator(op)} '%{f.value}%'")
        if op == FilterOperator.IS_EMPTY:
            return text(f"metadata_->>'{f.key}' IS NULL")
        try:
            return text(
                f"(metadata_->>'{f.key}')::float {self._to_sql_operator(op)} {float(f.value)}"
            )
        except (TypeError, ValueError):
            return text(f"metadata_->>'{f.key}' {self._to_sql_operator(op)} '{f.value}'")

    def _apply_filters(self, filters: MetadataFilters) -> Any:
        from sqlalchemy.sql import and_, or_

        combiner = {"and": and_, "or": or_}.get((filters.condition or "and").lower(), and_)
        return combiner(
            *(
                self._apply_filters(f)
                if isinstance(f, MetadataFilters)
                else self._build_filter_clause(f)
                for f in filters.filters
            )
        )

    def _apply_filters_and_limit(
        self, stmt: Select, limit: int, filters: MetadataFilters | None
    ) -> Select:
        if filters:
            stmt = stmt.where(self._apply_filters(filters))
        return stmt.limit(limit)

    def _distance_expr(self, embedding: list[float]) -> Any:
        col = self._table_class.embedding
        if self._distance_op == "<=>":
            return col.cosine_distance(embedding)
        if self._distance_op == "<->":
            return col.l2_distance(embedding)
        return col.negative_inner_product(embedding)

    def _build_query(
        self,
        embedding: list[float],
        limit: int,
        filters: MetadataFilters | None,
        with_embedding: bool = False,
        **kwargs: Any,
    ) -> Select:
        cols = [
            self._table_class.id,
            self._table_class.node_id,
            self._table_class.text,
            self._table_class.metadata_,
            self._distance_expr(embedding).label("distance"),
        ]
        if with_embedding:
            cols.insert(-1, self._table_class.embedding)
        stmt = select(*cols).order_by(text("distance asc"))
        if self._customize_query_fn is not None:
            stmt = self._customize_query_fn(stmt, self._table_class, **kwargs)
        return self._apply_filters_and_limit(stmt, limit, filters)

    def _session_pre_stmts(self, **kwargs: Any) -> list[tuple[str, dict]]:
        out: list[tuple[str, dict]] = []
        beam = kwargs.get("vector_search_beam_size") or self.vector_search_beam_size
        if beam is not None:
            out.append(
                (
                    "SET LOCAL vector_search_beam_size = :beam",
                    {"beam": int(beam)},
                )
            )
        return out

    @staticmethod
    def _row_to_dbrow(item: Any, dist_attr: str = "distance") -> DBEmbeddingRow:
        d = item._asdict()
        distance = d.get(dist_attr)
        return DBEmbeddingRow(
            node_id=item.node_id,
            text=item.text,
            metadata=item.metadata_,
            custom_fields={
                k: v
                for k, v in d.items()
                if k
                not in {
                    "id",
                    "node_id",
                    "text",
                    "metadata_",
                    "distance",
                    "embedding",
                }
            },
            similarity=(1 - distance) if distance is not None else 0.0,
        )

    def _query_default(
        self,
        embedding: list[float],
        limit: int,
        filters: MetadataFilters | None,
        **kwargs: Any,
    ) -> list[DBEmbeddingRow]:
        stmt = self._build_query(embedding, limit, filters, **kwargs)
        with self._session() as session, session.begin():
            for sql, params in self._session_pre_stmts(**kwargs):
                session.execute(text(sql), params)
            res = session.execute(stmt)
            return [self._row_to_dbrow(item) for item in res.all()]

    async def _aquery_default(
        self,
        embedding: list[float],
        limit: int,
        filters: MetadataFilters | None,
        **kwargs: Any,
    ) -> list[DBEmbeddingRow]:
        stmt = self._build_query(embedding, limit, filters, **kwargs)
        async with self._async_session() as session, session.begin():
            for sql, params in self._session_pre_stmts(**kwargs):
                await session.execute(text(sql), params)
            res = await session.execute(stmt)
            return [self._row_to_dbrow(item) for item in res.all()]

    def _query_with_embeddings(
        self,
        embedding: list[float],
        limit: int,
        filters: MetadataFilters | None,
        **kwargs: Any,
    ) -> list[tuple[DBEmbeddingRow, list[float]]]:
        stmt = self._build_query(embedding, limit, filters, with_embedding=True, **kwargs)
        with self._session() as session, session.begin():
            for sql, params in self._session_pre_stmts(**kwargs):
                session.execute(text(sql), params)
            res = session.execute(stmt)
            out: list[tuple[DBEmbeddingRow, list[float]]] = []
            for item in res.all():
                emb = list(item.embedding) if item.embedding is not None else []
                out.append((self._row_to_dbrow(item), emb))
            return out

    async def _aquery_with_embeddings(
        self,
        embedding: list[float],
        limit: int,
        filters: MetadataFilters | None,
        **kwargs: Any,
    ) -> list[tuple[DBEmbeddingRow, list[float]]]:
        stmt = self._build_query(embedding, limit, filters, with_embedding=True, **kwargs)
        async with self._async_session() as session, session.begin():
            for sql, params in self._session_pre_stmts(**kwargs):
                await session.execute(text(sql), params)
            res = await session.execute(stmt)
            out: list[tuple[DBEmbeddingRow, list[float]]] = []
            for item in res.all():
                emb = list(item.embedding) if item.embedding is not None else []
                out.append((self._row_to_dbrow(item), emb))
            return out

    def _prepare_mmr(self, query: VectorStoreQuery, **kwargs: Any) -> tuple[int, float | None]:
        if query.query_embedding is None:
            raise ValueError("MMR query requires query_embedding")
        if (
            kwargs.get("mmr_prefetch_factor") is not None
            and kwargs.get("mmr_prefetch_k") is not None
        ):
            raise ValueError("mmr_prefetch_factor and mmr_prefetch_k are mutually exclusive")
        if kwargs.get("mmr_prefetch_k") is not None:
            prefetch_k = int(kwargs["mmr_prefetch_k"])
        else:
            prefetch_k = int(
                query.similarity_top_k
                * kwargs.get("mmr_prefetch_factor", DEFAULT_MMR_PREFETCH_FACTOR)
            )
        prefetch_k = max(prefetch_k, query.similarity_top_k)
        threshold = (
            query.mmr_threshold if query.mmr_threshold is not None else kwargs.get("mmr_threshold")
        )
        if threshold is not None and not (0 <= threshold <= 1):
            raise ValueError(f"mmr_threshold must be between 0 and 1, got {threshold}")
        return prefetch_k, threshold

    def _mmr_rerank(
        self,
        query: VectorStoreQuery,
        prefetched: list[tuple[DBEmbeddingRow, list[float]]],
        threshold: float | None,
    ) -> VectorStoreQueryResult | None:
        if not prefetched:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        embeddings = [emb for _, emb in prefetched]
        ids = [row.node_id for row, _ in prefetched]
        valid = [i for i, e in enumerate(embeddings) if e]
        if not valid:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        valid_embeddings = [embeddings[i] for i in valid]
        valid_ids = [ids[i] for i in valid]
        if len(valid_embeddings) < query.similarity_top_k:
            return None
        sims, picked_ids = get_top_k_mmr_embeddings(
            query_embedding=query.query_embedding,
            embeddings=valid_embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=valid_ids,
            mmr_threshold=threshold,
        )
        row_map = {row.node_id: row for row, _ in prefetched}
        ordered = [
            DBEmbeddingRow(
                node_id=row_map[nid].node_id,
                text=row_map[nid].text,
                metadata=row_map[nid].metadata,
                custom_fields=row_map[nid].custom_fields,
                similarity=sim,
            )
            for sim, nid in zip(sims, picked_ids, strict=False)
            if nid in row_map
        ]
        return self._rows_to_result(ordered)

    def _mmr_query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        prefetch_k, threshold = self._prepare_mmr(query, **kwargs)
        db_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"mmr_prefetch_factor", "mmr_prefetch_k", "mmr_threshold"}
        }
        prefetched = self._query_with_embeddings(
            query.query_embedding, prefetch_k, query.filters, **db_kwargs
        )
        out = self._mmr_rerank(query, prefetched, threshold)
        if out is not None:
            return out
        rows = self._query_default(
            query.query_embedding, query.similarity_top_k, query.filters, **db_kwargs
        )
        return self._rows_to_result(rows)

    async def _amrr_query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        prefetch_k, threshold = self._prepare_mmr(query, **kwargs)
        db_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in {"mmr_prefetch_factor", "mmr_prefetch_k", "mmr_threshold"}
        }
        prefetched = await self._aquery_with_embeddings(
            query.query_embedding, prefetch_k, query.filters, **db_kwargs
        )
        out = self._mmr_rerank(query, prefetched, threshold)
        if out is not None:
            return out
        rows = await self._aquery_default(
            query.query_embedding, query.similarity_top_k, query.filters, **db_kwargs
        )
        return self._rows_to_result(rows)

    def _rows_to_result(self, rows: list[DBEmbeddingRow]) -> VectorStoreQueryResult:
        nodes: list[BaseNode] = []
        sims: list[float] = []
        ids: list[str] = []
        for row in rows:
            try:
                node = metadata_dict_to_node(row.metadata)
                node.set_content(str(row.text))
            except Exception:
                node = TextNode(id_=row.node_id, text=row.text, metadata=row.metadata)
            if row.custom_fields:
                node.metadata["custom_fields"] = row.custom_fields
            nodes.append(node)
            sims.append(row.similarity)
            ids.append(row.node_id)
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        self._initialize()
        if query.mode == VectorStoreQueryMode.DEFAULT:
            rows = self._query_default(
                query.query_embedding, query.similarity_top_k, query.filters, **kwargs
            )
            return self._rows_to_result(rows)
        if query.mode == VectorStoreQueryMode.MMR:
            return self._mmr_query(query, **kwargs)
        raise NotImplementedError(
            f"CockroachDBVectorStore does not support query mode {query.mode}. "
            f"Supported modes: DEFAULT, MMR. (HYBRID/SPARSE/TEXT_SEARCH require a "
            f"tsvector-equivalent feature not yet available in CockroachDB.)"
        )

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        self._initialize()
        if query.mode == VectorStoreQueryMode.DEFAULT:
            rows = await self._aquery_default(
                query.query_embedding, query.similarity_top_k, query.filters, **kwargs
            )
            return self._rows_to_result(rows)
        if query.mode == VectorStoreQueryMode.MMR:
            return await self._amrr_query(query, **kwargs)
        raise NotImplementedError(
            f"CockroachDBVectorStore does not support query mode {query.mode}."
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        from sqlalchemy import delete

        self._initialize()
        with self._session() as session, session.begin():
            stmt = delete(self._table_class).where(
                self._table_class.metadata_["ref_doc_id"].astext == ref_doc_id
            )
            session.execute(stmt)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        from sqlalchemy import delete

        self._initialize()
        async with self._async_session() as session, session.begin():
            stmt = delete(self._table_class).where(
                self._table_class.metadata_["ref_doc_id"].astext == ref_doc_id
            )
            await session.execute(stmt)

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        if not node_ids and not filters:
            return
        from sqlalchemy import delete

        self._initialize()
        with self._session() as session, session.begin():
            stmt = delete(self._table_class)
            if node_ids:
                stmt = stmt.where(self._table_class.node_id.in_(node_ids))
            if filters:
                stmt = stmt.where(self._apply_filters(filters))
            session.execute(stmt)

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        if not node_ids and not filters:
            return
        from sqlalchemy import delete

        self._initialize()
        async with self._async_session() as session, session.begin():
            stmt = delete(self._table_class)
            if node_ids:
                stmt = stmt.where(self._table_class.node_id.in_(node_ids))
            if filters:
                stmt = stmt.where(self._apply_filters(filters))
            await session.execute(stmt)

    def clear(self) -> None:
        from sqlalchemy import delete

        self._initialize()
        with self._session() as session, session.begin():
            session.execute(delete(self._table_class))

    async def aclear(self) -> None:
        from sqlalchemy import delete

        self._initialize()
        async with self._async_session() as session, session.begin():
            await session.execute(delete(self._table_class))

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided")
        self._initialize()
        stmt = select(
            self._table_class.node_id,
            self._table_class.text,
            self._table_class.metadata_,
            self._table_class.embedding,
        )
        if node_ids:
            stmt = stmt.where(self._table_class.node_id.in_(node_ids))
        if filters:
            stmt = stmt.where(self._apply_filters(filters))
        out: list[BaseNode] = []
        with self._session() as session, session.begin():
            for item in session.execute(stmt).fetchall():
                try:
                    node = metadata_dict_to_node(item.metadata_)
                    node.set_content(str(item.text))
                    node.embedding = item.embedding
                except Exception:
                    node = TextNode(
                        id_=item.node_id,
                        text=item.text,
                        metadata=item.metadata_,
                        embedding=item.embedding,
                    )
                out.append(node)
        return out

    async def aget_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        if node_ids is None and filters is None:
            raise ValueError("Either node_ids or filters must be provided")
        self._initialize()
        stmt = select(
            self._table_class.node_id,
            self._table_class.text,
            self._table_class.metadata_,
            self._table_class.embedding,
        )
        if node_ids:
            stmt = stmt.where(self._table_class.node_id.in_(node_ids))
        if filters:
            stmt = stmt.where(self._apply_filters(filters))
        out: list[BaseNode] = []
        async with self._async_session() as session, session.begin():
            res = await session.execute(stmt)
            for item in res.fetchall():
                try:
                    node = metadata_dict_to_node(item.metadata_)
                    node.set_content(str(item.text))
                    node.embedding = item.embedding
                except Exception:
                    node = TextNode(
                        id_=item.node_id,
                        text=item.text,
                        metadata=item.metadata_,
                        embedding=item.embedding,
                    )
                out.append(node)
        return out

    async def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
        if self._async_engine is not None:
            await self._async_engine.dispose()
        self._is_initialized = False
