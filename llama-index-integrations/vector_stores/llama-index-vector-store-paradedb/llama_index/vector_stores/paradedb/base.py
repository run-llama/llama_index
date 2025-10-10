import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from llama_index.core.vector_stores.types import MetadataFilters

import sqlalchemy
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.vector_stores.types import VectorStoreQuery
from sqlalchemy.sql.selectable import Select

from llama_index.vector_stores.postgres.base import (
    PGVectorStore,
    DBEmbeddingRow,
    PGType,
)


_logger = logging.getLogger(__name__)


def get_bm25_data_model(
    base: Any,
    index_name: str,
    schema_name: str,
    hybrid_search: bool,
    text_search_config: str,
    cache_okay: bool,
    embed_dim: int = 1536,
    use_jsonb: bool = False,
    use_halfvec: bool = False,
    indexed_metadata_keys: Optional[Set[Tuple[str, PGType]]] = None,
) -> Any:
    """
    Creates a data model optimized for BM25 search (without text_search_tsv column).
    """
    from pgvector.sqlalchemy import Vector, HALFVEC
    from sqlalchemy import Column
    from sqlalchemy.dialects.postgresql import BIGINT, JSON, JSONB, VARCHAR
    from sqlalchemy import cast, column, String, Integer, Numeric, Float, Boolean, Date, DateTime
    from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, UUID
    from sqlalchemy.schema import Index

    pg_type_map = {
        "text": String,
        "int": Integer,
        "integer": Integer,
        "numeric": Numeric,
        "float": Float,
        "double precision": DOUBLE_PRECISION,
        "boolean": Boolean,
        "date": Date,
        "timestamp": DateTime,
        "uuid": UUID,
    }

    indexed_metadata_keys = indexed_metadata_keys or set()
    
    for key, pg_type in indexed_metadata_keys:
        if pg_type not in pg_type_map:
            raise ValueError(
                f"Invalid type {pg_type} for key {key}. "
                f"Must be one of {list(pg_type_map.keys())}"
            )

    tablename = f"data_{index_name}"
    class_name = f"Data{index_name}"
    indexname = f"{index_name}_idx"

    metadata_dtype = JSONB if use_jsonb else JSON
    embedding_col = Column(HALFVEC(embed_dim)) if use_halfvec else Column(Vector(embed_dim))

    metadata_indices = [
        Index(
            f"{indexname}_{key}_{pg_type.replace(' ', '_')}",
            cast(column("metadata_").op("->>")(key), pg_type_map[pg_type]),
            postgresql_using="btree",
        )
        for key, pg_type in indexed_metadata_keys
    ]

    class BM25AbstractData(base):
        __abstract__ = True
        id = Column(BIGINT, primary_key=True, autoincrement=True)
        text = Column(VARCHAR, nullable=False)
        metadata_ = Column(metadata_dtype)
        node_id = Column(VARCHAR)
        embedding = embedding_col

    model = type(
        class_name,
        (BM25AbstractData,),
        {
            "__tablename__": tablename,
            "__table_args__": (*metadata_indices, {"schema": schema_name}),
        },
    )

    Index(
        f"{indexname}_1",
        model.metadata_["ref_doc_id"].astext,
        postgresql_using="btree",
    )

    return model


class ParadeDBVectorStore(PGVectorStore, BaseModel):
    """
    ParadeDB Vector Store with BM25 search support.
    
    Inherits from PGVectorStore and adds BM25 full-text search capabilities
    using ParadeDB's pg_search extension.

    Examples:
        `pip install llama-index-vector-stores-paradedb`

        ```python
        from llama_index.vector_stores.paradedb import ParadeDBVectorStore

        vector_store = ParadeDBVectorStore.from_params(
            database="vector_db",
            host="localhost",
            password="password",
            port=5432,
            user="postgres",
            table_name="documents",
            hybrid_search=True,
            use_bm25=True,
            embed_dim=1536,
            use_halfvec=True
        )
        ```
    """

    connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = Field(default=None)
    async_connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = Field(default=None)
    table_name: Optional[str] = Field(default=None)
    schema_name: Optional[str] = Field(default="paradedb")
    hybrid_search: bool = Field(default=False)
    text_search_config: str = Field(default="english")
    embed_dim: int = Field(default=1536)
    cache_ok: bool = Field(default=False) 
    perform_setup: bool = Field(default=True)
    debug: bool = Field(default=False)
    use_jsonb: bool = Field(default=False)
    create_engine_kwargs: Optional[Dict[str, Any]] = Field(default=None)
    hnsw_kwargs: Optional[Dict[str, Any]] = Field(default=None)
    use_bm25: bool = Field(default=True)

    def __init__(
        self,
        connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        async_connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        table_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        hybrid_search: bool = False,
        text_search_config: str = "english", 
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        initialization_fail_on_error: bool = False,
        use_halfvec: bool = False,
        use_bm25: bool = True,
        engine: Optional[sqlalchemy.engine.Engine] = None,
        async_engine: Optional[sqlalchemy.ext.asyncio.AsyncEngine] = None,
        indexed_metadata_keys: Optional[Set[Tuple[str, PGType]]] = None,
        customize_query_fn: Optional[Callable[[Select, Any, Any], Select]] = None,
    ) -> None:
        """Constructor."""
        # Initialize Pydantic model with all fields
        BaseModel.__init__(
            self,
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=table_name, 
            schema_name=schema_name or "paradedb",
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs,
            use_bm25=use_bm25
        )
        
        # Call parent constructor
        PGVectorStore.__init__(
            self,
            connection_string=str(connection_string) if connection_string else None,
            async_connection_string=str(async_connection_string) if async_connection_string else None,
            table_name=table_name,
            schema_name=self.schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs or {},
            initialization_fail_on_error=initialization_fail_on_error,
            use_halfvec=use_halfvec,
            engine=engine,
            async_engine=async_engine,
            indexed_metadata_keys=indexed_metadata_keys,
            customize_query_fn=customize_query_fn,
        )
        
        # Override table model if using BM25
        if self.use_bm25:
            from sqlalchemy.orm import declarative_base
            self._base = declarative_base()
            self._table_class = get_bm25_data_model(
                self._base,
                self.table_name,
                self.schema_name,
                self.hybrid_search,
                self.text_search_config,
                self.cache_ok,
                embed_dim=self.embed_dim,
                use_jsonb=self.use_jsonb,
                use_halfvec=self.use_halfvec,
                indexed_metadata_keys=self.indexed_metadata_keys,
            )

    @classmethod
    def class_name(cls) -> str:
        return "ParadeDBVectorStore"

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "llamaindex",
        schema_name: str = "paradedb",
        connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        async_connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        hybrid_search: bool = False,
        use_bm25: bool = True,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        use_halfvec: bool = False,
        indexed_metadata_keys: Optional[Set[Tuple[str, PGType]]] = None,
        customize_query_fn: Optional[Callable[[Select, Any, Any], Select]] = None,
    ) -> "ParadeDBVectorStore":
        """
        Construct from params.

        Args:
            use_bm25 (bool, optional): Enable BM25 search. Defaults to False.
            All other args inherited from PGVectorStore.

        Returns:
            ParadeDBVectorStore: Instance of ParadeDBVectorStore.
        """
        conn_str = (
            connection_string
            or f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )
        async_conn_str = async_connection_string or (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        )
        return cls(
            connection_string=conn_str,
            async_connection_string=async_conn_str,
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            use_bm25=use_bm25,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs,
            use_halfvec=use_halfvec,
            indexed_metadata_keys=indexed_metadata_keys,
            customize_query_fn=customize_query_fn,
        )

    def _create_extension(self) -> None:
        """Override to add pg_search extension for BM25."""
        super()._create_extension()
        
        if self.use_bm25:
            with self._session() as session, session.begin():
                try:
                    session.execute(
                        sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS pg_search")
                    )
                    session.commit()
                except Exception as e:
                    _logger.warning(f"PG Setup: pg_search extension not created: {e}")

    def _create_bm25_index(self) -> None:
        """Create BM25 index using ParadeDB's pg_search."""
        table_fq = f"{self.schema_name}.{self._table_class.__tablename__}"
        index_name = f"{self._table_class.__tablename__}_bm25_idx"

        with self._session() as session, session.begin():
            try:
                statement = sqlalchemy.text(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {table_fq}
                    USING bm25 (id, text)
                    WITH (key_field = 'id');
                """)
                session.execute(statement)
                session.commit()
                _logger.info(f"BM25 index created: {table_fq}")
            except Exception as e:
                session.rollback()
                _logger.warning(f"Failed to create BM25 index {table_fq}: {e}")
                raise

    def _initialize(self) -> None:
        """Override to add BM25 index creation."""
        if not self._is_initialized:
            super()._initialize()
            
            if self.use_bm25 and self.perform_setup:
                try:
                    self._create_bm25_index()
                except Exception as e:
                    _logger.warning(f"PG Setup: Error creating BM25 index: {e}")
                    if self.initialization_fail_on_error:
                        raise

    def _build_sparse_query(
        self,
        query_str: Optional[str],
        limit: int,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> Any:
        """Override to use BM25 if enabled, otherwise use parent's ts_vector."""
        if not self.use_bm25:
            return super()._build_sparse_query(query_str, limit, metadata_filters, **kwargs)
        
        from sqlalchemy import text
        
        if query_str is None:
            raise ValueError("query_str must be specified for a sparse vector query.")

        query_str_clean = re.sub(r"[^\w\s]", " ", query_str).strip()

        base_query = f"""
            SELECT id, node_id, text, metadata_, paradedb.score(id) AS rank
            FROM {self.schema_name}.{self._table_class.__tablename__}
            WHERE text @@@ :query
        """

        if metadata_filters:
            _logger.warning("Metadata filters not fully implemented for BM25 raw SQL")

        stmt = text(f"""
            {base_query}
            ORDER BY rank DESC
            LIMIT :limit
        """).bindparams(query=query_str_clean, limit=limit)

        return stmt

    def _sparse_query_with_rank(
        self,
        query_str: Optional[str] = None,
        limit: int = 10,
        metadata_filters: Optional[Any] = None,
    ) -> List[DBEmbeddingRow]:
        """Override to handle BM25 results properly."""
        if not self.use_bm25:
            return super()._sparse_query_with_rank(query_str, limit, metadata_filters)
        
        stmt = self._build_sparse_query(query_str, limit, metadata_filters)
        with self._session() as session, session.begin():
            res = session.execute(stmt)
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    custom_fields={
                        key: val
                        for key, val in item._asdict().items()
                        if key not in ["id", "node_id", "text", "metadata_", "rank"]
                    },
                    similarity=item.rank,
                )
                for item in res.all()
            ]

    async def _async_sparse_query_with_rank(
        self,
        query_str: Optional[str] = None,
        limit: int = 10,
        metadata_filters: Optional[Any] = None,
    ) -> List[DBEmbeddingRow]:
        """Override to handle async BM25 results properly."""
        if not self.use_bm25:
            return await super()._async_sparse_query_with_rank(query_str, limit, metadata_filters)
        
        stmt = self._build_sparse_query(query_str, limit, metadata_filters)
        async with self._async_session() as session, session.begin():
            res = await session.execute(stmt)
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    custom_fields={
                        key: val
                        for key, val in item._asdict().items()
                        if key not in ["id", "node_id", "text", "metadata_", "rank"]
                    },
                    similarity=item.rank,
                )
                for item in res.all()
            ]