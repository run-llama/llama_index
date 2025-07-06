import logging
from llama_index.vector_stores.postgres.base import PGVectorStore
from typing import (
    Any,
    NamedTuple,
    Type,
    Optional,
    Union,
    Set,
    Dict,
    List,
    Tuple,
    Literal,
)
from llama_index.core.vector_stores.types import (
    MetadataFilters,
)
import sqlalchemy

_logger = logging.getLogger(__name__)


class DBEmbeddingRow(NamedTuple):
    node_id: str
    text: str
    metadata: dict
    similarity: float


PGType = Literal[
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


def get_data_model(
    base: Type,
    index_name: str,
    schema_name: str,
    hybrid_search: bool,
    text_search_config: str,
    cache_okay: bool,
    embed_dim: int = 1536,
    use_jsonb: bool = False,
    indexed_metadata_keys: Optional[Set[Tuple[str, PGType]]] = None,
) -> Any:
    """
    This part create a dynamic sqlalchemy model with a new table.
    """
    from opengauss_sqlalchemy.usertype import Vector
    from sqlalchemy import Column, Computed
    from sqlalchemy.dialects.postgresql import (
        BIGINT,
        JSON,
        JSONB,
        TSVECTOR,
        VARCHAR,
        UUID,
        DOUBLE_PRECISION,
    )
    from sqlalchemy import cast, column
    from sqlalchemy import String, Integer, Numeric, Float, Boolean, Date, DateTime
    from sqlalchemy.schema import Index
    from sqlalchemy.types import TypeDecorator

    pg_type_map = {
        "text": String,
        "int": Integer,
        "integer": Integer,
        "numeric": Numeric,
        "float": Float,
        "double precision": DOUBLE_PRECISION,  # or Float(precision=53)
        "boolean": Boolean,
        "date": Date,
        "timestamp": DateTime,
        "uuid": UUID,
    }

    indexed_metadata_keys = indexed_metadata_keys or set()
    # check that types are in pg_type_map
    for key, pg_type in indexed_metadata_keys:
        if pg_type not in pg_type_map:
            raise ValueError(
                f"Invalid type {pg_type} for key {key}. "
                f"Must be one of {list(pg_type_map.keys())}"
            )

    class TSVector(TypeDecorator):
        impl = TSVECTOR
        cache_ok = cache_okay

    tablename = "data_og_%s" % index_name  # dynamic table name
    class_name = "Data_og_%s" % index_name  # dynamic class name
    indexname = "%s_og_idx" % index_name  # dynamic class name

    metadata_dtype = JSONB if use_jsonb else JSON

    embedding_col = Column(Vector(embed_dim))

    metadata_indices = [
        Index(
            f"{indexname}_{key}_{pg_type.replace(' ', '_')}",
            cast(column("metadata_").op("->>")(key), pg_type_map[pg_type]),
            postgresql_using="btree",
        )
        for key, pg_type in indexed_metadata_keys
    ]

    if hybrid_search:

        class HybridAbstractData(base):  # type: ignore
            __abstract__ = True  # this line is necessary
            id = Column(BIGINT, primary_key=True, autoincrement=True)
            text = Column(VARCHAR, nullable=False)
            metadata_ = Column(metadata_dtype)
            node_id = Column(VARCHAR)
            embedding = embedding_col
            text_search_tsv = Column(  # type: ignore
                TSVector(),
                Computed(
                    "to_tsvector('%s', text)" % text_search_config, persisted=True
                ),
            )

        model = type(
            class_name,
            (HybridAbstractData,),
            {
                "__tablename__": tablename,
                "__table_args__": (*metadata_indices, {"schema": schema_name}),
            },
        )

        Index(
            indexname,
            model.text_search_tsv,  # type: ignore
            postgresql_using="gin",
        )

        Index(
            f"{indexname}_1",
            model.metadata_["ref_doc_id"].astext,  # type: ignore
            postgresql_using="btree",
        )
    else:

        class AbstractData(base):  # type: ignore
            __abstract__ = True  # this line is necessary
            id = Column(BIGINT, primary_key=True, autoincrement=True)
            text = Column(VARCHAR, nullable=False)
            metadata_ = Column(metadata_dtype)
            node_id = Column(VARCHAR)
            embedding = embedding_col

        model = type(
            class_name,
            (AbstractData,),
            {
                "__tablename__": tablename,
                "__table_args__": (*metadata_indices, {"schema": schema_name}),
            },
        )

        Index(
            f"{indexname}_1",
            model.metadata_["ref_doc_id"].astext,  # type: ignore
            postgresql_using="btree",
        )

    return model


class OpenGaussStore(PGVectorStore):
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
        engine: Optional[sqlalchemy.engine.Engine] = None,
        async_engine: Optional[sqlalchemy.ext.asyncio.AsyncEngine] = None,
        indexed_metadata_keys: Optional[Set[Tuple[str, PGType]]] = None,
    ) -> None:
        """
        Constructor.

        Args:
            connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to postgres db.
            async_connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to async pg db.
            table_name (str): Table name.
            schema_name (str): Schema name.
            hybrid_search (bool, optional): Enable hybrid search. Defaults to False.
            text_search_config (str, optional): Text search configuration. Defaults to "english".
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            cache_ok (bool, optional): Enable cache. Defaults to False.
            perform_setup (bool, optional): If db should be set up. Defaults to True.
            debug (bool, optional): Debug mode. Defaults to False.
            use_jsonb (bool, optional): Use JSONB instead of JSON. Defaults to False.
            hnsw_kwargs (Optional[Dict[str, Any]], optional): HNSW kwargs, a dict that
                contains "hnsw_ef_construction", "hnsw_ef_search", "hnsw_m", and optionally "hnsw_dist_method". Defaults to None,
                which turns off HNSW search.
            create_engine_kwargs (Optional[Dict[str, Any]], optional): Engine parameters to pass to create_engine. Defaults to None.
            engine (Optional[sqlalchemy.engine.Engine], optional): SQLAlchemy engine instance to use. Defaults to None.
            async_engine (Optional[sqlalchemy.ext.asyncio.AsyncEngine], optional): SQLAlchemy async engine instance to use. Defaults to None.
            indexed_metadata_keys (Optional[List[Tuple[str, str]]], optional): Set of metadata keys with their type to index. Defaults to None.

        """
        super().__init__(
            connection_string=str(connection_string),
            async_connection_string=str(async_connection_string),
            table_name=table_name,
            schema_name=schema_name,
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
            use_halfvec=False,
            indexed_metadata_keys=indexed_metadata_keys,
        )
        self._table_class = get_data_model(
            self._base,
            table_name,
            schema_name,
            hybrid_search,
            text_search_config,
            cache_ok,
            embed_dim=embed_dim,
            use_jsonb=use_jsonb,
            indexed_metadata_keys=indexed_metadata_keys,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OpenGaussStore"

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "llamaindex",
        schema_name: str = "public",
        connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        async_connection_string: Optional[Union[str, sqlalchemy.engine.URL]] = None,
        hybrid_search: bool = False,
        text_search_config: str = "english",
        embed_dim: int = 1536,
        cache_ok: bool = False,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
        hnsw_kwargs: Optional[Dict[str, Any]] = None,
        create_engine_kwargs: Optional[Dict[str, Any]] = None,
        indexed_metadata_keys: Optional[Set[Tuple[str, PGType]]] = None,
    ) -> "OpenGaussStore":
        """
        Construct from params.

        Args:
            host (Optional[str], optional): Host of postgres connection. Defaults to None.
            port (Optional[str], optional): Port of postgres connection. Defaults to None.
            database (Optional[str], optional): Postgres DB name. Defaults to None.
            user (Optional[str], optional): Postgres username. Defaults to None.
            password (Optional[str], optional): Postgres password. Defaults to None.
            table_name (str): Table name. Defaults to "llamaindex".
            schema_name (str): Schema name. Defaults to "public".
            connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to postgres db
            async_connection_string (Union[str, sqlalchemy.engine.URL]): Connection string to async pg db
            hybrid_search (bool, optional): Enable hybrid search. Defaults to False.
            text_search_config (str, optional): Text search configuration. Defaults to "english".
            embed_dim (int, optional): Embedding dimensions. Defaults to 1536.
            cache_ok (bool, optional): Enable cache. Defaults to False.
            perform_setup (bool, optional): If db should be set up. Defaults to True.
            debug (bool, optional): Debug mode. Defaults to False.
            use_jsonb (bool, optional): Use JSONB instead of JSON. Defaults to False.
            hnsw_kwargs (Optional[Dict[str, Any]], optional): HNSW kwargs, a dict that
                contains "hnsw_ef_construction", "hnsw_ef_search", "hnsw_m", and optionally "hnsw_dist_method". Defaults to None,
                which turns off HNSW search.
            create_engine_kwargs (Optional[Dict[str, Any]], optional): Engine parameters to pass to create_engine. Defaults to None.
            indexed_metadata_keys (Optional[Set[Tuple[str, str]]], optional): Set of metadata keys to index. Defaults to None.

        Returns:
            PGVectorStore: Instance of PGVectorStore constructed from params.

        """
        conn_str = (
            connection_string
            or f"opengauss+psycopg2://{user}:{password}@{host}:{port}/{database}"
        )
        async_conn_str = async_connection_string or (
            f"opengauss+asyncpg://{user}:{password}@{host}:{port}/{database}"
        )
        return cls(
            connection_string=conn_str,
            async_connection_string=async_conn_str,
            table_name=table_name,
            schema_name=schema_name,
            hybrid_search=hybrid_search,
            text_search_config=text_search_config,
            embed_dim=embed_dim,
            cache_ok=cache_ok,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            hnsw_kwargs=hnsw_kwargs,
            create_engine_kwargs=create_engine_kwargs,
            indexed_metadata_keys=indexed_metadata_keys,
        )

    def _initialize(self) -> None:
        fail_on_error = self.initialization_fail_on_error
        if not self._is_initialized:
            self._connect()
            if self.perform_setup:
                try:
                    self._create_schema_if_not_exists()
                except Exception as e:
                    _logger.warning(f"PG Setup: Error creating schema: {e}")
                    if fail_on_error:
                        raise
                try:
                    self._create_tables_if_not_exists()
                except Exception as e:
                    _logger.warning(f"PG Setup: Error creating tables: {e}")
                    if fail_on_error:
                        raise
                if self.hnsw_kwargs is not None:
                    try:
                        self._create_hnsw_index()
                    except Exception as e:
                        _logger.warning(f"PG Setup: Error creating HNSW index: {e}")
                        if fail_on_error:
                            raise
            self._is_initialized = True

    def _connect(self) -> Any:
        from sqlalchemy import create_engine, event
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker
        from opengauss_sqlalchemy.register_async import register_vector

        self._engine = self._engine or create_engine(
            self.connection_string, echo=self.debug, **self.create_engine_kwargs
        )
        self._session = sessionmaker(self._engine)

        self._async_engine = self._async_engine or create_async_engine(
            self.async_connection_string, **self.create_engine_kwargs
        )

        @event.listens_for(self._async_engine.sync_engine, "connect")
        def _connect_event(dbapi_connection, connection_record):
            dbapi_connection.run_async(register_vector)

        self._async_session = sessionmaker(self._async_engine, class_=AsyncSession)  # type: ignore

    def _query_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_query(embedding, limit, metadata_filters)
        with self._session() as session, session.begin():
            from sqlalchemy import text

            if kwargs.get("ivfflat_probes"):
                ivfflat_probes = kwargs.get("ivfflat_probes")
                session.execute(
                    text(f"SET ivfflat_probes = :ivfflat_probes"),
                    {"ivfflat_probes": ivfflat_probes},
                )
            if self.hnsw_kwargs:
                hnsw_ef_search = (
                    kwargs.get("hnsw_ef_search") or self.hnsw_kwargs["hnsw_ef_search"]
                )
                session.execute(
                    text(f"SET hnsw_ef_search = :hnsw_ef_search"),
                    {"hnsw_ef_search": hnsw_ef_search},
                )

            res = session.execute(
                stmt,
            )
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    similarity=(1 - item.distance) if item.distance is not None else 0,
                )
                for item in res.all()
            ]

    async def _aquery_with_score(
        self,
        embedding: Optional[List[float]],
        limit: int = 10,
        metadata_filters: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[DBEmbeddingRow]:
        stmt = self._build_query(embedding, limit, metadata_filters)
        async with self._async_session() as async_session, async_session.begin():
            from sqlalchemy import text

            if self.hnsw_kwargs:
                hnsw_ef_search = (
                    kwargs.get("hnsw_ef_search") or self.hnsw_kwargs["hnsw_ef_search"]
                )
                await async_session.execute(
                    text(f"SET hnsw_ef_search = {hnsw_ef_search}"),
                )
            if kwargs.get("ivfflat_probes"):
                ivfflat_probes = kwargs.get("ivfflat_probes")
                await async_session.execute(
                    text(f"SET ivfflat_probes = :ivfflat_probes"),
                    {"ivfflat_probes": ivfflat_probes},
                )

            res = await async_session.execute(stmt)
            return [
                DBEmbeddingRow(
                    node_id=item.node_id,
                    text=item.text,
                    metadata=item.metadata_,
                    similarity=(1 - item.distance) if item.distance is not None else 0,
                )
                for item in res.all()
            ]
