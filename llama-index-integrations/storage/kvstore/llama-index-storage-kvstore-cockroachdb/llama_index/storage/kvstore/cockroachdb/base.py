"""CockroachDB KV store backend for LlamaIndex."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

try:
    import sqlalchemy
    import sqlalchemy.ext.asyncio  # noqa: F401
    from sqlalchemy import Column, Index, Integer, UniqueConstraint, inspect
    from sqlalchemy.schema import CreateSchema
except ImportError as exc:  # pragma: no cover
    raise ImportError("`sqlalchemy[asyncio]` package should be pre installed") from exc


def get_data_model(
    base: type,
    index_name: str,
    schema_name: str,
    use_jsonb: bool = True,
) -> Any:
    """Build a dynamic SQLAlchemy model for the KV table.

    CockroachDB ships JSONB as the canonical JSON type, so we default to it.
    """
    from sqlalchemy.dialects.postgresql import JSON, JSONB, VARCHAR

    tablename = f"data_{index_name}"
    class_name = f"Data{index_name}"

    metadata_dtype = JSONB if use_jsonb else JSON

    class AbstractData(base):  # type: ignore[misc, valid-type]
        __abstract__ = True
        id = Column(Integer, primary_key=True, autoincrement=True)
        key = Column(VARCHAR, nullable=False)
        namespace = Column(VARCHAR, nullable=False)
        value = Column(metadata_dtype)

    return type(
        class_name,
        (AbstractData,),
        {
            "__tablename__": tablename,
            "__table_args__": (
                UniqueConstraint("key", "namespace", name=f"{tablename}:unique_key_namespace"),
                Index(f"{tablename}:idx_key_namespace", "key", "namespace"),
                {"schema": schema_name},
            ),
        },
    )


class CockroachDBKVStore(BaseKVStore):
    """CockroachDB-backed Key-Value store.

    Uses the ``cockroachdb+psycopg2`` and ``cockroachdb+asyncpg`` SQLAlchemy
    dialects so that CockroachDB's transparent retries on
    ``SERIALIZATION_FAILURE`` errors are applied automatically.

    Args:
        connection_string: psycopg2 connection string.
        async_connection_string: asyncpg connection string.
        table_name: table name (will be lowercased and prefixed with ``data_``).
        schema_name: schema name (defaults to ``public``).
        perform_setup: create schema + table on first use.
        debug: enable SQLAlchemy echo.
        use_jsonb: store ``value`` as JSONB (recommended on CRDB).
        create_engine_kwargs: extra kwargs forwarded to ``create_engine`` and
            ``create_async_engine`` (for example ``connect_args`` for pooling).

    """

    connection_string: str | None
    async_connection_string: str | None
    table_name: str
    schema_name: str
    perform_setup: bool
    debug: bool
    use_jsonb: bool
    create_engine_kwargs: dict[str, Any]
    _engine: sqlalchemy.engine.Engine | None = PrivateAttr()
    _async_engine: sqlalchemy.ext.asyncio.AsyncEngine | None = PrivateAttr()

    def __init__(
        self,
        table_name: str,
        connection_string: str | None = None,
        async_connection_string: str | None = None,
        schema_name: str = "public",
        create_engine_kwargs: dict[str, Any] | None = None,
        engine: sqlalchemy.engine.Engine | None = None,
        async_engine: sqlalchemy.ext.asyncio.AsyncEngine | None = None,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = True,
    ) -> None:
        try:
            import asyncpg  # noqa: F401
            import psycopg2  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "`psycopg2-binary` and `asyncpg` packages should be pre installed"
            ) from exc

        table_name = table_name.lower()
        schema_name = schema_name.lower()
        self.connection_string = connection_string
        self.async_connection_string = async_connection_string
        self.table_name = table_name
        self.schema_name = schema_name
        self.perform_setup = perform_setup
        self.debug = debug
        self.use_jsonb = use_jsonb
        self.create_engine_kwargs = create_engine_kwargs or {}
        self._engine = engine
        self._async_engine = async_engine
        self._is_initialized = False

        if not self._async_engine and not self.async_connection_string:
            raise ValueError(
                "Provide an asynchronous connection string when no async engine is supplied."
            )
        if not self._engine and not self.connection_string:
            raise ValueError(
                "Provide a synchronous connection string when no sync engine is supplied."
            )

        from sqlalchemy.orm import declarative_base

        self._base = declarative_base()
        self._table_class = get_data_model(
            self._base,
            table_name,
            schema_name,
            use_jsonb=use_jsonb,
        )

    @classmethod
    def class_name(cls) -> str:
        return "CockroachDBKVStore"

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
        table_name: str = "kvstore",
        schema_name: str = "public",
        connection_string: str | None = None,
        async_connection_string: str | None = None,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = True,
        create_engine_kwargs: dict[str, Any] | None = None,
    ) -> CockroachDBKVStore:
        """Build a store from individual connection parameters.

        Default port is 26257. SSL is on by default; pass ``sslmode="disable"``
        for an insecure local cluster.
        """
        port = port or 26257
        if connection_string is None:
            params: list[str] = []
            if sslmode:
                params.append(f"sslmode={sslmode}")
            if sslrootcert:
                params.append(f"sslrootcert={sslrootcert}")
            qs = "?" + "&".join(params) if params else ""
            connection_string = (
                f"cockroachdb+psycopg2://{user}:{password}@{host}:{port}/{database}{qs}"
            )
        if async_connection_string is None:
            params = []
            if sslmode and sslmode != "disable":
                params.append("ssl=true")
            qs = "?" + "&".join(params) if params else ""
            async_connection_string = (
                f"cockroachdb+asyncpg://{user}:{password}@{host}:{port}/{database}{qs}"
            )
        return cls(
            connection_string=connection_string,
            async_connection_string=async_connection_string,
            table_name=table_name,
            schema_name=schema_name,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            create_engine_kwargs=create_engine_kwargs,
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "kvstore",
        schema_name: str = "public",
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = True,
        create_engine_kwargs: dict[str, Any] | None = None,
    ) -> CockroachDBKVStore:
        """Build a store from a URI like ``cockroachdb://user:pass@host:26257/db``."""
        params = params_from_uri(uri)
        return cls.from_params(
            **params,
            table_name=table_name,
            schema_name=schema_name,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
            create_engine_kwargs=create_engine_kwargs,
        )

    def _connect(self) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        self._engine = self._engine or create_engine(
            self.connection_string, echo=self.debug, **self.create_engine_kwargs
        )
        self._session = sessionmaker(self._engine)

        self._async_engine = self._async_engine or create_async_engine(
            self.async_connection_string, **self.create_engine_kwargs
        )
        self._async_session = sessionmaker(self._async_engine, class_=AsyncSession)

    def _create_schema_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            inspector = inspect(session.connection())
            existing_schemas = inspector.get_schema_names()
            if self.schema_name not in existing_schemas:
                session.execute(CreateSchema(self.schema_name))

    def _create_tables_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            self._base.metadata.create_all(session.connection())

    def _initialize(self) -> None:
        if not self._is_initialized:
            self._connect()
            if self.perform_setup:
                self._create_schema_if_not_exists()
                self._create_tables_if_not_exists()
            self._is_initialized = True

    def put(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        self.put_all([(key, val)], collection=collection)

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        await self.aput_all([(key, val)], collection=collection)

    def put_all(
        self,
        kv_pairs: list[tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        from sqlalchemy.dialects.postgresql import insert

        self._initialize()
        with self._session() as session:
            for i in range(0, len(kv_pairs), batch_size):
                batch = kv_pairs[i : i + batch_size]
                rows = [
                    {"key": key, "namespace": collection, "value": value} for key, value in batch
                ]
                stmt = insert(self._table_class).values(rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["key", "namespace"],
                    set_={"value": stmt.excluded.value},
                )
                session.execute(stmt)
                session.commit()

    async def aput_all(
        self,
        kv_pairs: list[tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        from sqlalchemy.dialects.postgresql import insert

        self._initialize()
        async with self._async_session() as session:
            for i in range(0, len(kv_pairs), batch_size):
                batch = kv_pairs[i : i + batch_size]
                rows = [
                    {"key": key, "namespace": collection, "value": value} for key, value in batch
                ]
                stmt = insert(self._table_class).values(rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["key", "namespace"],
                    set_={"value": stmt.excluded.value},
                )
                await session.execute(stmt)
                await session.commit()

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> dict | None:
        from sqlalchemy import select

        self._initialize()
        with self._session() as session:
            result = session.execute(
                select(self._table_class).filter_by(key=key).filter_by(namespace=collection)
            )
            row = result.scalars().first()
            return row.value if row else None

    async def aget(self, key: str, collection: str = DEFAULT_COLLECTION) -> dict | None:
        from sqlalchemy import select

        self._initialize()
        async with self._async_session() as session:
            result = await session.execute(
                select(self._table_class).filter_by(key=key).filter_by(namespace=collection)
            )
            row = result.scalars().first()
            return row.value if row else None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> dict[str, dict]:
        from sqlalchemy import select

        self._initialize()
        with self._session() as session:
            results = session.execute(select(self._table_class).filter_by(namespace=collection))
            rows = results.scalars().all()
        return {row.key: row.value for row in rows} if rows else {}

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> dict[str, dict]:
        from sqlalchemy import select

        self._initialize()
        async with self._async_session() as session:
            results = await session.execute(
                select(self._table_class).filter_by(namespace=collection)
            )
            rows = results.scalars().all()
        return {row.key: row.value for row in rows} if rows else {}

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        from sqlalchemy import delete

        self._initialize()
        with self._session() as session:
            result = session.execute(
                delete(self._table_class).filter_by(namespace=collection).filter_by(key=key)
            )
            session.commit()
        return result.rowcount > 0

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        from sqlalchemy import delete

        self._initialize()
        async with self._async_session() as session, session.begin():
            result = await session.execute(
                delete(self._table_class).filter_by(namespace=collection).filter_by(key=key)
            )
        return result.rowcount > 0


def params_from_uri(uri: str) -> dict:
    result = urlparse(uri)
    database = result.path[1:]
    port = result.port if result.port else 26257
    return {
        "database": database,
        "user": result.username,
        "password": result.password,
        "host": result.hostname,
        "port": port,
    }
