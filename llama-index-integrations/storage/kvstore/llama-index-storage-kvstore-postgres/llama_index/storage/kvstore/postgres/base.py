from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse
from llama_index.core.bridge.pydantic import PrivateAttr

try:
    import sqlalchemy
    import sqlalchemy.ext.asyncio  # noqa
    from sqlalchemy import Column, Index, Integer, UniqueConstraint, inspect
    from sqlalchemy.schema import CreateSchema
except ImportError:
    raise ImportError("`sqlalchemy[asyncio]` package should be pre installed")

from llama_index.core.storage.kvstore.types import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION,
    BaseKVStore,
)

IMPORT_ERROR_MSG = "`asyncpg` package not found, please run `pip install asyncpg`"


def get_data_model(
    base: Type,
    index_name: str,
    schema_name: str,
    use_jsonb: bool = False,
) -> Any:
    """
    This part create a dynamic sqlalchemy model with a new table.
    """
    from sqlalchemy.dialects.postgresql import JSON, JSONB, VARCHAR

    tablename = "data_%s" % index_name  # dynamic table name
    class_name = "Data%s" % index_name  # dynamic class name

    metadata_dtype = JSONB if use_jsonb else JSON

    class AbstractData(base):  # type: ignore
        __abstract__ = True  # this line is necessary
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
                UniqueConstraint(
                    "key", "namespace", name=f"{tablename}:unique_key_namespace"
                ),
                Index(f"{tablename}:idx_key_namespace", "key", "namespace"),
                {"schema": schema_name},
            ),
        },
    )


class PostgresKVStore(BaseKVStore):
    """
    Postgres Key-Value store.

    Args:
        connection_string (str): psycopg2 connection string
        async_connection_string (str): asyncpg connection string
        table_name (str): table name
        schema_name (Optional[str]): schema name
        perform_setup (Optional[bool]): perform table setup
        debug (Optional[bool]): debug mode
        use_jsonb (Optional[bool]): use JSONB data type for storage

    """

    connection_string: Optional[str]
    async_connection_string: Optional[str]
    table_name: str
    schema_name: str
    perform_setup: bool
    debug: bool
    use_jsonb: bool
    _engine: Optional[sqlalchemy.engine.Engine] = PrivateAttr()
    _async_engine: Optional[sqlalchemy.ext.asyncio.AsyncEngine] = PrivateAttr()

    def __init__(
        self,
        table_name: str,
        connection_string: Optional[str] = None,
        async_connection_string: Optional[str] = None,
        schema_name: str = "public",
        engine: Optional[sqlalchemy.engine.Engine] = None,
        async_engine: Optional[sqlalchemy.ext.asyncio.AsyncEngine] = None,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> None:
        try:
            import asyncpg  # noqa
            import psycopg2  # noqa
        except ImportError:
            raise ImportError(
                "`psycopg2-binary` and `asyncpg` packages should be pre installed"
            )

        table_name = table_name.lower()
        schema_name = schema_name.lower()
        self.connection_string = connection_string
        self.async_connection_string = async_connection_string
        self.table_name = table_name
        self.schema_name = schema_name
        self.perform_setup = perform_setup
        self.debug = debug
        self.use_jsonb = use_jsonb
        self._engine = engine
        self._async_engine = async_engine
        self._is_initialized = False

        if not self._async_engine and not self.async_connection_string:
            raise ValueError(
                "You should provide an asynchronous connection string, if you do not provide an asynchronous SqlAlchemy engine"
            )
        elif not self._engine and not self.connection_string:
            raise ValueError(
                "You should provide a synchronous connection string, if you do not provide a synchronous SqlAlchemy engine"
            )
        elif (
            not self._engine
            and not self._async_engine
            and (not self.connection_string or not self.connection_string)
        ):
            raise ValueError(
                "If a SqlAlchemy engine is not provided, you should provide a synchronous and an asynchronous connection string"
            )

        from sqlalchemy.orm import declarative_base

        # sqlalchemy model
        self._base = declarative_base()
        self._table_class = get_data_model(
            self._base,
            table_name,
            schema_name,
            use_jsonb=use_jsonb,
        )

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "kvstore",
        schema_name: str = "public",
        connection_string: Optional[str] = None,
        async_connection_string: Optional[str] = None,
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> "PostgresKVStore":
        """Return connection string from database parameters."""
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
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "kvstore",
        schema_name: str = "public",
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> "PostgresKVStore":
        """Return connection string from database parameters."""
        params = params_from_uri(uri)
        return cls.from_params(
            **params,
            table_name=table_name,
            schema_name=schema_name,
            perform_setup=perform_setup,
            debug=debug,
            use_jsonb=use_jsonb,
        )

    def _connect(self) -> Any:
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        self._engine = self._engine or create_engine(
            self.connection_string, echo=self.debug
        )
        self._session = sessionmaker(self._engine)

        self._async_engine = self._async_engine or create_async_engine(
            self.async_connection_string
        )
        self._async_session = sessionmaker(self._async_engine, class_=AsyncSession)

    def _create_schema_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            inspector = inspect(session.connection())
            existing_schemas = inspector.get_schema_names()
            if self.schema_name not in existing_schemas:
                session.execute(CreateSchema(self.schema_name, if_not_exists=True))
            session.commit()

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
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        self.put_all([(key, val)], collection=collection)

    async def aput(
        self,
        key: str,
        val: dict,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Put a key-value pair into the store.

        Args:
            key (str): key
            val (dict): value
            collection (str): collection name

        """
        await self.aput_all([(key, val)], collection=collection)

    def put_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        from sqlalchemy.dialects.postgresql import insert

        self._initialize()
        with self._session() as session:
            for i in range(0, len(kv_pairs), batch_size):
                batch = kv_pairs[i : i + batch_size]

                values_to_insert = [
                    {
                        "key": key,
                        "namespace": collection,
                        "value": value,
                    }
                    for key, value in batch
                ]

                stmt = insert(self._table_class).values(values_to_insert)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["key", "namespace"],
                    set_={"value": stmt.excluded.value},
                )

                session.execute(stmt)
                session.commit()

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        from sqlalchemy.dialects.postgresql import insert

        self._initialize()
        async with self._async_session() as session:
            for i in range(0, len(kv_pairs), batch_size):
                batch = kv_pairs[i : i + batch_size]

                values_to_insert = [
                    {
                        "key": key,
                        "namespace": collection,
                        "value": value,
                    }
                    for key, value in batch
                ]

                stmt = insert(self._table_class).values(values_to_insert)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["key", "namespace"],
                    set_={"value": stmt.excluded.value},
                )

                await session.execute(stmt)
                await session.commit()

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        from sqlalchemy import select

        self._initialize()
        with self._session() as session:
            result = session.execute(
                select(self._table_class)
                .filter_by(key=key)
                .filter_by(namespace=collection)
            )
            result = result.scalars().first()
            if result:
                return result.value
        return None

    async def aget(
        self, key: str, collection: str = DEFAULT_COLLECTION
    ) -> Optional[dict]:
        """
        Get a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        from sqlalchemy import select

        self._initialize()
        async with self._async_session() as session:
            result = await session.execute(
                select(self._table_class)
                .filter_by(key=key)
                .filter_by(namespace=collection)
            )
            result = result.scalars().first()
            if result:
                return result.value
        return None

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        from sqlalchemy import select

        self._initialize()
        with self._session() as session:
            results = session.execute(
                select(self._table_class).filter_by(namespace=collection)
            )
            results = results.scalars().all()
        return {result.key: result.value for result in results} if results else {}

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        """
        Get all values from the store.

        Args:
            collection (str): collection name

        """
        from sqlalchemy import select

        self._initialize()
        async with self._async_session() as session:
            results = await session.execute(
                select(self._table_class).filter_by(namespace=collection)
            )
            results = results.scalars().all()
        return {result.key: result.value for result in results} if results else {}

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        from sqlalchemy import delete

        self._initialize()
        with self._session() as session:
            result = session.execute(
                delete(self._table_class)
                .filter_by(namespace=collection)
                .filter_by(key=key)
            )
            session.commit()
        return result.rowcount > 0

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """
        Delete a value from the store.

        Args:
            key (str): key
            collection (str): collection name

        """
        from sqlalchemy import delete

        self._initialize()
        async with self._async_session() as session:
            async with session.begin():
                result = await session.execute(
                    delete(self._table_class)
                    .filter_by(namespace=collection)
                    .filter_by(key=key)
                )
        return result.rowcount > 0


def params_from_uri(uri: str) -> dict:
    result = urlparse(uri)
    database = result.path[1:]
    port = result.port if result.port else 5432
    return {
        "database": database,
        "user": result.username,
        "password": result.password,
        "host": result.hostname,
        "port": port,
    }
