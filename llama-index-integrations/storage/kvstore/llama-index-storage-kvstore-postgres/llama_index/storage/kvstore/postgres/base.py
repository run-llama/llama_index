import json
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse

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
    from sqlalchemy import Column, Index, Integer, UniqueConstraint
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
    """Postgres Key-Value store.

    Args:
        mongo_client (Any): MongoDB client
        uri (Optional[str]): MongoDB URI
        host (Optional[str]): MongoDB host
        port (Optional[int]): MongoDB port
        db_name (Optional[str]): MongoDB database name
    """

    connection_string: str
    async_connection_string: str
    table_name: str
    schema_name: str
    perform_setup: bool
    debug: bool
    use_jsonb: bool

    def __init__(
        self,
        connection_string: str,
        async_connection_string: str,
        table_name: str,
        schema_name: str = "public",
        perform_setup: bool = True,
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> None:
        try:
            import asyncpg  # noqa
            import psycopg2  # noqa
            import sqlalchemy
            import sqlalchemy.ext.asyncio  # noqa
        except ImportError:
            raise ImportError(
                "`sqlalchemy[asyncio]`, `psycopg2-binary` and `asyncpg` "
                "packages should be pre installed"
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
        self._is_initialized = False

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

        self._engine = create_engine(self.connection_string, echo=self.debug)
        self._session = sessionmaker(self._engine)

        self._async_engine = create_async_engine(self.async_connection_string)
        self._async_session = sessionmaker(self._async_engine, class_=AsyncSession)

    def _create_schema_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            from sqlalchemy import text

            # Check if the specified schema exists with "CREATE" statement
            check_schema_statement = text(
                f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{self.schema_name}'"
            )
            result = session.execute(check_schema_statement).fetchone()

            # If the schema does not exist, then create it
            if not result:
                create_schema_statement = text(
                    f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}"
                )
                session.execute(create_schema_statement)

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
        """Put a key-value pair into the store.

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
        """Put a key-value pair into the store.

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
        from sqlalchemy import text

        self._initialize()
        with self._session() as session:
            for i in range(0, len(kv_pairs), batch_size):
                batch = kv_pairs[i : i + batch_size]

                # Prepare the VALUES part of the SQL statement
                values_clause = ", ".join(
                    f"(:key_{i}, :namespace_{i}, :value_{i})"
                    for i, _ in enumerate(batch)
                )

                # Prepare the raw SQL for bulk upsert
                # Note: This SQL is PostgreSQL-specific. Adjust for other databases.
                stmt = text(
                    f"""
                INSERT INTO {self.schema_name}.{self._table_class.__tablename__} (key, namespace, value)
                VALUES {values_clause}
                ON CONFLICT (key, namespace)
                DO UPDATE SET
                value = EXCLUDED.value;
                """
                )

                # Flatten the list of tuples for execute parameters
                params = {}
                for i, (key, value) in enumerate(batch):
                    params[f"key_{i}"] = key
                    params[f"namespace_{i}"] = collection
                    params[f"value_{i}"] = json.dumps(value)

                # Execute the bulk upsert
                session.execute(stmt, params)
                session.commit()

    async def aput_all(
        self,
        kv_pairs: List[Tuple[str, dict]],
        collection: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        from sqlalchemy import text

        self._initialize()
        async with self._async_session() as session:
            for i in range(0, len(kv_pairs), batch_size):
                batch = kv_pairs[i : i + batch_size]

                # Prepare the VALUES part of the SQL statement
                values_clause = ", ".join(
                    f"(:key_{i}, :namespace_{i}, :value_{i})"
                    for i, _ in enumerate(batch)
                )

                # Prepare the raw SQL for bulk upsert
                # Note: This SQL is PostgreSQL-specific. Adjust for other databases.
                stmt = text(
                    f"""
                INSERT INTO {self.schema_name}.{self._table_class.__tablename__} (key, namespace, value)
                VALUES {values_clause}
                ON CONFLICT (key, namespace)
                DO UPDATE SET
                value = EXCLUDED.value;
                """
                )

                # Flatten the list of tuples for execute parameters
                params = {}
                for i, (key, value) in enumerate(batch):
                    params[f"key_{i}"] = key
                    params[f"namespace_{i}"] = collection
                    params[f"value_{i}"] = json.dumps(value)

                # Execute the bulk upsert
                await session.execute(stmt, params)
                await session.commit()

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        """Get a value from the store.

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
        """Get a value from the store.

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
        """Get all values from the store.

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
        """Get all values from the store.

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
        """Delete a value from the store.

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
        """Delete a value from the store.

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
