import json
from typing import Any, Optional
from urllib.parse import urlparse

from sqlalchemy import (
    Index,
    Column,
    Integer,
    UniqueConstraint,
    text,
    delete,
    select,
    create_engine,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from llama_index.core.llms import ChatMessage
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.dialects.postgresql import JSON, ARRAY, JSONB, VARCHAR
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.storage.chat_store.base import BaseChatStore


def get_data_model(
    base: type,
    index_name: str,
    schema_name: str,
    use_jsonb: bool = False,
) -> Any:
    """
    This part create a dynamic sqlalchemy model with a new table.
    """
    tablename = f"data_{index_name}"  # dynamic table name
    class_name = f"Data{index_name}"  # dynamic class name

    chat_dtype = JSONB if use_jsonb else JSON

    class AbstractData(base):  # type: ignore
        __abstract__ = True  # this line is necessary
        id = Column(Integer, primary_key=True, autoincrement=True)  # Add primary key
        key = Column(VARCHAR, nullable=False)
        value = Column(ARRAY(chat_dtype))

    return type(
        class_name,
        (AbstractData,),
        {
            "__tablename__": tablename,
            "__table_args__": (
                UniqueConstraint("key", name=f"{tablename}:unique_key"),
                Index(f"{tablename}:idx_key", "key"),
                {"schema": schema_name},
            ),
        },
    )


class PostgresChatStore(BaseChatStore):
    table_name: Optional[str] = Field(
        default="chatstore", description="Postgres table name."
    )
    schema_name: Optional[str] = Field(
        default="public", description="Postgres schema name."
    )

    _table_class: Optional[Any] = PrivateAttr()
    _session: Optional[sessionmaker] = PrivateAttr()
    _async_session: Optional[sessionmaker] = PrivateAttr()

    def __init__(
        self,
        session: sessionmaker,
        async_session: sessionmaker,
        table_name: str,
        schema_name: str = "public",
        use_jsonb: bool = False,
    ):
        super().__init__(
            table_name=table_name.lower(),
            schema_name=schema_name.lower(),
        )

        # sqlalchemy model
        base = declarative_base()
        self._table_class = get_data_model(
            base,
            table_name,
            schema_name,
            use_jsonb=use_jsonb,
        )
        self._session = session
        self._async_session = async_session
        self._initialize(base)

    @classmethod
    def from_params(
        cls,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: str = "chatstore",
        schema_name: str = "public",
        connection_string: Optional[str] = None,
        async_connection_string: Optional[str] = None,
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> "PostgresChatStore":
        """Return connection string from database parameters."""
        conn_str = (
            connection_string
            or f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
        )
        async_conn_str = async_connection_string or (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        )
        session, async_session = cls._connect(conn_str, async_conn_str, debug)
        return cls(
            session=session,
            async_session=async_session,
            table_name=table_name,
            schema_name=schema_name,
            use_jsonb=use_jsonb,
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "chatstore",
        schema_name: str = "public",
        debug: bool = False,
        use_jsonb: bool = False,
    ) -> "PostgresChatStore":
        """Return connection string from database parameters."""
        params = params_from_uri(uri)
        return cls.from_params(
            **params,
            table_name=table_name,
            schema_name=schema_name,
            debug=debug,
            use_jsonb=use_jsonb,
        )

    @classmethod
    def _connect(
        cls, connection_string: str, async_connection_string: str, debug: bool
    ) -> tuple[sessionmaker, sessionmaker]:
        _engine = create_engine(connection_string, echo=debug)
        session = sessionmaker(_engine)

        _async_engine = create_async_engine(async_connection_string)
        async_session = sessionmaker(_async_engine, class_=AsyncSession)
        return session, async_session

    def _create_schema_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
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

    def _create_tables_if_not_exists(self, base) -> None:
        with self._session() as session, session.begin():
            base.metadata.create_all(session.connection())

    def _initialize(self, base) -> None:
        self._create_schema_if_not_exists()
        self._create_tables_if_not_exists(base)

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """Set messages for a key."""
        with self._session() as session:
            stmt = text(
                f"""
                INSERT INTO {self.schema_name}.{self._table_class.__tablename__} (key, value)
                VALUES (:key, :value)
                ON CONFLICT (key)
                DO UPDATE SET
                value = EXCLUDED.value;
                """
            )

            params = {
                "key": key,
                "value": [json.dumps(message.dict()) for message in messages],
            }

            # Execute the bulk upsert
            session.execute(stmt, params)
            session.commit()

    async def aset_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """Async version of Get messages for a key."""
        async with self._async_session() as session:
            stmt = text(
                f"""
                INSERT INTO {self.schema_name}.{self._table_class.__tablename__} (key, value)
                VALUES (:key, :value)
                ON CONFLICT (key)
                DO UPDATE SET
                value = EXCLUDED.value;
                """
            )

            params = {
                "key": key,
                "value": [json.dumps(message.dict()) for message in messages],
            }

            # Execute the bulk upsert
            await session.execute(stmt, params)
            await session.commit()

    def get_messages(self, key: str) -> list[ChatMessage]:
        """Get messages for a key."""
        with self._session() as session:
            result = session.execute(select(self._table_class).filter_by(key=key))
            result = result.scalars().first()
            if result:
                return [
                    ChatMessage.model_validate(removed_message)
                    for removed_message in result.value
                ]
            return []

    async def aget_messages(self, key: str) -> list[ChatMessage]:
        """Async version of Get messages for a key."""
        async with self._async_session() as session:
            result = await session.execute(select(self._table_class).filter_by(key=key))
            result = result.scalars().first()
            if result:
                return [
                    ChatMessage.model_validate(removed_message)
                    for removed_message in result.value
                ]
            return []

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        with self._session() as session:
            stmt = text(
                f"""
                INSERT INTO {self.schema_name}.{self._table_class.__tablename__} (key, value)
                VALUES (:key, :value)
                ON CONFLICT (key)
                DO UPDATE SET
                    value = array_cat({self._table_class.__tablename__}.value, :value);
                """
            )
            params = {"key": key, "value": [json.dumps(message.dict())]}
            session.execute(stmt, params)
            session.commit()

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """Async version of Add a message for a key."""
        async with self._async_session() as session:
            stmt = text(
                f"""
                INSERT INTO {self.schema_name}.{self._table_class.__tablename__} (key, value)
                VALUES (:key, :value)
                ON CONFLICT (key)
                DO UPDATE SET
                    value = array_cat({self._table_class.__tablename__}.value, :value);
                """
            )
            params = {"key": key, "value": [json.dumps(message.dict())]}
            await session.execute(stmt, params)
            await session.commit()

    def delete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """Delete messages for a key."""
        with self._session() as session:
            session.execute(delete(self._table_class).filter_by(key=key))
            session.commit()
        return None

    async def adelete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """Async version of Delete messages for a key."""
        async with self._async_session() as session:
            await session.execute(delete(self._table_class).filter_by(key=key))
            await session.commit()
        return None

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        with self._session() as session:
            # First, retrieve the current list of messages
            stmt = select(self._table_class.value).where(self._table_class.key == key)
            result = session.execute(stmt).scalar_one_or_none()

            if result is None or idx < 0 or idx >= len(result):
                # If the key doesn't exist or the index is out of bounds
                return None

            # Remove the message at the given index
            removed_message = result[idx]

            stmt = text(
                f"""
                UPDATE {self._table_class.__tablename__}
                SET value = array_cat(
                               {self._table_class.__tablename__}.value[: :idx],
                               {self._table_class.__tablename__}.value[:idx+2:]
                           )
                WHERE key = :key;
                """
            )

            params = {"key": key, "idx": idx}
            session.execute(stmt, params)
            session.commit()

            return ChatMessage.model_validate(removed_message)

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Async version of Delete specific message for a key."""
        async with self._async_session() as session:
            # First, retrieve the current list of messages
            stmt = select(self._table_class.value).where(self._table_class.key == key)
            result = (await session.execute(stmt)).scalar_one_or_none()

            if result is None or idx < 0 or idx >= len(result):
                # If the key doesn't exist or the index is out of bounds
                return None

            # Remove the message at the given index
            removed_message = result[idx]

            stmt = text(
                f"""
                UPDATE {self._table_class.__tablename__}
                SET value = array_cat(
                               {self._table_class.__tablename__}.value[: :idx],
                               {self._table_class.__tablename__}.value[:idx+2:]
                           )
                WHERE key = :key;
                """
            )

            params = {"key": key, "idx": idx}
            await session.execute(stmt, params)
            await session.commit()

            return ChatMessage.model_validate(removed_message)

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        with self._session() as session:
            # First, retrieve the current list of messages
            stmt = select(self._table_class.value).where(self._table_class.key == key)
            result = session.execute(stmt).scalar_one_or_none()

            if result is None or len(result) == 0:
                # If the key doesn't exist or the array is empty
                return None

            # Remove the message at the given index
            removed_message = result[-1]

            stmt = text(
                f"""
                UPDATE {self._table_class.__tablename__}
                SET value = value[1:array_length(value, 1) - 1]
                WHERE key = :key;
                """
            )
            params = {"key": key}
            session.execute(stmt, params)
            session.commit()

            return ChatMessage.model_validate(removed_message)

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Async version of Delete last message for a key."""
        async with self._async_session() as session:
            # First, retrieve the current list of messages
            stmt = select(self._table_class.value).where(self._table_class.key == key)
            result = (await session.execute(stmt)).scalar_one_or_none()

            if result is None or len(result) == 0:
                # If the key doesn't exist or the array is empty
                return None

            # Remove the message at the given index
            removed_message = result[-1]

            stmt = text(
                f"""
                        UPDATE {self._table_class.__tablename__}
                        SET value = value[1:array_length(value, 1) - 1]
                        WHERE key = :key;
                        """
            )
            params = {"key": key}
            await session.execute(stmt, params)
            await session.commit()

            return ChatMessage.model_validate(removed_message)

    def get_keys(self) -> list[str]:
        """Get all keys."""
        with self._session() as session:
            stmt = select(self._table_class.key)

            return session.execute(stmt).scalars().all()

    async def aget_keys(self) -> list[str]:
        """Async version of Get all keys."""
        async with self._async_session() as session:
            stmt = select(self._table_class.key)

            return (await session.execute(stmt)).scalars().all()


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
