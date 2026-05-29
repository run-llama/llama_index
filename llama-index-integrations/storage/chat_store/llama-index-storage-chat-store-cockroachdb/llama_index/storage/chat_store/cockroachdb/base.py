"""CockroachDB Chat store for LlamaIndex."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from sqlalchemy import (
    Column,
    Index,
    Integer,
    UniqueConstraint,
    create_engine,
    delete,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSON, JSONB, VARCHAR
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker


def get_data_model(
    base: type,
    index_name: str,
    schema_name: str,
    use_jsonb: bool = True,
) -> Any:
    """Build a dynamic SQLAlchemy model for the chat store table.

    On CockroachDB we store the list of messages as a single JSONB column
    holding a JSON array, because CRDB does not support ``ARRAY(JSON)`` as a
    column type (see crdb issue 23468).
    """
    index_name = index_name or "chatstore"
    tablename = index_name
    class_name = f"{index_name[0].upper()}{index_name[1:]}"

    chat_dtype = JSONB if use_jsonb else JSON

    class AbstractData(base):  # type: ignore[misc, valid-type]
        __abstract__ = True
        id = Column(Integer, primary_key=True, autoincrement=True)
        key = Column(VARCHAR, nullable=False)
        value = Column(chat_dtype)

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


def _serialize(messages: list[ChatMessage]) -> str:
    return json.dumps([json.loads(m.model_dump_json()) for m in messages])


def _deserialize(value: Any) -> list[dict]:
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return list(value)


class CockroachDBChatStore(BaseChatStore):
    """CockroachDB-backed Chat store.

    Each conversation key maps to a single ``JSONB`` cell containing a JSON
    array of messages. Uses the ``cockroachdb+psycopg2`` and
    ``cockroachdb+asyncpg`` SQLAlchemy dialects so retries on
    ``SERIALIZATION_FAILURE`` are transparent.
    """

    table_name: str | None = Field(default="chatstore", description="CRDB table name.")
    schema_name: str | None = Field(default="public", description="CRDB schema name.")

    _table_class: Any | None = PrivateAttr()
    _session: sessionmaker | None = PrivateAttr()
    _async_session: sessionmaker | None = PrivateAttr()

    def __init__(
        self,
        session: sessionmaker,
        async_session: sessionmaker,
        table_name: str,
        schema_name: str = "public",
        use_jsonb: bool = True,
    ) -> None:
        super().__init__(
            table_name=table_name.lower(),
            schema_name=schema_name.lower(),
        )

        base = declarative_base()
        self._table_class = get_data_model(
            base,
            self.table_name,
            self.schema_name,
            use_jsonb=use_jsonb,
        )
        self._session = session
        self._async_session = async_session
        self._initialize(base)

    @classmethod
    def class_name(cls) -> str:
        return "CockroachDBChatStore"

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
        table_name: str = "chatstore",
        schema_name: str = "public",
        connection_string: str | None = None,
        async_connection_string: str | None = None,
        debug: bool = False,
        use_jsonb: bool = True,
    ) -> CockroachDBChatStore:
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
        session, async_session = cls._connect(connection_string, async_connection_string, debug)
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
        use_jsonb: bool = True,
    ) -> CockroachDBChatStore:
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
        sync_engine = create_engine(connection_string, echo=debug)
        session = sessionmaker(sync_engine)
        async_engine = create_async_engine(async_connection_string)
        async_session = sessionmaker(async_engine, class_=AsyncSession)
        return session, async_session

    def _create_schema_if_not_exists(self) -> None:
        with self._session() as session, session.begin():
            inspector = inspect(session.connection())
            existing_schemas = inspector.get_schema_names()
            if self.schema_name not in existing_schemas:
                session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}"))

    def _create_tables_if_not_exists(self, base: type) -> None:
        with self._session() as session, session.begin():
            base.metadata.create_all(session.connection())

    def _initialize(self, base: type) -> None:
        self._create_schema_if_not_exists()
        self._create_tables_if_not_exists(base)

    def _qualified_table(self) -> str:
        return f"{self.schema_name}.{self._table_class.__tablename__}"

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        with self._session() as session:
            stmt = text(
                f"""
                INSERT INTO {self._qualified_table()} (key, value)
                VALUES (:key, CAST(:value AS JSONB))
                ON CONFLICT (key)
                DO UPDATE SET value = EXCLUDED.value;
                """
            )
            session.execute(stmt, {"key": key, "value": _serialize(messages)})
            session.commit()

    async def aset_messages(self, key: str, messages: list[ChatMessage]) -> None:
        async with self._async_session() as session:
            stmt = text(
                f"""
                INSERT INTO {self._qualified_table()} (key, value)
                VALUES (:key, CAST(:value AS JSONB))
                ON CONFLICT (key)
                DO UPDATE SET value = EXCLUDED.value;
                """
            )
            await session.execute(stmt, {"key": key, "value": _serialize(messages)})
            await session.commit()

    def get_messages(self, key: str) -> list[ChatMessage]:
        with self._session() as session:
            row = session.execute(select(self._table_class).filter_by(key=key)).scalars().first()
            if row is None:
                return []
            return [ChatMessage.model_validate(m) for m in _deserialize(row.value)]

    async def aget_messages(self, key: str) -> list[ChatMessage]:
        async with self._async_session() as session:
            result = await session.execute(select(self._table_class).filter_by(key=key))
            row = result.scalars().first()
            if row is None:
                return []
            return [ChatMessage.model_validate(m) for m in _deserialize(row.value)]

    def add_message(self, key: str, message: ChatMessage) -> None:
        # JSONB || operator concatenates two JSON arrays. Atomic at row level.
        with self._session() as session:
            stmt = text(
                f"""
                INSERT INTO {self._qualified_table()} (key, value)
                VALUES (:key, CAST(:value AS JSONB))
                ON CONFLICT (key)
                DO UPDATE SET value = COALESCE({self._qualified_table()}.value, '[]'::JSONB) || EXCLUDED.value;
                """
            )
            session.execute(
                stmt,
                {"key": key, "value": _serialize([message])},
            )
            session.commit()

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        async with self._async_session() as session:
            stmt = text(
                f"""
                INSERT INTO {self._qualified_table()} (key, value)
                VALUES (:key, CAST(:value AS JSONB))
                ON CONFLICT (key)
                DO UPDATE SET value = COALESCE({self._qualified_table()}.value, '[]'::JSONB) || EXCLUDED.value;
                """
            )
            await session.execute(
                stmt,
                {"key": key, "value": _serialize([message])},
            )
            await session.commit()

    def delete_messages(self, key: str) -> list[ChatMessage] | None:
        with self._session() as session:
            session.execute(delete(self._table_class).filter_by(key=key))
            session.commit()
        return None

    async def adelete_messages(self, key: str) -> list[ChatMessage] | None:
        async with self._async_session() as session:
            await session.execute(delete(self._table_class).filter_by(key=key))
            await session.commit()
        return None

    def delete_message(self, key: str, idx: int) -> ChatMessage | None:
        with self._session() as session:
            row = session.execute(select(self._table_class).filter_by(key=key)).scalars().first()
            if row is None:
                return None
            current = _deserialize(row.value)
            if idx < 0 or idx >= len(current):
                return None
            removed = current[idx]
            new_value = current[:idx] + current[idx + 1 :]
            update = text(
                f"""
                UPDATE {self._qualified_table()}
                SET value = CAST(:value AS JSONB)
                WHERE key = :key;
                """
            )
            session.execute(update, {"key": key, "value": json.dumps(new_value)})
            session.commit()
            return ChatMessage.model_validate(removed)

    async def adelete_message(self, key: str, idx: int) -> ChatMessage | None:
        async with self._async_session() as session:
            result = await session.execute(select(self._table_class).filter_by(key=key))
            row = result.scalars().first()
            if row is None:
                return None
            current = _deserialize(row.value)
            if idx < 0 or idx >= len(current):
                return None
            removed = current[idx]
            new_value = current[:idx] + current[idx + 1 :]
            update = text(
                f"""
                UPDATE {self._qualified_table()}
                SET value = CAST(:value AS JSONB)
                WHERE key = :key;
                """
            )
            await session.execute(update, {"key": key, "value": json.dumps(new_value)})
            await session.commit()
            return ChatMessage.model_validate(removed)

    def delete_last_message(self, key: str) -> ChatMessage | None:
        with self._session() as session:
            row = session.execute(select(self._table_class).filter_by(key=key)).scalars().first()
            if row is None:
                return None
            current = _deserialize(row.value)
            if not current:
                return None
            removed = current[-1]
            new_value = current[:-1]
            update = text(
                f"""
                UPDATE {self._qualified_table()}
                SET value = CAST(:value AS JSONB)
                WHERE key = :key;
                """
            )
            session.execute(update, {"key": key, "value": json.dumps(new_value)})
            session.commit()
            return ChatMessage.model_validate(removed)

    async def adelete_last_message(self, key: str) -> ChatMessage | None:
        async with self._async_session() as session:
            result = await session.execute(select(self._table_class).filter_by(key=key))
            row = result.scalars().first()
            if row is None:
                return None
            current = _deserialize(row.value)
            if not current:
                return None
            removed = current[-1]
            new_value = current[:-1]
            update = text(
                f"""
                UPDATE {self._qualified_table()}
                SET value = CAST(:value AS JSONB)
                WHERE key = :key;
                """
            )
            await session.execute(update, {"key": key, "value": json.dumps(new_value)})
            await session.commit()
            return ChatMessage.model_validate(removed)

    def get_keys(self) -> list[str]:
        with self._session() as session:
            return list(session.execute(select(self._table_class.key)).scalars().all())

    async def aget_keys(self) -> list[str]:
        async with self._async_session() as session:
            return list((await session.execute(select(self._table_class.key))).scalars().all())


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
