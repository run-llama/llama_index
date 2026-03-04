from typing import Optional
from urllib.parse import urlparse

import sqlalchemy
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from packaging import version

SQLALCHEMY_VERSION = version.parse(sqlalchemy.__version__).release
SQLALCHEMY_1_4_0_PLUS = (1, 4, 0) <= SQLALCHEMY_VERSION < (2, 0, 0)
SQLALCHEMY_2_0_0_PLUS = SQLALCHEMY_VERSION >= (2, 0, 0)

if SQLALCHEMY_1_4_0_PLUS:
    from sqlalchemy.engine.create import create_engine
    from sqlalchemy.ext.asyncio.engine import create_async_engine
    from sqlalchemy.ext.asyncio.session import AsyncSession
    from sqlalchemy.orm import DeclarativeMeta as DeclarativeBase
    from sqlalchemy.orm import declarative_base
    from sqlalchemy.orm.session import sessionmaker
    from sqlalchemy.orm.session import sessionmaker as async_sessionmaker
    from sqlalchemy.sql.expression import delete, insert, select
    from sqlalchemy.sql.schema import Column, Table
    from sqlalchemy.types import JSON, Integer, String

elif SQLALCHEMY_2_0_0_PLUS:
    from sqlalchemy import (
        JSON,
        Column,
        Integer,
        String,
        Table,
        create_engine,
        delete,
        insert,
        select,
    )
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import DeclarativeBase, declarative_base, sessionmaker

else:
    raise ImportError(f"Unsupported version of sqlalchemy: {SQLALCHEMY_VERSION}")


class TableProtocol(Table):
    """A table protocol class for typing."""

    id: Column
    key: Column  # type: ignore
    value: Column


if SQLALCHEMY_1_4_0_PLUS:

    def get_data_model(
        base: DeclarativeBase,
        index_name: str,
    ) -> TableProtocol:
        """
        This part create a dynamic sqlalchemy model with a new table.
        """
        class_name = f"Data{index_name}"  # dynamic class name

        class AbstractData(base):  # type: ignore
            __tablename__ = f"data_{index_name}"  # dynamic table name
            __abstract__ = True  # this line is necessary

            id = Column(
                Integer(),
                primary_key=True,
                autoincrement=True,
                index=True,
            )  # Add primary key
            key = Column(
                String(),
                nullable=False,
                index=True,
            )
            value = Column(JSON())

        return type(
            class_name,
            (AbstractData,),
            {},
        )  # type: ignore


elif SQLALCHEMY_2_0_0_PLUS:
    from sqlalchemy.orm import Mapped, mapped_column

    def get_data_model(
        base: DeclarativeBase,
        index_name: str,
    ) -> TableProtocol:
        """
        This part create a dynamic sqlalchemy model with a new table.
        """
        class_name = f"Data{index_name}"  # dynamic class name

        class AbstractData(base):  # type: ignore
            __tablename__ = f"data_{index_name}"  # dynamic table name
            __abstract__ = True  # this line is necessary

            id: Mapped[int] = mapped_column(
                Integer(),
                primary_key=True,
                autoincrement=True,
                index=True,
            )  # Add primary key
            key: Mapped[str] = mapped_column(
                String(),
                nullable=False,
                index=True,
            )
            value: Mapped[str] = mapped_column(JSON())

        return type(
            class_name,
            (AbstractData,),
            {},
        )  # type: ignore


class SQLiteChatStore(BaseChatStore):
    table_name: Optional[str] = Field(
        default="chatstore", description="SQLite table name."
    )

    _table_class: TableProtocol = PrivateAttr()  # type: ignore
    _session: sessionmaker = PrivateAttr()  # type: ignore
    _async_session: async_sessionmaker = PrivateAttr()  # type: ignore

    def __init__(
        self,
        session: sessionmaker,
        async_session: async_sessionmaker,
        table_name: str,
    ):
        super().__init__(
            table_name=table_name.lower(),
        )

        # sqlalchemy model
        base = declarative_base()
        self._table_class = get_data_model(
            base,
            table_name,
        )
        self._session = session
        self._async_session = async_session
        self._initialize(base)

    @classmethod
    def from_params(
        cls,
        database: str,
        table_name: str = "chatstore",
        connection_string: Optional[str] = None,
        async_connection_string: Optional[str] = None,
        debug: bool = False,
    ) -> "SQLiteChatStore":
        """Return connection string from database parameters."""
        conn_str = connection_string or f"sqlite:///{database}"
        async_conn_str = async_connection_string or (f"sqlite+aiosqlite:///{database}")
        session, async_session = cls._connect(conn_str, async_conn_str, debug)
        return cls(
            session=session,
            async_session=async_session,
            table_name=table_name,
        )

    @classmethod
    def from_uri(
        cls,
        uri: str,
        table_name: str = "chatstore",
        debug: bool = False,
    ) -> "SQLiteChatStore":
        """Return connection string from database parameters."""
        params = params_from_uri(uri)
        return cls.from_params(
            **params,
            table_name=table_name,
            debug=debug,
        )

    @classmethod
    def _connect(
        cls, connection_string: str, async_connection_string: str, debug: bool
    ) -> tuple[sessionmaker, async_sessionmaker]:
        _engine = create_engine(connection_string, echo=debug)
        session = sessionmaker(_engine)

        _async_engine = create_async_engine(async_connection_string)
        async_session = async_sessionmaker(_async_engine, class_=AsyncSession)
        return session, async_session

    def _create_tables_if_not_exists(self, base) -> None:
        with self._session() as session, session.begin():
            base.metadata.create_all(session.connection())

    def _initialize(self, base) -> None:
        self._create_tables_if_not_exists(base)

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """Set messages for a key."""
        with self._session() as session:
            session.execute(
                insert(self._table_class),
                [
                    {"key": key, "value": message.model_dump(mode="json")}
                    for message in messages
                ],
            )
            session.commit()

    async def aset_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """Async version of Get messages for a key."""
        async with self._async_session() as session:
            await session.execute(
                insert(self._table_class),
                [
                    {"key": key, "value": message.model_dump(mode="json")}
                    for message in messages
                ],
            )
            await session.commit()

    def get_messages(self, key: str) -> list[ChatMessage]:
        """Get messages for a key."""
        with self._session() as session:
            result = session.execute(
                select(self._table_class)
                .where(self._table_class.key == key)
                .order_by(self._table_class.id)
            )
            result = result.scalars().all()
            if result:
                return [ChatMessage.model_validate(row.value) for row in result]
            return []

    async def aget_messages(self, key: str) -> list[ChatMessage]:
        """Async version of Get messages for a key."""
        async with self._async_session() as session:
            result = await session.execute(
                select(self._table_class)
                .where(self._table_class.key == key)
                .order_by(self._table_class.id)
            )
            result = result.scalars().all()
            if result:
                return [ChatMessage.model_validate(row.value) for row in result]
            return []

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Add a message for a key."""
        with self._session() as session:
            session.execute(
                insert(self._table_class),
                [{"key": key, "value": message.model_dump(mode="json")}],
            )
            session.commit()

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """Async version of Add a message for a key."""
        async with self._async_session() as session:
            await session.execute(
                insert(self._table_class),
                [{"key": key, "value": message.model_dump(mode="json")}],
            )
            await session.commit()

    def delete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """Delete messages for a key."""
        with self._session() as session:
            session.execute(
                delete(self._table_class).where(self._table_class.key == key)
            )
            session.commit()
        return None

    async def adelete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """Async version of Delete messages for a key."""
        async with self._async_session() as session:
            await session.execute(
                delete(self._table_class).where(self._table_class.key == key)
            )
            await session.commit()
        return None

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete specific message for a key."""
        with self._session() as session:
            # First, retrieve message
            result = session.execute(
                select(self._table_class.value).where(
                    self._table_class.key == key, self._table_class.id == idx
                )
            ).scalar_one_or_none()

            if result is None:
                return None

            session.execute(
                delete(self._table_class).where(
                    self._table_class.key == key, self._table_class.id == idx
                )
            )
            session.commit()

            return ChatMessage.model_validate(result)

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Async version of Delete specific message for a key."""
        async with self._async_session() as session:
            # First, retrieve message
            result = (
                await session.execute(
                    select(self._table_class.value).where(
                        self._table_class.key == key, self._table_class.id == idx
                    )
                )
            ).scalar_one_or_none()

            if result is None:
                return None

            await session.execute(
                delete(self._table_class).where(
                    self._table_class.key == key, self._table_class.id == idx
                )
            )
            await session.commit()

            return ChatMessage.model_validate(result)

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Delete last message for a key."""
        with self._session() as session:
            # First, retrieve the current list of messages
            stmt = (
                select(self._table_class.id, self._table_class.value)
                .where(self._table_class.key == key)
                .order_by(self._table_class.id.desc())
                .limit(1)
            )
            result = session.execute(stmt).all()

            if not result:
                # If the key doesn't exist or the array is empty
                return None

            session.execute(
                delete(self._table_class).where(self._table_class.id == result[0][0])
            )
            session.commit()

            return ChatMessage.model_validate(result[0][1])

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Async version of Delete last message for a key."""
        async with self._async_session() as session:
            # First, retrieve the current list of messages
            stmt = (
                select(self._table_class.id, self._table_class.value)
                .where(self._table_class.key == key)
                .order_by(self._table_class.id.desc())
                .limit(1)
            )
            result = (await session.execute(stmt)).all()

            if not result:
                # If the key doesn't exist or the array is empty
                return None

            await session.execute(
                delete(self._table_class).where(self._table_class.id == result[0][0])
            )
            await session.commit()

            return ChatMessage.model_validate(result[0][1])

    def get_keys(self) -> list[str]:
        """Get all keys."""
        with self._session() as session:
            stmt = select(self._table_class.key.distinct())

            return session.execute(stmt).scalars().all()

    async def aget_keys(self) -> list[str]:
        """Async version of Get all keys."""
        async with self._async_session() as session:
            stmt = select(self._table_class.key.distinct())

            return (await session.execute(stmt)).scalars().all()


def params_from_uri(uri: str) -> dict:
    result = urlparse(uri)
    database = result.path[1:]
    return {"database": database}
