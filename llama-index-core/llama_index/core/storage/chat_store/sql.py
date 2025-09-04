import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (
    JSON,
    Column,
    BigInteger,
    Integer,
    MetaData,
    String,
    Table,
    delete,
    select,
    insert,
    update,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from llama_index.core.async_utils import asyncio_run
from llama_index.core.bridge.pydantic import Field, PrivateAttr, model_serializer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.chat_store.base_db import AsyncDBChatStore, MessageStatus


DEFAULT_ASYNC_DATABASE_URI = "sqlite+aiosqlite:///:memory:"
Base = declarative_base()


class SQLAlchemyChatStore(AsyncDBChatStore):
    """
    Base class for SQLAlchemy-based chat stores.

    This class provides a foundation for creating chat stores that use SQLAlchemy
    to interact with SQL databases. It handles common operations like managing
    sessions, creating tables, and CRUD operations on chat messages.

    Enhanced with status tracking for better FIFO queue management for short-term memory.

    This class is meant to replace all other chat store classes.
    """

    table_name: str = Field(description="Name of the table to store messages")
    async_database_uri: str = Field(
        default=DEFAULT_ASYNC_DATABASE_URI,
        description="SQLAlchemy async connection URI",
    )
    db_schema: Optional[str] = Field(
        default=None,
        description="Database schema name (for PostgreSQL and other databases that support schemas)",
    )

    _async_engine: Optional[AsyncEngine] = PrivateAttr(default=None)
    _async_session_factory: Optional[sessionmaker] = PrivateAttr(default=None)
    _metadata: MetaData = PrivateAttr(default_factory=MetaData)
    _table: Optional[Table] = PrivateAttr(default=None)
    _db_data: Optional[List[Dict[str, Any]]] = PrivateAttr(default=None)

    def __init__(
        self,
        table_name: str,
        async_database_uri: Optional[str] = DEFAULT_ASYNC_DATABASE_URI,
        async_engine: Optional[AsyncEngine] = None,
        db_data: Optional[List[Dict[str, Any]]] = None,
        db_schema: Optional[str] = None,
    ):
        """Initialize the SQLAlchemy chat store."""
        super().__init__(
            table_name=table_name,
            async_database_uri=async_database_uri or DEFAULT_ASYNC_DATABASE_URI,
            db_schema=db_schema,
        )
        self._async_engine = async_engine
        self._db_data = db_data

    @staticmethod
    def _is_in_memory_uri(uri: Optional[str]) -> bool:
        """Check if the URI points to an in-memory SQLite database."""
        # Handles both :memory: and empty path which also means in-memory for sqlite
        return uri == "sqlite+aiosqlite:///:memory:" or uri == "sqlite+aiosqlite://"

    def _is_sqlite_database(self) -> bool:
        """Check if the database is SQLite (which doesn't support schemas)."""
        return self.async_database_uri.startswith("sqlite")

    async def _initialize(self) -> Tuple[sessionmaker, Table]:
        """Initialize the chat store. Used to avoid HTTP connections in constructor."""
        if self._async_session_factory is not None and self._table is not None:
            return self._async_session_factory, self._table

        async_engine, async_session_factory = await self._setup_connections()
        table = await self._setup_tables(async_engine)

        # Restore data from in-memory database if provided
        if self._db_data:
            async with async_session_factory() as session:
                await session.execute(insert(table).values(self._db_data))
                await session.commit()

                # clear the data after it's inserted
                self._db_data = None

        return async_session_factory, table

    async def _setup_connections(
        self,
    ) -> Tuple[AsyncEngine, sessionmaker]:
        """Set up database connections and session factories."""
        # Create async engine and session factory if async URI is provided
        if self._async_session_factory is not None and self._async_engine is not None:
            return self._async_engine, self._async_session_factory
        elif self.async_database_uri or self._async_engine:
            self._async_engine = self._async_engine or create_async_engine(
                self.async_database_uri
            )
            if self.async_database_uri is None:
                self.async_database_uri = self._async_engine.url

            self._async_session_factory = sessionmaker(  # type: ignore
                bind=self._async_engine, expire_on_commit=False, class_=AsyncSession
            )

            return self._async_engine, self._async_session_factory  # type: ignore
        else:
            raise ValueError(
                "No async database URI or engine provided, cannot initialize DB sessionmaker"
            )

    async def _setup_tables(self, async_engine: AsyncEngine) -> Table:
        """Set up database tables."""
        # Create metadata with schema
        if self.db_schema is not None and not self._is_sqlite_database():
            # Only set schema for databases that support it
            self._metadata = MetaData(schema=self.db_schema)

            # Create schema if it doesn't exist (PostgreSQL, SQL Server, etc.)
            async with async_engine.begin() as conn:
                await conn.execute(
                    text(f'CREATE SCHEMA IF NOT EXISTS "{self.db_schema}"')
                )

        # Create messages table with status column
        self._table = Table(
            f"{self.table_name}",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("key", String, nullable=False, index=True),
            Column("timestamp", BigInteger, nullable=False, index=True),
            Column("role", String, nullable=False),
            Column(
                "status",
                String,
                nullable=False,
                default=MessageStatus.ACTIVE.value,
                index=True,
            ),
            Column("data", JSON, nullable=False),
        )

        # Create tables in the database
        async with async_engine.begin() as conn:
            await conn.run_sync(self._metadata.create_all)

        return self._table

    async def get_messages(
        self,
        key: str,
        status: Optional[MessageStatus] = MessageStatus.ACTIVE,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get all messages for a key with the specified status (async).

        Returns a list of messages.
        """
        session_factory, table = await self._initialize()

        query = select(table).where(table.c.key == key)

        if limit is not None:
            query = query.limit(limit)

        if offset is not None:
            query = query.offset(offset)

        if status is not None:
            query = query.where(table.c.status == status.value)

        async with session_factory() as session:
            result = await session.execute(
                query.order_by(table.c.timestamp, table.c.id)
            )
            rows = result.fetchall()

            return [ChatMessage.model_validate(row.data) for row in rows]

    async def count_messages(
        self,
        key: str,
        status: Optional[MessageStatus] = MessageStatus.ACTIVE,
    ) -> int:
        """Count messages for a key with the specified status (async)."""
        session_factory, table = await self._initialize()

        query = select(table.c.id).where(table.c.key == key)

        if status is not None:
            query = query.where(table.c.status == status.value)

        async with session_factory() as session:
            result = await session.execute(query)
            rows = result.fetchall()
            return len(rows)

    async def add_message(
        self,
        key: str,
        message: ChatMessage,
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Add a message for a key with the specified status (async)."""
        session_factory, table = await self._initialize()

        async with session_factory() as session:
            await session.execute(
                insert(table).values(
                    key=key,
                    timestamp=time.time_ns(),
                    role=message.role,
                    status=status.value,
                    data=message.model_dump(mode="json"),
                )
            )
            await session.commit()

    async def add_messages(
        self,
        key: str,
        messages: List[ChatMessage],
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Add a list of messages in batch for the specified key and status (async)."""
        session_factory, table = await self._initialize()

        async with session_factory() as session:
            await session.execute(
                insert(table).values(
                    [
                        {
                            "key": key,
                            "timestamp": time.time_ns() + i,
                            "role": message.role,
                            "status": status.value,
                            "data": message.model_dump(mode="json"),
                        }
                        for i, message in enumerate(messages)
                    ]
                )
            )
            await session.commit()

    async def set_messages(
        self,
        key: str,
        messages: List[ChatMessage],
        status: MessageStatus = MessageStatus.ACTIVE,
    ) -> None:
        """Set all messages for a key (replacing existing ones) with the specified status (async)."""
        session_factory, table = await self._initialize()

        # First delete all existing messages
        await self.delete_messages(key)

        # Then add new messages
        current_time = time.time_ns()

        async with session_factory() as session:
            for i, message in enumerate(messages):
                await session.execute(
                    insert(table).values(
                        key=key,
                        # Preserve order with incremental timestamps
                        timestamp=current_time + i,
                        role=message.role,
                        status=status.value,
                        data=message.model_dump(mode="json"),
                    )
                )
            await session.commit()

    async def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a specific message by ID and return it (async)."""
        session_factory, table = await self._initialize()

        async with session_factory() as session:
            # First get the message
            result = await session.execute(
                select(table).where(table.c.key == key, table.c.id == idx)
            )
            row = result.fetchone()

            if not row:
                return None

            # Store the message we're about to delete
            message = ChatMessage.model_validate(row.data)

            # Delete the message
            await session.execute(delete(table).where(table.c.id == idx))
            await session.commit()

            return message

    async def delete_messages(
        self, key: str, status: Optional[MessageStatus] = None
    ) -> None:
        """Delete all messages for a key with the specified status (async)."""
        session_factory, table = await self._initialize()

        query = delete(table).where(table.c.key == key)

        if status is not None:
            query = query.where(table.c.status == status.value)

        async with session_factory() as session:
            await session.execute(query)
            await session.commit()

    async def delete_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        """Delete the oldest n messages for a key and return them (async)."""
        session_factory, table = await self._initialize()

        oldest_messages = []

        async with session_factory() as session:
            # First get the oldest n messages
            result = await session.execute(
                select(table)
                .where(
                    table.c.key == key,
                    table.c.status == MessageStatus.ACTIVE.value,
                )
                .order_by(table.c.timestamp, table.c.id)
                .limit(n)
            )
            rows = result.fetchall()

            if not rows:
                return []

            # Store the messages we're about to delete
            oldest_messages = [ChatMessage.model_validate(row.data) for row in rows]

            # Get the IDs to delete
            ids_to_delete = [row.id for row in rows]

            # Delete the messages
            await session.execute(delete(table).where(table.c.id.in_(ids_to_delete)))
            await session.commit()

        return oldest_messages

    async def archive_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        """Archive the oldest n messages for a key and return them (async)."""
        session_factory, table = await self._initialize()

        async with session_factory() as session:
            # First get the oldest n messages
            result = await session.execute(
                select(table)
                .where(
                    table.c.key == key,
                    table.c.status == MessageStatus.ACTIVE.value,
                )
                .order_by(table.c.timestamp, table.c.id)
                .limit(n)
            )
            rows = result.fetchall()

            if not rows:
                return []

            # Store the messages we're about to archive
            archived_messages = [ChatMessage.model_validate(row.data) for row in rows]

            # Get the IDs to archive
            ids_to_archive = [row.id for row in rows]

            # Update message status to archived
            await session.execute(
                update(table)
                .where(table.c.id.in_(ids_to_archive))
                .values(status=MessageStatus.ARCHIVED.value)
            )
            await session.commit()

        return archived_messages

    async def get_keys(self) -> List[str]:
        """Get all unique keys in the store (async)."""
        session_factory, table = await self._initialize()

        async with session_factory() as session:
            result = await session.execute(select(table.c.key).distinct())
            return [row[0] for row in result.fetchall()]

    async def _dump_db_data(self) -> List[Dict[str, Any]]:
        """Dump the data from the database."""
        session_factory, table = await self._initialize()

        async with session_factory() as session:
            result = await session.execute(select(table))
            rows = result.fetchall()
            return [
                {
                    "key": row.key,
                    "timestamp": row.timestamp,
                    "role": row.role,
                    "status": row.status,
                    "data": row.data,
                }
                for row in rows
            ]

    @model_serializer()
    def dump_store(self) -> dict:
        """
        Dump the store's configuration and data (if in-memory).

        Returns:
            A dictionary containing the store's configuration and potentially its data.

        """
        dump_data = {
            "table_name": self.table_name,
            "async_database_uri": self.async_database_uri,
            "db_schema": self.db_schema,
        }

        if self._is_in_memory_uri(self.async_database_uri):
            # switch to sync sqlite
            dump_data["db_data"] = asyncio_run(self._dump_db_data())

        return dump_data

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "SQLAlchemyChatStore"
