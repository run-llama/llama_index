import time
from typing import List, Optional
from enum import Enum

from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    delete,
    select,
    insert,
    update,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.chat_store.base_db import AsyncDBChatStore


DEFAULT_ASYNC_DATABASE_URI = "sqlite+aiosqlite:///:memory:"
Base = declarative_base()


class MessageStatus(str, Enum):
    """Status of a message in the chat store."""

    # Message is in the active FIFO queue
    ACTIVE = "active"

    # Message has been processed and is archived, removed from the active queue
    ARCHIVED = "archived"


class SQLAlchemyChatStore(AsyncDBChatStore):
    """Base class for SQLAlchemy-based chat stores.

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

    _async_engine: Optional[AsyncEngine] = PrivateAttr(default=None)
    _async_session_factory: Optional[sessionmaker] = PrivateAttr(default=None)
    _metadata: MetaData = PrivateAttr(default_factory=MetaData)
    _table: Optional[Table] = PrivateAttr(default=None)
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(
        self,
        table_name: str,
        async_database_uri: Optional[str] = DEFAULT_ASYNC_DATABASE_URI,
        async_engine: Optional[AsyncEngine] = None,
    ):
        """Initialize the SQLAlchemy chat store."""
        super().__init__(
            table_name=table_name,
            async_database_uri=async_database_uri or DEFAULT_ASYNC_DATABASE_URI,
        )
        self._async_engine = async_engine

    async def _initialize(self) -> None:
        """Initialize the chat store. Used to avoid HTTP connections in constructor."""
        if self._is_initialized:
            return

        await self._setup_connections()
        await self._setup_tables()
        self._is_initialized = True

    async def _setup_connections(self) -> None:
        """Set up database connections and session factories."""
        # Create async engine and session factory if async URI is provided
        if self.async_database_uri or self._async_engine:
            self._async_engine = self._async_engine or create_async_engine(
                self.async_database_uri
            )
            if self.async_database_uri is None:
                self.async_database_uri = self._async_engine.url

            self._async_session_factory = sessionmaker(
                bind=self._async_engine, class_=AsyncSession
            )

    async def _setup_tables(self) -> None:
        """Set up database tables."""
        # Create messages table with status column
        self._table = Table(
            f"{self.table_name}",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("key", String, nullable=False, index=True),
            Column("timestamp", Integer, nullable=False, index=True),
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
        async with self._async_engine.begin() as conn:
            await conn.run_sync(self._metadata.create_all)

    async def get_messages(
        self,
        key: str,
        status: Optional[MessageStatus] = MessageStatus.ACTIVE,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ChatMessage]:
        """Get all messages for a key with the specified status (async).

        Returns a list of messages.
        """
        await self._initialize()

        query = select(self._table).where(self._table.c.key == key)

        if limit is not None:
            query = query.limit(limit)

        if offset is not None:
            query = query.offset(offset)

        if status is not None:
            query = query.where(self._table.c.status == status.value)

        async with self._async_session_factory() as session:
            result = await session.execute(
                query.order_by(self._table.c.timestamp, self._table.c.id)
            )
            rows = result.fetchall()

            return [ChatMessage.model_validate(row.data) for row in rows]

    async def count_messages(
        self,
        key: str,
        status: Optional[MessageStatus] = MessageStatus.ACTIVE,
    ) -> int:
        """Count messages for a key with the specified status (async)."""
        await self._initialize()

        query = select(self._table.c.id).where(self._table.c.key == key)

        if status is not None:
            query = query.where(self._table.c.status == status.value)

        async with self._async_session_factory() as session:
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
        await self._initialize()

        async with self._async_session_factory() as session:
            await session.execute(
                insert(self._table).values(
                    key=key,
                    timestamp=int(time.time()),
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
        await self._initialize()

        async with self._async_session_factory() as session:
            await session.execute(
                insert(self._table).values(
                    [
                        {
                            "key": key,
                            "timestamp": int(time.time()),
                            "role": message.role,
                            "status": status.value,
                            "data": message.model_dump(mode="json"),
                        }
                        for message in messages
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
        await self._initialize()

        # First delete all existing messages
        await self.delete_messages(key)

        # Then add new messages
        current_time = int(time.time())

        async with self._async_session_factory() as session:
            for i, message in enumerate(messages):
                await session.execute(
                    insert(self._table).values(
                        key=key,
                        timestamp=current_time
                        + i,  # Preserve order with incremental timestamps
                        role=message.role,
                        status=status.value,
                        data=message.model_dump(mode="json"),
                    )
                )
            await session.commit()

    async def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Delete a specific message by ID and return it (async)."""
        await self._initialize()

        async with self._async_session_factory() as session:
            # First get the message
            result = await session.execute(
                select(self._table).where(
                    self._table.c.key == key, self._table.c.id == idx
                )
            )
            row = result.fetchone()

            if not row:
                return None

            # Store the message we're about to delete
            message = ChatMessage.model_validate(row.data)

            # Delete the message
            await session.execute(delete(self._table).where(self._table.c.id == idx))
            await session.commit()

            return message

    async def delete_messages(
        self, key: str, status: Optional[MessageStatus] = None
    ) -> None:
        """Delete all messages for a key with the specified status (async)."""
        await self._initialize()

        query = delete(self._table).where(self._table.c.key == key)

        if status is not None:
            query = query.where(self._table.c.status == status.value)

        async with self._async_session_factory() as session:
            await session.execute(query)
            await session.commit()

    async def delete_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        """Delete the oldest n messages for a key and return them (async)."""
        await self._initialize()

        oldest_messages = []

        async with self._async_session_factory() as session:
            # First get the oldest n messages
            result = await session.execute(
                select(self._table)
                .where(
                    self._table.c.key == key,
                    self._table.c.status == MessageStatus.ACTIVE.value,
                )
                .order_by(self._table.c.timestamp, self._table.c.id)
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
            await session.execute(
                delete(self._table).where(self._table.c.id.in_(ids_to_delete))
            )
            await session.commit()

        return oldest_messages

    async def archive_oldest_messages(self, key: str, n: int) -> List[ChatMessage]:
        """Archive the oldest n messages for a key and return them (async)."""
        await self._initialize()

        async with self._async_session_factory() as session:
            # First get the oldest n messages
            result = await session.execute(
                select(self._table)
                .where(
                    self._table.c.key == key,
                    self._table.c.status == MessageStatus.ACTIVE.value,
                )
                .order_by(self._table.c.timestamp, self._table.c.id)
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
                update(self._table)
                .where(self._table.c.id.in_(ids_to_archive))
                .values(status=MessageStatus.ARCHIVED.value)
            )
            await session.commit()

        return archived_messages

    async def get_keys(self) -> List[str]:
        """Get all unique keys in the store (async)."""
        await self._initialize()

        async with self._async_session_factory() as session:
            result = await session.execute(select(self._table.c.key).distinct())
            return [row[0] for row in result.fetchall()]

    @classmethod
    def class_name(cls) -> str:
        """Return the class name."""
        return "SQLAlchemyChatStore"
