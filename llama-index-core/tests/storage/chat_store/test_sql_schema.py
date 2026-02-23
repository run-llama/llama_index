"""Tests for SQLAlchemyChatStore schema functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.storage.chat_store.sql import SQLAlchemyChatStore


class TestSQLAlchemyChatStoreSchema:
    """Test schema functionality in SQLAlchemyChatStore."""

    def test_schema_parameter_initialization(self):
        """Test schema parameter initialization."""
        # Without schema
        store_no_schema = SQLAlchemyChatStore(
            table_name="test_messages",
            async_database_uri="sqlite+aiosqlite:///:memory:",
        )
        assert store_no_schema.db_schema is None

        # With schema
        store_with_schema = SQLAlchemyChatStore(
            table_name="test_messages",
            async_database_uri="sqlite+aiosqlite:///:memory:",
            db_schema="test_schema",
        )
        assert store_with_schema.db_schema == "test_schema"

    def test_schema_serialization(self):
        """Test that schema is included in serialization."""
        store = SQLAlchemyChatStore(
            table_name="test_table",
            async_database_uri="postgresql+asyncpg://user:pass@host/db",
            db_schema="test_schema",
        )

        # Test dump_store
        dumped = store.dump_store()
        assert "db_schema" in dumped
        assert dumped["db_schema"] == "test_schema"

        # Test model serialization
        store_dict = store.model_dump()
        assert "db_schema" in store_dict
        assert store_dict["db_schema"] == "test_schema"

    @pytest.mark.asyncio
    async def test_postgresql_schema_creation(self):
        """Test that CREATE SCHEMA SQL is called for PostgreSQL."""
        store = SQLAlchemyChatStore(
            table_name="test_messages",
            async_database_uri="postgresql+asyncpg://user:pass@host/db",
            db_schema="test_schema",
        )

        # Mock the engine and connection
        async_engine = MagicMock()
        async_engine.begin.return_value.__aenter__ = AsyncMock()
        async_engine.begin.return_value.__aexit__ = AsyncMock()

        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.run_sync = AsyncMock()

        async_engine.begin.return_value.__aenter__.return_value = mock_conn
        store._async_engine = async_engine

        # Call _setup_tables
        await store._setup_tables(async_engine)

        # Verify schema creation was called
        mock_conn.execute.assert_called()
        call_args = mock_conn.execute.call_args_list[0][0][0]
        assert 'CREATE SCHEMA IF NOT EXISTS "test_schema"' in str(call_args)

        # Verify MetaData has schema
        assert store._metadata.schema == "test_schema"

    @pytest.mark.asyncio
    async def test_sqlite_schema_behavior(self):
        """Test that SQLite preserves schema parameter but doesn't use it in MetaData."""
        store = SQLAlchemyChatStore(
            table_name="test_messages",
            async_database_uri="sqlite+aiosqlite:///:memory:",
            db_schema="test_schema",
        )

        # Add a message to trigger initialization
        await store.add_message("test_user", ChatMessage(role="user", content="Hello!"))

        # Schema parameter is preserved
        assert store.db_schema == "test_schema"

        # But MetaData doesn't have schema (SQLite limitation)
        assert store._metadata.schema is None

        # Operations still work
        messages = await store.get_messages("test_user")
        assert len(messages) == 1
        assert messages[0].content == "Hello!"

    def test_is_sqlite_database_with_custom_engine(self):
        """Test that _is_sqlite_database checks the engine URL when a custom engine is provided.

        Regression test for https://github.com/run-llama/llama_index/issues/20746
        When a custom async_engine is passed without an explicit async_database_uri,
        the URI defaults to SQLite. _is_sqlite_database() should check the actual
        engine URL instead of the default URI.
        """
        # Simulate a PostgreSQL engine passed without explicit URI
        mock_pg_engine = MagicMock()
        mock_pg_engine.url = "postgresql+asyncpg://user:pass@host/db"

        store = SQLAlchemyChatStore(
            table_name="test_messages",
            async_engine=mock_pg_engine,
            db_schema="test_schema",
        )

        # The default URI is SQLite, but the engine is PostgreSQL
        assert store.async_database_uri.startswith("sqlite")
        # _is_sqlite_database should check the engine, not the default URI
        assert not store._is_sqlite_database()

    def test_is_sqlite_database_with_sqlite_engine(self):
        """Test that _is_sqlite_database returns True for an actual SQLite engine."""
        mock_sqlite_engine = MagicMock()
        mock_sqlite_engine.url = "sqlite+aiosqlite:///:memory:"

        store = SQLAlchemyChatStore(
            table_name="test_messages",
            async_engine=mock_sqlite_engine,
            db_schema="test_schema",
        )

        assert store._is_sqlite_database()

    def test_is_sqlite_database_without_engine(self):
        """Test that _is_sqlite_database falls back to URI when no engine is provided."""
        store = SQLAlchemyChatStore(
            table_name="test_messages",
            async_database_uri="postgresql+asyncpg://user:pass@host/db",
        )
        assert not store._is_sqlite_database()

        store_sqlite = SQLAlchemyChatStore(
            table_name="test_messages",
            async_database_uri="sqlite+aiosqlite:///:memory:",
        )
        assert store_sqlite._is_sqlite_database()

    @pytest.mark.asyncio
    async def test_custom_engine_with_schema_creates_schema(self):
        """Test that db_schema is respected when a custom non-SQLite engine is provided.

        Regression test for https://github.com/run-llama/llama_index/issues/20746
        """
        mock_pg_engine = MagicMock()
        mock_pg_engine.url = "postgresql+asyncpg://user:pass@host/db"
        mock_pg_engine.begin.return_value.__aenter__ = AsyncMock()
        mock_pg_engine.begin.return_value.__aexit__ = AsyncMock()

        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_pg_engine.begin.return_value.__aenter__.return_value = mock_conn

        store = SQLAlchemyChatStore(
            table_name="test_messages",
            async_engine=mock_pg_engine,
            db_schema="my_schema",
        )

        await store._setup_tables(mock_pg_engine)

        # Verify schema creation SQL was called
        mock_conn.execute.assert_called()
        call_args = mock_conn.execute.call_args_list[0][0][0]
        assert 'CREATE SCHEMA IF NOT EXISTS "my_schema"' in str(call_args)
        assert store._metadata.schema == "my_schema"

    @pytest.mark.asyncio
    async def test_basic_operations_with_schema(self):
        """Test that basic operations work with schema."""
        store = SQLAlchemyChatStore(
            table_name="test_messages",
            async_database_uri="sqlite+aiosqlite:///:memory:",
            db_schema="test_schema",
        )

        # Add and retrieve message
        await store.add_message(
            "schema_user", ChatMessage(role="user", content="Hello with schema!")
        )

        messages = await store.get_messages("schema_user")
        assert len(messages) == 1
        assert messages[0].content == "Hello with schema!"

        # Verify schema is preserved
        assert store.db_schema == "test_schema"
