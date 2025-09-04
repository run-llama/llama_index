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
