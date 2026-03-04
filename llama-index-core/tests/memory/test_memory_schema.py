"""Tests for Memory class schema functionality."""

import pytest

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory.memory import Memory
from llama_index.core.storage.chat_store.sql import SQLAlchemyChatStore


class TestMemorySchema:
    """Test schema functionality in Memory class."""

    def test_from_defaults_schema_parameter(self):
        """Test Memory.from_defaults with and without schema parameter."""
        # Without schema
        memory_no_schema = Memory.from_defaults(
            token_limit=1000,
            table_name="test_memory",
        )
        assert memory_no_schema.sql_store.db_schema is None
        assert memory_no_schema.sql_store.table_name == "test_memory"

        # With schema
        memory_with_schema = Memory.from_defaults(
            token_limit=1000,
            table_name="test_memory",
            db_schema="test_schema",
        )
        assert memory_with_schema.sql_store.db_schema == "test_schema"
        assert memory_with_schema.sql_store.table_name == "test_memory"

    def test_schema_parameter_passing(self):
        """Test that schema parameter is correctly passed to SQLAlchemyChatStore."""
        memory = Memory.from_defaults(
            table_name="param_test",
            async_database_uri="postgresql+asyncpg://user:pass@host/db",
            db_schema="param_schema",
        )

        # Verify the SQL store is correctly configured
        assert isinstance(memory.sql_store, SQLAlchemyChatStore)
        assert memory.sql_store.db_schema == "param_schema"
        assert memory.sql_store.table_name == "param_test"
        assert (
            memory.sql_store.async_database_uri
            == "postgresql+asyncpg://user:pass@host/db"
        )

    @pytest.mark.asyncio
    async def test_memory_operations_with_schema(self):
        """Test that Memory operations work with schema."""
        memory = Memory.from_defaults(
            token_limit=1000,
            table_name="integration_test",
            db_schema="integration_schema",
        )

        # Add a message
        message = ChatMessage(role="user", content="Hello from Memory with schema!")
        await memory.aput(message)

        # Retrieve messages
        messages = await memory.aget()
        assert len(messages) >= 1

        # Find our message (there might be system messages)
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "Hello from Memory with schema!"

        # Verify schema is preserved
        assert memory.sql_store.db_schema == "integration_schema"

    @pytest.mark.asyncio
    async def test_memory_reset_preserves_schema(self):
        """Test that memory reset preserves schema configuration."""
        memory = Memory.from_defaults(
            token_limit=1000,
            table_name="reset_test",
            db_schema="reset_schema",
        )

        # Add a message
        await memory.aput(ChatMessage(role="user", content="Before reset"))

        # Reset memory
        await memory.areset()

        # Verify schema is still set
        assert memory.sql_store.db_schema == "reset_schema"

        # Verify messages are cleared
        messages = await memory.aget_all()
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) == 0
