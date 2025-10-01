import pytest
from unittest.mock import MagicMock, patch
import json

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.memory import InsertMethod
from llama_index.memory.bedrock_agentcore.base import (
    AgentCoreMemory,
    AgentCoreMemoryContext,
)


@pytest.fixture()
def mock_client():
    """Create a mock Bedrock AgentCore client."""
    client = MagicMock()
    client.create_event.return_value = {"event": {"eventId": "test-event-id"}}
    client.list_events.return_value = {"events": [], "nextToken": None}
    client.retrieve_memory_records.return_value = {"memoryRecordSummaries": []}
    return client


@pytest.fixture()
def memory_context():
    """Create a basic AgentCore Memory context for testing."""
    return AgentCoreMemoryContext(
        actor_id="test-actor",
        memory_id="test-memory-store",
        session_id="test-session-id",
        memory_strategy_id="test-semantic-memory-strategy",
    )


@pytest.fixture()
def memory(mock_client, memory_context):
    """Create a basic AgentCore Memory instance for testing."""
    return AgentCoreMemory(context=memory_context, client=mock_client)


class TestAgentCoreMemoryContext:
    """Test AgentCoreMemoryContext class."""

    def test_context_creation(self):
        """Test creating a memory context."""
        context = AgentCoreMemoryContext(
            actor_id="test-actor", memory_id="test-memory", session_id="test-session"
        )
        assert context.actor_id == "test-actor"
        assert context.memory_id == "test-memory"
        assert context.session_id == "test-session"
        assert context.namespace == "/"
        assert context.memory_strategy_id is None

    def test_context_with_optional_fields(self):
        """Test creating a memory context with optional fields."""
        context = AgentCoreMemoryContext(
            actor_id="test-actor",
            memory_id="test-memory",
            session_id="test-session",
            namespace="/custom",
            memory_strategy_id="custom-strategy",
        )
        assert context.namespace == "/custom"
        assert context.memory_strategy_id == "custom-strategy"

    def test_get_context(self):
        """Test getting context as dictionary."""
        context = AgentCoreMemoryContext(
            actor_id="test-actor",
            memory_id="test-memory",
            session_id="test-session",
            memory_strategy_id="test-strategy",
        )
        context_dict = context.get_context()
        expected = {
            "actor_id": "test-actor",
            "memory_id": "test-memory",
            "session_id": "test-session",
            "namespace": "/",
            "memory_strategy_id": "test-strategy",
        }
        assert context_dict == expected


class TestBaseAgentCoreMemoryMethods:
    """Test BaseAgentCoreMemory methods using AgentCoreMemory instance."""

    def test_create_event_success(self, memory):
        """Test successful event creation."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

        memory.create_event(
            memory_id="test-memory",
            actor_id="test-actor",
            messages=messages,
            session_id="test-session",
        )

        assert memory._client.create_event.called
        call_args = memory._client.create_event.call_args
        assert call_args[1]["memoryId"] == "test-memory"
        assert call_args[1]["actorId"] == "test-actor"
        assert call_args[1]["sessionId"] == "test-session"

    def test_create_event_no_client(self, memory_context):
        """Test create_event raises error when client is None."""
        with patch("boto3.Session"):
            memory = AgentCoreMemory(context=memory_context)
            memory._client = None  # Set client to None after initialization
            messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

            with pytest.raises(ValueError, match="Client is not initialized"):
                memory.create_event(
                    memory_id="test-memory",
                    actor_id="test-actor",
                    messages=messages,
                    session_id="test-session",
                )

    def test_create_event_empty_messages(self, memory):
        """Test create_event raises error when messages is empty."""
        with pytest.raises(ValueError, match="The messages field cannot be empty"):
            memory.create_event(
                memory_id="test-memory",
                actor_id="test-actor",
                messages=[],
                session_id="test-session",
            )

    def test_create_event_no_event_id(self, memory):
        """Test create_event raises error when no event ID is returned."""
        memory._client.create_event.return_value = {"event": {"eventId": None}}
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

        with pytest.raises(
            RuntimeError, match="Bedrock AgentCore did not return an event ID"
        ):
            memory.create_event(
                memory_id="test-memory",
                actor_id="test-actor",
                messages=messages,
                session_id="test-session",
            )

    def test_list_events_simple(self, memory):
        """Test listing events with simple user message first."""
        # Mock response with a user message first
        mock_events = [
            {
                "payload": [
                    {"blob": json.dumps({})},
                    {"conversational": {"role": "USER", "content": {"text": "Hello"}}},
                ]
            }
        ]
        memory._client.list_events.return_value = {
            "events": mock_events,
            "nextToken": None,
        }

        messages = memory.list_events(
            memory_id="test-memory", session_id="test-session", actor_id="test-actor"
        )

        assert len(messages) == 1
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hello"

    def test_list_events_with_pagination(self, memory):
        """Test listing events with pagination to find user message."""
        # First call returns assistant message
        mock_events_1 = [
            {
                "payload": [
                    {"blob": json.dumps({})},
                    {
                        "conversational": {
                            "role": "ASSISTANT",
                            "content": {"text": "Hi there"},
                        }
                    },
                ]
            }
        ]

        # Second call returns user message
        mock_events_2 = [
            {
                "payload": [
                    {"blob": json.dumps({})},
                    {"conversational": {"role": "USER", "content": {"text": "Hello"}}},
                ]
            }
        ]

        memory._client.list_events.side_effect = [
            {"events": mock_events_1, "nextToken": "token1"},
            {"events": mock_events_2, "nextToken": None},
        ]

        messages = memory.list_events(
            memory_id="test-memory", session_id="test-session", actor_id="test-actor"
        )

        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hello"
        assert messages[1].role == MessageRole.ASSISTANT
        assert messages[1].content == "Hi there"

    def test_retrieve_memories(self, memory):
        """Test retrieving memory records."""
        memory._client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [{"content": "Memory 1"}, {"content": "Memory 2"}]
        }

        memories = memory.retrieve_memories(
            memory_id="test-memory", search_criteria={"searchQuery": "test query"}
        )

        assert memories == ["Memory 1", "Memory 2"]
        memory._client.retrieve_memory_records.assert_called_once_with(
            memoryId="test-memory",
            namespace="/",
            searchCriteria={"searchQuery": "test query"},
            maxResults=20,
        )


class TestAgentCoreMemory:
    """Test AgentCoreMemory class."""

    def test_initialization(self, memory_context):
        """Test AgentCoreMemory initialization."""
        with patch("boto3.Session") as mock_session:
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            memory = AgentCoreMemory(context=memory_context)

            assert memory._context == memory_context
            assert memory._client == mock_client
            assert memory.search_msg_limit == 5
            assert memory.insert_method == InsertMethod.SYSTEM

    def test_initialization_with_custom_client(self, memory_context, mock_client):
        """Test initialization with custom client."""
        memory = AgentCoreMemory(context=memory_context, client=mock_client)
        assert memory._client == mock_client

    def test_class_name(self):
        """Test class name method."""
        assert AgentCoreMemory.class_name() == "AgentCoreMemory"

    def test_from_defaults_not_implemented(self):
        """Test that from_defaults raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Use either from_client or from_config"
        ):
            AgentCoreMemory.from_defaults()

    def test_serialize_memory(self, memory):
        """Test memory serialization."""
        serialized = memory.serialize_memory()

        assert "search_msg_limit" in serialized
        assert serialized["search_msg_limit"] == 5

        # primary_memory is no longer included in serialization
        assert "primary_memory" not in serialized

    def test_get_context(self, memory, memory_context):
        """Test getting context."""
        context = memory.get_context()
        assert context == memory_context.get_context()

    def test_get_with_system_insert(self, memory):
        """Test get method with SYSTEM insert method."""
        # Mock the underlying methods that get() calls
        mock_events = [ChatMessage(role=MessageRole.USER, content="Hello")]
        mock_memories = ["Memory 1", "Memory 2"]

        # Mock the client methods that are actually called
        memory._client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {"blob": json.dumps({})},
                        {
                            "conversational": {
                                "role": "USER",
                                "content": {"text": "Hello"},
                            }
                        },
                    ]
                }
            ],
            "nextToken": None,
        }

        memory._client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [
                {"content": {"text": "Memory 1"}},
                {"content": {"text": "Memory 2"}},
            ]
        }

        # Test the get method
        result = memory.get(input="test input")

        # Should have system message + user message
        assert len(result) >= 1
        assert memory._client.list_events.called
        assert memory._client.retrieve_memory_records.called

    def test_get_with_user_insert(self, memory):
        """Test get method with USER insert method."""
        # Setup
        memory.insert_method = InsertMethod.USER

        # Mock the client methods
        memory._client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {"blob": json.dumps({})},
                        {
                            "conversational": {
                                "role": "USER",
                                "content": {"text": "Hello"},
                            }
                        },
                    ]
                }
            ],
            "nextToken": None,
        }

        memory._client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [{"content": {"text": "Memory 1"}}]
        }

        # Test
        result = memory.get()

        # Should have at least one message
        assert len(result) >= 1
        assert memory._client.list_events.called
        assert memory._client.retrieve_memory_records.called

    def test_get_all(self, memory):
        """Test get_all method."""
        mock_messages = [ChatMessage(role=MessageRole.USER, content="Test")]

        # Mock the client's list_events method since get_all calls self.list_events which uses the client
        memory._client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {"blob": json.dumps({})},
                        {
                            "conversational": {
                                "role": "USER",
                                "content": {"text": "Test"},
                            }
                        },
                    ]
                }
            ],
            "nextToken": None,
        }

        result = memory.get_all()

        assert len(result) == 1
        assert result[0].role == MessageRole.USER
        assert result[0].content == "Test"
        memory._client.list_events.assert_called()

    def test_put(self, memory):
        """Test put method."""
        message = ChatMessage(role=MessageRole.USER, content="Hello")

        # Mock the _add_msgs_to_client_memory method
        with patch.object(memory, "_add_msgs_to_client_memory") as mock_add_msgs:
            memory.put(message)

            mock_add_msgs.assert_called_once_with([message])

    @pytest.mark.asyncio
    async def test_aput(self, memory):
        """Test async put method."""
        message = ChatMessage(role=MessageRole.USER, content="Hello")

        with patch.object(memory, "_add_msgs_to_client_memory") as mock_add_msgs:
            await memory.aput(message)
            mock_add_msgs.assert_called_once_with([message])

    @pytest.mark.asyncio
    async def test_aput_messages(self, memory):
        """Test async put messages method."""
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ]

        with patch.object(memory, "_add_msgs_to_client_memory") as mock_add_msgs:
            await memory.aput_messages(messages)
            mock_add_msgs.assert_called_once_with(messages)

    def test_set(self, memory):
        """Test set method."""
        existing_messages = [ChatMessage(role=MessageRole.USER, content="Old")]
        new_messages = [
            ChatMessage(role=MessageRole.USER, content="Old"),
            ChatMessage(role=MessageRole.ASSISTANT, content="New"),
        ]

        # Mock the client's list_events method since set() calls get_all() which calls list_events()
        memory._client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {"blob": json.dumps({})},
                        {
                            "conversational": {
                                "role": "USER",
                                "content": {"text": "Old"},
                            }
                        },
                    ]
                }
            ],
            "nextToken": None,
        }

        # Mock the _add_msgs_to_client_memory method
        with patch.object(memory, "_add_msgs_to_client_memory") as mock_add_msgs:
            memory.set(new_messages)

            # Should only add the new message (since existing has 1 message, new has 2)
            mock_add_msgs.assert_called_once_with([new_messages[1]])

    def test_reset(self, memory):
        """Test reset method."""
        # The reset method now just passes (no-op) as per the implementation
        # This test verifies that reset can be called without errors
        memory.reset()
        # No assertions needed since reset() is now a no-op

    def test_add_msgs_to_client_memory(self, memory):
        """Test adding messages to client memory."""
        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

        # Test the actual implementation - it should call create_event
        memory._add_msgs_to_client_memory(messages)

        # Verify create_event was called with correct parameters
        memory._client.create_event.assert_called()


class TestIntegration:
    """Integration tests for AgentCoreMemory."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, memory_context, mock_client):
        """Test a complete workflow with AgentCoreMemory."""
        # Setup mock responses
        mock_client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {"blob": json.dumps({})},
                        {
                            "conversational": {
                                "role": "USER",
                                "content": {"text": "Hello"},
                            }
                        },
                    ]
                }
            ],
            "nextToken": None,
        }
        mock_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [{"content": {"text": "User likes greetings"}}]
        }

        # Create memory instance
        memory = AgentCoreMemory(context=memory_context, client=mock_client)

        # Add a message
        message = ChatMessage(role=MessageRole.USER, content="New message")
        await memory.aput(message)

        # Verify create_event was called
        assert mock_client.create_event.called

        # Get messages (this will call list_events and retrieve_memories)
        messages = memory.get()

        # Should have system message + user message
        assert len(messages) >= 1
        assert mock_client.list_events.called
        assert mock_client.retrieve_memory_records.called


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_boto3_import_error(self, memory_context):
        """Test handling of boto3 import error."""
        with patch("boto3.Session", side_effect=ImportError("boto3 not found")):
            with pytest.raises(ImportError, match="boto3  package not found"):
                AgentCoreMemory(context=memory_context)

    def test_client_initialization_error(self, memory_context):
        """Test handling of client initialization errors."""
        with patch("boto3.Session") as mock_session:
            mock_session.side_effect = Exception("AWS credentials not found")

            with pytest.raises(Exception, match="AWS credentials not found"):
                AgentCoreMemory(context=memory_context)


# Integration test with existing tests
@pytest.mark.asyncio
async def test_aput(memory):
    """Test adding a message."""
    message = ChatMessage(role="user", content="New message")

    await memory.aput(message)

    # Verify that create_event was called
    assert memory._client.create_event.called


@pytest.mark.asyncio
async def test_aput_messages(memory):
    """Test adding multiple messages."""
    messages = [
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Response 1"),
    ]

    await memory.aput_messages(messages)

    # Verify that create_event was called for each message
    assert memory._client.create_event.call_count == 1
