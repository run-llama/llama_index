from unittest.mock import patch, MagicMock

from llama_index.memory.mem0.base import Mem0Memory, Mem0Context
from llama_index.core.memory.chat_memory_buffer import ChatMessage, MessageRole
from llama_index.memory.mem0.utils import (
    convert_chat_history_to_dict,
    convert_messages_to_string,
)


def test_mem0_memory_from_client():
    # Mock context
    context = {"user_id": "test_user"}

    # Mock arguments for MemoryClient
    api_key = "test_api_key"
    host = "test_host"
    org_id = "test_org"
    project_id = "test_project"
    search_msg_limit = 10  # Add this line

    # Patch MemoryClient
    with patch("llama_index.memory.mem0.base.MemoryClient") as MockMemoryClient:
        mock_client = MagicMock()
        MockMemoryClient.return_value = mock_client

        # Call from_client method
        mem0_memory = Mem0Memory.from_client(
            context=context,
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
            search_msg_limit=search_msg_limit,  # Add this line
        )

        # Assert that MemoryClient was called with the correct arguments
        MockMemoryClient.assert_called_once_with(
            api_key=api_key, host=host, org_id=org_id, project_id=project_id
        )

        # Assert that the returned object is an instance of Mem0Memory
        assert isinstance(mem0_memory, Mem0Memory)

        # Assert that the context was set correctly
        assert isinstance(mem0_memory.context, Mem0Context)
        assert mem0_memory.context.user_id == "test_user"

        # Assert that the client was set correctly
        assert mem0_memory._client == mock_client

        # Assert that the search_msg_limit was set correctly
        assert mem0_memory.search_msg_limit == search_msg_limit  # Add this line


def test_mem0_memory_from_config():
    # Mock context
    context = {"user_id": "test_user"}

    # Mock config
    config = {"test": "test"}

    # Set search_msg_limit
    search_msg_limit = 15  # Add this line

    # Patch Memory
    with patch("llama_index.memory.mem0.base.Memory") as MockMemory:
        mock_client = MagicMock()
        MockMemory.from_config.return_value = mock_client

        # Call from_config method
        mem0_memory = Mem0Memory.from_config(
            context=context,
            config=config,
            search_msg_limit=search_msg_limit,  # Add this line
        )

        # Assert that the client was set correctly
        assert mem0_memory._client == mock_client

        # Assert that the search_msg_limit was set correctly
        assert mem0_memory.search_msg_limit == search_msg_limit  # Add this line


def test_mem0_memory_set():
    # Mock context
    context = {"user_id": "test_user"}

    # Mock arguments for MemoryClient
    api_key = "test_api_key"
    host = "test_host"
    org_id = "test_org"
    project_id = "test_project"

    # Patch MemoryClient
    with patch("llama_index.memory.mem0.base.MemoryClient") as MockMemoryClient:
        mock_client = MagicMock()
        MockMemoryClient.return_value = mock_client

        # Create Mem0Memory instance
        mem0_memory = Mem0Memory.from_client(
            context=context,
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
        )

        # Create a list of alternating user and assistant messages
        messages = [
            ChatMessage(role=MessageRole.USER, content="User message 1"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Assistant message 1"),
            ChatMessage(role=MessageRole.USER, content="User message 2"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Assistant message 2"),
        ]

        # Call the set method
        mem0_memory.set(messages)

        # Assert that add was called only for user messages
        assert mock_client.add.call_count == 1
        expected_messages = convert_chat_history_to_dict(messages)
        mock_client.add.assert_called_once_with(
            messages=expected_messages, user_id="test_user"
        )

        # Assert that the primary_memory was set with all messages
        assert mem0_memory.primary_memory.get_all() == messages

        # Test setting messages when chat history is not empty
        new_messages = [
            ChatMessage(role=MessageRole.USER, content="User message 3"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Assistant message 3"),
        ]

        # Reset the mock to clear previous calls
        mock_client.add.reset_mock()

        # Call the set method again
        mem0_memory.set(messages + new_messages)

        # Assert that add was called only for the new messages
        expected_new_messages = convert_chat_history_to_dict(new_messages)
        mock_client.add.assert_called_once_with(
            messages=expected_new_messages, user_id="test_user"
        )

        # Assert that the primary_memory was updated with all messages
        assert mem0_memory.primary_memory.get_all() == messages + new_messages


def test_mem0_memory_get():
    # Mock context
    context = {"user_id": "test_user"}

    # Mock arguments for MemoryClient
    api_key = "test_api_key"
    host = "test_host"
    org_id = "test_org"
    project_id = "test_project"

    # Patch MemoryClient
    with patch("llama_index.memory.mem0.base.MemoryClient") as MockMemoryClient:
        mock_client = MagicMock()
        MockMemoryClient.return_value = mock_client

        # Create Mem0Memory instance
        mem0_memory = Mem0Memory.from_client(
            context=context,
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
        )

        # Set dummy chat history
        dummy_messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ChatMessage(role=MessageRole.USER, content="How are you?"),
            ChatMessage(
                role=MessageRole.ASSISTANT, content="I'm doing well, thank you!"
            ),
        ]
        mem0_memory.primary_memory.set(dummy_messages)

        # Set dummy response for search
        dummy_search_results = [
            {
                "categories": ["greeting"],
                "memory": "The user usually starts with a greeting.",
            },
            {"categories": ["mood"], "memory": "The user often asks about well-being."},
        ]
        mock_client.search.return_value = dummy_search_results

        # Call get method
        result = mem0_memory.get(input="How are you?")

        # Assert that search was called with correct arguments
        expected_query = convert_messages_to_string(
            dummy_messages, "How are you?", limit=mem0_memory.search_msg_limit
        )
        mock_client.search.assert_called_once_with(
            query=expected_query, user_id="test_user"
        )

        # Assert that the result contains the correct number of messages
        assert len(result) == len(dummy_messages) + 1  # +1 for the system message

        # Assert that the first message is a system message
        assert result[0].role == MessageRole.SYSTEM

        # Assert that the system message contains the search results
        assert "The user usually starts with a greeting." in result[0].content
        assert "The user often asks about well-being." in result[0].content

        # Assert that the rest of the messages match the dummy messages
        assert result[1:] == dummy_messages

        # Test get method without input (should use last user message)
        mock_client.search.reset_mock()
        result_no_input = mem0_memory.get()

        # Assert that search was called with the last user message
        expected_query_no_input = convert_messages_to_string(
            dummy_messages, limit=mem0_memory.search_msg_limit
        )
        mock_client.search.assert_called_once_with(
            query=expected_query_no_input, user_id="test_user"
        )

        # Assert that the results are the same as before
        assert result_no_input == result


def test_mem0_memory_put():
    # Mock context
    context = {"user_id": "test_user"}

    # Mock arguments for MemoryClient
    api_key = "test_api_key"
    host = "test_host"
    org_id = "test_org"
    project_id = "test_project"

    # Patch MemoryClient
    with patch("llama_index.memory.mem0.base.MemoryClient") as MockMemoryClient:
        mock_client = MagicMock()
        MockMemoryClient.return_value = mock_client

        # Create Mem0Memory instance
        mem0_memory = Mem0Memory.from_client(
            context=context,
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
        )

        # Create a test message
        test_message = ChatMessage(role=MessageRole.USER, content="Hello, world!")

        # Call the put method
        mem0_memory.put(test_message)

        # Assert that the message was added to primary_memory
        assert mem0_memory.primary_memory.get_all() == [test_message]

        # Assert that add was called with the correct arguments
        expected_messages = convert_chat_history_to_dict([test_message])
        mock_client.add.assert_called_once_with(
            messages=expected_messages, user_id="test_user"
        )
