import pytest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from azure.ai.agents.models import (
    MessageInputTextBlock,
    MessageInputImageFileBlock,
    MessageInputImageUrlBlock,
)
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import Agent as AzureAgent, AgentThread

from llama_index.agent.azure_foundry_agent.base import AzureFoundryAgent
from llama_index.core.base.llms.types import ChatMessage, TextBlock, ImageBlock
from llama_index.core.agent.workflow.multi_agent_workflow import AgentWorkflow
from llama_index.core.memory import ChatMemoryBuffer


# Helper for async iteration (ensure only one definition)
class DummyAsyncIterator:
    def __init__(self, items):
        self._items = items

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def test_azure_foundry_agent_constructor():
    """Test the constructor of AzureFoundryAgent."""
    endpoint = "https://test-endpoint.com"
    model = "gpt-4o"
    name = "test-azure-agent"
    instructions = "You are a test agent."
    thread_id = "test-thread-123"
    verbose = True
    run_retrieve_sleep_time = 0.5

    mock_project_client_instance = MagicMock(spec=AIProjectClient)
    mock_azure_agent_instance = MagicMock(spec=AzureAgent)
    mock_azure_agent_instance.id = "mock_agent_id_123"
    mock_thread_instance = MagicMock(spec=AgentThread)
    mock_thread_instance.id = thread_id

    # Patch async methods with AsyncMock
    mock_project_client_instance.agents.create_agent = AsyncMock(
        return_value=mock_azure_agent_instance
    )
    mock_project_client_instance.agents.threads.create = AsyncMock(
        return_value=mock_thread_instance
    )

    # Mock DefaultAzureCredential to avoid actual credential loading
    with patch(
        "llama_index.agent.azure_foundry_agent.base.DefaultAzureCredential", MagicMock()
    ) as mock_default_credential:
        # Mock AIProjectClient constructor to return our mock instance
        with patch(
            "llama_index.agent.azure_foundry_agent.base.AIProjectClient",
            return_value=mock_project_client_instance,
        ) as mock_ai_project_client_constructor:
            # Mock the create_agent call
            mock_project_client_instance.agents.create_agent.return_value = (
                mock_azure_agent_instance
            )
            # Mock the threads.create call for when thread_id is None
            mock_project_client_instance.agents.threads.create.return_value = (
                mock_thread_instance
            )

            # Test case 1: Initialize with a specific thread_id
            agent_with_thread = AzureFoundryAgent(
                endpoint=endpoint,
                model=model,
                name=name,
                instructions=instructions,
                thread_id=thread_id,
                verbose=verbose,
                run_retrieve_sleep_time=run_retrieve_sleep_time,
            )

            mock_ai_project_client_constructor.assert_called_once_with(
                endpoint=endpoint, credential=mock_default_credential.return_value
            )
            # Ensure threads.create was NOT called because thread_id was provided
            mock_project_client_instance.agents.threads.create.assert_not_called()
            assert isinstance(agent_with_thread, AzureFoundryAgent)
            assert agent_with_thread._endpoint == endpoint
            assert agent_with_thread._model == model
            assert agent_with_thread.name == name
            assert agent_with_thread._instructions == instructions
            assert agent_with_thread._thread_id == thread_id
            assert agent_with_thread._verbose == verbose
            assert agent_with_thread._run_retrieve_sleep_time == run_retrieve_sleep_time
            assert agent_with_thread._client == mock_project_client_instance

            # Reset mocks for the next instantiation test
            mock_ai_project_client_constructor.reset_mock()
            mock_project_client_instance.reset_mock()
            mock_default_credential.reset_mock()

            # Mock the threads.create call for when thread_id is None
            # Re-assign thread_id for the new mock thread instance if it's different
            new_mock_thread_id = "new-mock-thread-456"
            mock_thread_instance_new = MagicMock(spec=AgentThread)
            mock_thread_instance_new.id = new_mock_thread_id
            mock_project_client_instance.agents.threads.create = AsyncMock(
                return_value=mock_thread_instance_new
            )

            # Test case 2: Initialize without a specific thread_id (should create one)
            agent_new_thread = AzureFoundryAgent(
                endpoint=endpoint,
                model=model,
                name=name,
                instructions=instructions,
                thread_id=None,  # Test thread creation
                verbose=verbose,
                run_retrieve_sleep_time=run_retrieve_sleep_time,
            )
            assert agent_new_thread.name == name
            assert agent_new_thread._client == mock_project_client_instance
            # At this point, thread should not be created yet
            mock_project_client_instance.agents.threads.create.assert_not_called()
            # Now, trigger thread creation by calling _ensure_agent
            import asyncio

            asyncio.run(agent_new_thread._ensure_agent([]))
            mock_project_client_instance.agents.threads.create.assert_called_once()
            assert agent_new_thread._thread_id == new_mock_thread_id


@patch("azure.identity.aio.DefaultAzureCredential")
@patch("azure.ai.projects.aio.AIProjectClient")
@pytest.mark.asyncio  # Added decorator
async def test_azure_foundry_agent_constructor_defaults(  # Added async and mock arguments
    mock_project_client_class: MagicMock, mock_credential_class: MagicMock
):
    """Test the constructor of AzureFoundryAgent with default values."""
    endpoint = "https://test-endpoint.com"
    model = "gpt-4o"
    name = "test-azure-agent-defaults"
    instructions = "You are a test agent. (defaults)"
    thread_id = None
    verbose = False
    run_retrieve_sleep_time = 1.0

    mock_project_client_instance = MagicMock(spec=AIProjectClient)
    mock_azure_agent_instance = MagicMock(spec=AzureAgent)
    mock_azure_agent_instance.id = "mock_agent_id_defaults"
    mock_thread_instance = MagicMock(spec=AgentThread)
    mock_thread_instance.id = "mock_thread_id_defaults"

    # Patch async methods with AsyncMock
    mock_project_client_instance.agents.create_agent = AsyncMock(
        return_value=mock_azure_agent_instance
    )
    mock_project_client_instance.agents.threads.create = AsyncMock(
        return_value=mock_thread_instance
    )

    with patch(
        "llama_index.agent.azure_foundry_agent.base.DefaultAzureCredential", MagicMock()
    ):
        with patch(
            "llama_index.agent.azure_foundry_agent.base.AIProjectClient",
            return_value=mock_project_client_instance,
        ):
            # Test initialization with defaults
            agent_defaults = AzureFoundryAgent(
                endpoint=endpoint,
                model=model,
                name=name,
                instructions=instructions,
                thread_id=thread_id,
                verbose=verbose,
                run_retrieve_sleep_time=run_retrieve_sleep_time,
            )

            assert agent_defaults.name == name
            assert agent_defaults._endpoint == endpoint
            assert agent_defaults._model == model
            assert agent_defaults._instructions == instructions
            assert agent_defaults._thread_id is None
            assert agent_defaults._verbose is False
            assert agent_defaults._run_retrieve_sleep_time == run_retrieve_sleep_time
            assert agent_defaults._client == mock_project_client_instance

            # Ensure that create_agent and threads.create are called only after _ensure_agent
            await agent_defaults._ensure_agent([])
            print(
                f"create_agent call count: {mock_project_client_instance.agents.create_agent.call_count}"
            )
            print(
                f"threads.create call count: {mock_project_client_instance.agents.threads.create.call_count}"
            )
            mock_project_client_instance.agents.create_agent.assert_called_once()
            mock_project_client_instance.agents.threads.create.assert_called_once()

            # Check that the thread_id was set to the created thread's ID
            assert agent_defaults._thread_id == mock_thread_instance.id


# Tests for _llama_to_azure_content_blocks
@pytest.mark.parametrize(
    ("desc", "chat_messages", "expected_types", "expected_values"),
    [
        (
            "empty input",
            [],
            [],
            [],
        ),
        (
            "text only",
            [ChatMessage(role="user", blocks=[TextBlock(text="Hello")])],
            [MessageInputTextBlock],
            ["Hello"],
        ),
        (
            "image url",
            [
                ChatMessage(
                    role="user",
                    blocks=[
                        ImageBlock(url="http://example.com/image.png", detail="low")
                    ],
                )
            ],
            [MessageInputImageUrlBlock],
            ["http://example.com/image.png"],
        ),
        (
            "no blocks, just content",
            [ChatMessage(role="user", content="Just text content, no blocks attr")],
            [MessageInputTextBlock],
            ["Just text content, no blocks attr"],
        ),
        (
            "empty blocks",
            [ChatMessage(role="user", blocks=[])],
            [],
            [],
        ),
        (
            "image block no path no url",
            [
                ChatMessage(
                    role="user",
                    blocks=[ImageBlock(image=b"some_image_data", detail="high")],
                )
            ],
            [],
            [],
        ),
    ],
)
def test_llama_to_azure_content_blocks_param(
    desc, chat_messages, expected_types, expected_values
):
    agent = AzureFoundryAgent(endpoint="dummy_endpoint")
    result = agent._llama_to_azure_content_blocks(chat_messages)
    assert len(result) == len(expected_types)
    for r, t, v in zip(result, expected_types, expected_values):
        assert isinstance(r, t)
        # Check value for text or url
        if isinstance(r, MessageInputTextBlock):
            assert r.text == v
        elif isinstance(r, MessageInputImageUrlBlock):
            assert r.image_url.url == v


def test_llama_to_azure_content_blocks_image_path_and_mixed():
    agent = AzureFoundryAgent(endpoint="dummy_endpoint")
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        # image path
        chat_messages = [
            ChatMessage(
                role="user", blocks=[ImageBlock(path=Path(tmp.name), detail="high")]
            )
        ]
        result = agent._llama_to_azure_content_blocks(chat_messages)
        assert len(result) == 1
        assert isinstance(result[0], MessageInputImageFileBlock)
        assert result[0].image_file.file_id == tmp.name
        assert result[0].image_file.detail == "high"

        # mixed content
        chat_messages = [
            ChatMessage(
                role="user",
                blocks=[
                    TextBlock(text="Describe this image:"),
                    ImageBlock(path=Path(tmp.name)),
                ],
            )
        ]
        result = agent._llama_to_azure_content_blocks(chat_messages)
        assert len(result) == 2
        assert isinstance(result[0], MessageInputTextBlock)
        assert result[0].text == "Describe this image:"
        assert isinstance(result[1], MessageInputImageFileBlock)
        assert result[1].image_file.file_id == tmp.name

    # image block path preferred over image attr
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        chat_messages = [
            ChatMessage(
                role="user",
                blocks=[ImageBlock(path=Path(tmp.name), image=b"image_bytes")],
            )
        ]
        result = agent._llama_to_azure_content_blocks(chat_messages)
        assert len(result) == 1
        assert isinstance(result[0], MessageInputImageFileBlock)
        assert result[0].image_file.file_id == tmp.name

    # image bytes only, should be skipped
    chat_messages_bytes_only = [
        ChatMessage(
            role="user", blocks=[ImageBlock(image=b"image_bytes_data", detail="auto")]
        )
    ]
    result_bytes_only = agent._llama_to_azure_content_blocks(chat_messages_bytes_only)
    assert len(result_bytes_only) == 0


def test_llama_to_azure_content_blocks_multiple_messages():
    agent = AzureFoundryAgent(endpoint="dummy_endpoint")
    with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
        chat_messages = [
            ChatMessage(role="user", blocks=[TextBlock(text="First message.")]),
            ChatMessage(
                role="user", blocks=[ImageBlock(url="http://images.com/pic.png")]
            ),
            ChatMessage(
                role="user",
                blocks=[
                    TextBlock(text="Third message text."),
                    ImageBlock(path=Path(tmp.name)),
                ],
            ),
        ]
        result = agent._llama_to_azure_content_blocks(chat_messages)
        assert len(result) == 4
        assert isinstance(result[0], MessageInputTextBlock)
        assert result[0].text == "First message."
        assert isinstance(result[1], MessageInputImageUrlBlock)
        assert result[1].image_url.url == "http://images.com/pic.png"
        assert isinstance(result[2], MessageInputTextBlock)
        assert result[2].text == "Third message text."
        assert isinstance(result[3], MessageInputImageFileBlock)
        assert result[3].image_file.file_id == tmp.name


# --- Workflow and tool call tests from the other file ---
@pytest.mark.asyncio
async def test_azure_foundry_agent_workflow():
    with (
        patch(
            "llama_index.agent.azure_foundry_agent.base.DefaultAzureCredential",
            MagicMock(),
        ),
        patch(
            "llama_index.agent.azure_foundry_agent.base.AIProjectClient"
        ) as mock_client_class,
    ):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.agents.create_agent = AsyncMock()
        mock_client.agents.threads.create = AsyncMock()
        mock_client.agents.get_agent = AsyncMock()
        mock_client.agents.messages.create = AsyncMock()
        mock_client.agents.runs.create = AsyncMock()
        mock_client.agents.runs.get = AsyncMock()
        mock_client.agents.messages.list.return_value = DummyAsyncIterator([])
        mock_client.agents.runs.submit_tool_outputs = AsyncMock()
        mock_client.close = AsyncMock()
        agent = AzureFoundryAgent(
            endpoint="https://fake-endpoint",
            model="gpt-4o",
            name="azure-agent",
            instructions="Test agent",
            verbose=True,
        )
        workflow = AgentWorkflow(
            agents=[agent],
        )
        memory = ChatMemoryBuffer.from_defaults()
        handler = workflow.run(user_msg="Hello, agent!", memory=memory)
        events = []
        async for event in handler.stream_events():
            events.append(event)
        response = await handler
        assert response is not None


@pytest.mark.asyncio
async def test_azure_foundry_agent_tool_call():
    with (
        patch(
            "llama_index.agent.azure_foundry_agent.base.DefaultAzureCredential",
            MagicMock(),
        ),
        patch(
            "llama_index.agent.azure_foundry_agent.base.AIProjectClient"
        ) as mock_client_class,
    ):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.agents.create_agent = AsyncMock()
        mock_client.agents.threads.create = AsyncMock()
        mock_client.agents.get_agent = AsyncMock()
        mock_client.agents.messages.create = AsyncMock()
        mock_client.close = AsyncMock()

        class DummyRun:
            def __init__(self, status, required_action=None):
                self.status = status
                self.required_action = required_action
                self.id = "runid"

        class DummyRequiredAction:
            type = "submit_tool_outputs"
            submit_tool_outputs = SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        id="toolid",
                        function=SimpleNamespace(
                            name="my_tool", arguments=json.dumps({"x": 1})
                        ),
                    )
                ]
            )

        mock_client.agents.runs.create = AsyncMock(
            return_value=DummyRun("requires_action", DummyRequiredAction())
        )
        mock_client.agents.runs.get = AsyncMock(
            side_effect=[
                DummyRun("requires_action", DummyRequiredAction()),
                DummyRun("completed"),
            ]
        )
        assistant_message = SimpleNamespace(
            role="assistant",
            content=[
                SimpleNamespace(
                    type="text", text=SimpleNamespace(value="Tool call complete!")
                )
            ],
        )

        def messages_list_side_effect(*args, **kwargs):
            return DummyAsyncIterator([assistant_message, assistant_message])

        mock_client.agents.messages.list.side_effect = messages_list_side_effect
        mock_client.agents.runs.submit_tool_outputs = AsyncMock()
        agent = AzureFoundryAgent(
            endpoint="https://fake-endpoint",
            model="gpt-4o",
            name="azure-agent",
            instructions="Test agent",
            verbose=True,
            tools=[lambda x: x],  # Dummy tool
        )
        workflow = AgentWorkflow(agents=[agent])
        memory = ChatMemoryBuffer.from_defaults()
        handler = workflow.run(user_msg="Trigger tool", memory=memory)
        events = []
        async for event in handler.stream_events():
            events.append(event)
        response = await handler
        assert "Tool call complete!" in response.response.content
