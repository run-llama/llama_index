from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.llms.vertex import Vertex
from llama_index.llms.vertex.utils import acompletion_with_retry, completion_with_retry


def test_vertex_metadata_function_calling():
    """Test that Vertex LLM metadata correctly identifies Gemini models as function calling models."""
    # This test uses mocks to avoid actual API calls
    with patch(
        "llama_index.llms.vertex.gemini_utils.create_gemini_client"
    ) as mock_create_client:
        # Test Gemini model
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")
        metadata = llm.metadata

        assert metadata.is_function_calling_model is True
        assert metadata.model_name == "gemini-pro"
        assert metadata.is_chat_model is True


def test_vertex_metadata_non_function_calling():
    """Test that Vertex LLM metadata correctly identifies non-Gemini models as non-function calling models."""
    with patch(
        "vertexai.language_models.ChatModel.from_pretrained"
    ) as mock_from_pretrained:
        mock_chat_client = Mock()
        mock_from_pretrained.return_value = mock_chat_client

        llm = Vertex(model="chat-bison")
        metadata = llm.metadata

        assert metadata.is_function_calling_model is False
        assert metadata.model_name == "chat-bison"
        assert metadata.is_chat_model is True


@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("vertexai.language_models.ChatModel.from_pretrained")
def test_vertex_complete_falls_back_to_chat(
    mock_from_pretrained: Mock, mock_init_vertexai: Mock
) -> None:
    mock_from_pretrained.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="chat-bison", project="test-project")
    response = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="fallback response")
    )

    with patch.object(Vertex, "chat", return_value=response) as mock:
        completion = llm.complete("Hi")

    assert completion.text == "fallback response"
    mock.assert_called_once()


@pytest.mark.asyncio
@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("vertexai.language_models.TextGenerationModel.from_pretrained")
async def test_vertex_achat_falls_back_to_acomplete(
    mock_from_pretrained: Mock, mock_init_vertexai: Mock
) -> None:
    mock_from_pretrained.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="text-bison", project="test-project")
    completion = CompletionResponse(text="fallback response")

    with patch.object(
        Vertex, "acomplete", new=AsyncMock(return_value=completion)
    ) as mock:
        response = await llm.achat([ChatMessage(role=MessageRole.USER, content="Hi")])

    assert response.message.content == "fallback response"
    mock.assert_awaited_once()


@pytest.mark.asyncio
@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("vertexai.language_models.TextGenerationModel.from_pretrained")
async def test_vertex_astream_chat_falls_back_to_astream_complete(
    mock_from_pretrained: Mock, mock_init_vertexai: Mock
) -> None:
    mock_from_pretrained.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="text-bison", project="test-project")

    async def completion_stream():
        yield CompletionResponse(text="Hi", delta="Hi")
        yield CompletionResponse(text="Hi!", delta="!")

    with patch.object(
        Vertex,
        "astream_complete",
        new=AsyncMock(return_value=completion_stream()),
    ) as mock:
        stream = await llm.astream_chat(
            [ChatMessage(role=MessageRole.USER, content="Hi")]
        )
        responses = [response async for response in stream]

    assert [response.delta for response in responses] == ["Hi", "!"]
    assert [response.message.content for response in responses] == ["Hi", "Hi!"]
    mock.assert_awaited_once()


@pytest.mark.asyncio
@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("vertexai.language_models.ChatModel.from_pretrained")
@patch("llama_index.llms.vertex.base.acompletion_with_retry")
async def test_vertex_astream_chat_streams_chat_models(
    mock_acompletion_with_retry: AsyncMock,
    mock_from_pretrained: Mock,
    mock_init_vertexai: Mock,
) -> None:
    mock_from_pretrained.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="chat-bison", project="test-project")

    async def chat_stream():
        yield SimpleNamespace(text="Hi")
        yield SimpleNamespace(text="!")

    mock_acompletion_with_retry.return_value = chat_stream()
    stream = await llm.astream_chat([ChatMessage(role=MessageRole.USER, content="Hi")])
    responses = [response async for response in stream]

    assert [response.delta for response in responses] == ["Hi", "!"]
    assert [response.message.content for response in responses] == ["Hi", "Hi!"]
    mock_acompletion_with_retry.assert_awaited_once()


@pytest.mark.asyncio
@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("vertexai.language_models.ChatModel.from_pretrained")
async def test_vertex_astream_complete_falls_back_to_astream_chat(
    mock_from_pretrained: Mock, mock_init_vertexai: Mock
) -> None:
    mock_from_pretrained.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="chat-bison", project="test-project")

    async def chat_stream():
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
            delta="Hi",
        )
        yield ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi!"),
            delta="!",
        )

    with patch.object(
        Vertex,
        "astream_chat",
        new=AsyncMock(return_value=chat_stream()),
    ) as mock:
        stream = await llm.astream_complete("Hi")
        responses = [response async for response in stream]

    assert [response.delta for response in responses] == ["Hi", "!"]
    assert [response.text for response in responses] == ["Hi", "Hi!"]
    mock.assert_awaited_once()


@pytest.mark.asyncio
@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("vertexai.language_models.TextGenerationModel.from_pretrained")
@patch("llama_index.llms.vertex.base.acompletion_with_retry")
async def test_vertex_astream_complete_streams_completion_models(
    mock_acompletion_with_retry: AsyncMock,
    mock_from_pretrained: Mock,
    mock_init_vertexai: Mock,
) -> None:
    mock_from_pretrained.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="text-bison", project="test-project")

    async def completion_stream():
        yield SimpleNamespace(text="Hi")
        yield SimpleNamespace(text="!")

    mock_acompletion_with_retry.return_value = completion_stream()
    stream = await llm.astream_complete("Hi")
    responses = [response async for response in stream]

    assert [response.delta for response in responses] == ["Hi", "!"]
    assert [response.text for response in responses] == ["Hi", "Hi!"]
    mock_acompletion_with_retry.assert_awaited_once()


@pytest.mark.asyncio
@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
async def test_vertex_gemini_astream_chat_uses_sync_fallback(
    mock_create_client: Mock, mock_init_vertexai: Mock
) -> None:
    mock_create_client.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="gemini-pro", project="test-project")
    chat_stream = iter(
        [
            ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
                delta="Hi",
            ),
            ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content="Hi!"),
                delta="!",
            ),
        ]
    )

    with patch.object(llm, "_stream_chat_impl", return_value=chat_stream) as mock:
        stream = await llm.astream_chat(
            [ChatMessage(role=MessageRole.USER, content="Hi")]
        )
        responses = [response async for response in stream]

    assert [response.delta for response in responses] == ["Hi", "!"]
    assert [response.message.content for response in responses] == ["Hi", "Hi!"]
    mock.assert_called_once()


@pytest.mark.asyncio
@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
async def test_vertex_gemini_astream_complete_uses_sync_fallback(
    mock_create_client: Mock, mock_init_vertexai: Mock
) -> None:
    mock_create_client.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="gemini-pro", project="test-project")
    completion_stream = iter(
        [
            CompletionResponse(text="Hi", delta="Hi"),
            CompletionResponse(text="Hi!", delta="!"),
        ]
    )

    with patch.object(
        llm, "_stream_complete_impl", return_value=completion_stream
    ) as mock:
        stream = await llm.astream_complete("Hi")
        responses = [response async for response in stream]

    assert [response.delta for response in responses] == ["Hi", "!"]
    assert [response.text for response in responses] == ["Hi", "Hi!"]
    mock.assert_called_once_with("Hi")


@pytest.mark.asyncio
async def test_acompletion_with_retry_gemini_stream_raises() -> None:
    mock_client = Mock()
    mock_client.start_chat.return_value = Mock()

    with pytest.raises(
        ValueError,
        match="Async streaming is not supported for Gemini through this Vertex integration.",
    ):
        await acompletion_with_retry(
            client=mock_client,
            prompt="Hi",
            is_gemini=True,
            stream=True,
        )


def test_completion_with_retry_gemini_binds_tool_config_on_model() -> None:
    class FakeChatSession:
        def __init__(self) -> None:
            self.send_message = Mock()

    class FakeClient:
        instances = []

        def __init__(
            self,
            model_name: str,
            generation_config=None,
            safety_settings=None,
            tools=None,
            tool_config=None,
            system_instruction=None,
            labels=None,
        ) -> None:
            self._model_name = model_name
            self._generation_config = generation_config
            self._safety_settings = safety_settings
            self._tools = tools
            self._tool_config = tool_config
            self._system_instruction = system_instruction
            self._labels = labels
            self.chat_session = FakeChatSession()
            self.history = None
            type(self).instances.append(self)

        def start_chat(self, history=None):
            self.history = history
            return self.chat_session

    tool_config = object()
    base_client = FakeClient("gemini-pro")

    with patch("llama_index.llms.vertex.utils.to_gemini_tools", return_value=["tool"]):
        completion_with_retry(
            client=base_client,
            prompt="Hi",
            is_gemini=True,
            params={"message_history": ["history"]},
            tools=[
                {"name": "add_numbers", "description": "Add numbers", "parameters": {}}
            ],
            tool_config=tool_config,
            temperature=0.2,
            max_output_tokens=32,
        )

    assert len(FakeClient.instances) == 2
    bound_client = FakeClient.instances[1]
    assert bound_client._tools == ["tool"]
    assert bound_client._tool_config is tool_config
    assert bound_client.history == ["history"]
    bound_client.chat_session.send_message.assert_called_once_with(
        "Hi",
        stream=False,
        generation_config={"temperature": 0.2, "max_output_tokens": 32},
    )


@pytest.mark.asyncio
async def test_acompletion_with_retry_gemini_binds_tool_config_on_model() -> None:
    class FakeChatSession:
        def __init__(self) -> None:
            self.send_message_async = AsyncMock()

    class FakeClient:
        instances = []

        def __init__(
            self,
            model_name: str,
            generation_config=None,
            safety_settings=None,
            tools=None,
            tool_config=None,
            system_instruction=None,
            labels=None,
        ) -> None:
            self._model_name = model_name
            self._generation_config = generation_config
            self._safety_settings = safety_settings
            self._tools = tools
            self._tool_config = tool_config
            self._system_instruction = system_instruction
            self._labels = labels
            self.chat_session = FakeChatSession()
            self.history = None
            type(self).instances.append(self)

        def start_chat(self, history=None):
            self.history = history
            return self.chat_session

    tool_config = object()
    base_client = FakeClient("gemini-pro")

    with patch("llama_index.llms.vertex.utils.to_gemini_tools", return_value=["tool"]):
        await acompletion_with_retry(
            client=base_client,
            prompt="Hi",
            is_gemini=True,
            params={"message_history": ["history"]},
            tools=[
                {"name": "add_numbers", "description": "Add numbers", "parameters": {}}
            ],
            tool_config=tool_config,
            temperature=0.2,
            max_output_tokens=32,
        )

    assert len(FakeClient.instances) == 2
    bound_client = FakeClient.instances[1]
    assert bound_client._tools == ["tool"]
    assert bound_client._tool_config is tool_config
    assert bound_client.history == ["history"]
    bound_client.chat_session.send_message_async.assert_awaited_once_with(
        "Hi",
        generation_config={"temperature": 0.2, "max_output_tokens": 32},
    )


@pytest.mark.asyncio
@patch("llama_index.llms.vertex.base.init_vertexai")
@patch("vertexai.language_models.CodeChatModel.from_pretrained")
@patch("llama_index.llms.vertex.base.acompletion_with_retry")
async def test_vertex_achat_code_models_await_twice(
    mock_acompletion_with_retry: AsyncMock,
    mock_from_pretrained: Mock,
    mock_init_vertexai: Mock,
) -> None:
    mock_from_pretrained.return_value = Mock()
    mock_init_vertexai.return_value = None

    llm = Vertex(model="codechat-bison", project="test-project", iscode=True)

    generation = Mock()
    generation.candidates = [Mock(function_calls=[])]
    generation.text = "fallback response"

    async def second_await():
        return generation

    mock_acompletion_with_retry.return_value = second_await()

    response = await llm.achat([ChatMessage(role=MessageRole.USER, content="Hi")])

    assert response.message.content == "fallback response"
    mock_acompletion_with_retry.assert_awaited_once()
