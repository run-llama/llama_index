"""Tests for thinking_delta functionality in agents."""

import pytest
from typing import Any, AsyncGenerator, List
from unittest.mock import AsyncMock

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, LLMMetadata
from llama_index.core.llms import MockLLM
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.codeact_agent import CodeActAgent
from llama_index.core.agent.workflow.react_agent import ReActAgent
from llama_index.core.agent.workflow.workflow_events import AgentStream
from llama_index.core.workflow import Context


class MockThinkingLLM(MockLLM):
    """Mock LLM that supports thinking_delta in responses."""

    def __init__(
        self, thinking_deltas: List[str] = None, response_deltas: List[str] = None
    ):
        super().__init__()
        self._thinking_deltas = thinking_deltas or [
            "",
            "I need to think about this...",
            " Let me consider the options.",
        ]
        self._response_deltas = response_deltas or ["Hello", " there", "!"]
        self._response_index = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    async def astream_chat_with_tools(
        self, tools: List[Any], chat_history: List[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        """Stream chat responses with thinking_delta."""

        async def _gen():
            for i in range(len(self._response_deltas)):
                response_delta = self._response_deltas[i]
                thinking_delta = self._thinking_deltas[i]

                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content=response_delta,
                    ),
                    delta=response_delta,
                    additional_kwargs={"thinking_delta": thinking_delta},
                    raw={
                        "message": {
                            "content": response_delta,
                            "thinking": thinking_delta,
                        }
                    },
                )

        return _gen()

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        """Stream chat responses for CodeActAgent and ReActAgent."""

        async def _gen():
            for i in range(len(self._response_deltas)):
                response_delta = self._response_deltas[i]
                thinking_delta = self._thinking_deltas[i]

                yield ChatResponse(
                    message=ChatMessage(
                        role="assistant",
                        content=response_delta,
                    ),
                    delta=response_delta,
                    additional_kwargs={"thinking_delta": thinking_delta},
                    raw={
                        "message": {
                            "content": response_delta,
                            "thinking": thinking_delta,
                        }
                    },
                )

        return _gen()

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = False
    ):
        """Mock method for getting tool calls from response."""
        return []


def test_agent_stream_with_thinking_delta():
    """Test AgentStream creation and serialization with thinking_delta."""
    stream = AgentStream(
        delta="Hello",
        response="Hello there",
        current_agent_name="test_agent",
        thinking_delta="I'm thinking about this response...",
    )

    assert stream.delta == "Hello"
    assert stream.response == "Hello there"
    assert stream.thinking_delta == "I'm thinking about this response..."
    assert stream.current_agent_name == "test_agent"


def test_agent_stream_default_thinking_delta_none():
    """
    Test AgentStream with thinking_delta value of None does not cause Pydantic validation error.
    For Ollama, thinking_delta comes from the message's thinking field, which can be None.
    """
    stream = AgentStream(
        delta="Hello",
        response="Hello there",
        current_agent_name="test_agent",
        thinking_delta=None,
    )

    assert stream.thinking_delta is None


def test_agent_stream_default_thinking_delta():
    """Test AgentStream defaults thinking_delta to None."""
    stream = AgentStream(
        delta="Hello", response="Hello there", current_agent_name="test_agent"
    )

    assert stream.thinking_delta is None


def test_thinking_delta_extraction():
    """Test that thinking_delta is correctly extracted from ChatResponse additional_kwargs."""
    from llama_index.core.base.llms.types import ChatResponse, ChatMessage

    # should have thinking_delta present
    response_with_thinking = ChatResponse(
        message=ChatMessage(role="assistant", content="Hello"),
        delta="Hello",
        additional_kwargs={"thinking_delta": "I'm thinking..."},
    )

    thinking_delta = response_with_thinking.additional_kwargs.get("thinking_delta", "")
    assert thinking_delta == "I'm thinking..."

    # should default to None
    response_without_thinking = ChatResponse(
        message=ChatMessage(role="assistant", content="Hello"),
        delta="Hello",
        additional_kwargs={},
    )

    thinking_delta = response_without_thinking.additional_kwargs.get(
        "thinking_delta", None
    )
    assert thinking_delta is None


@pytest.mark.asyncio
async def test_streaming_an_agent_with_thinking_delta_none():
    """Test an agent runs properly with thinking_delta value of None"""
    mock_llm = MockThinkingLLM(thinking_deltas=[None], response_deltas=[None])
    agent = FunctionAgent(llm=mock_llm, streaming=True)

    # Mock context to capture stream events
    mock_context = AsyncMock(spec=Context)
    stream_events = []

    def capture_event(event):
        stream_events.append(event)

    mock_context.write_event_to_stream.side_effect = capture_event

    # Call the streaming method
    await agent._get_streaming_response(
        mock_context, [ChatMessage(role="user", content="test")], []
    )

    # Verify AgentStream events contain thinking_delta
    agent_streams = [event for event in stream_events if isinstance(event, AgentStream)]
    assert len(agent_streams) == 1  # 1 deltas from mock

    # Check that thinking deltas are passed through correctly
    assert agent_streams[0].thinking_delta is None


@pytest.mark.asyncio
async def test_function_agent_comprehensive_thinking_streaming():
    """Comprehensive test: FunctionAgent streams thinking_delta correctly."""
    mock_llm = MockThinkingLLM()
    agent = FunctionAgent(llm=mock_llm, streaming=True)

    # Mock context to capture stream events
    mock_context = AsyncMock(spec=Context)
    stream_events = []

    def capture_event(event):
        stream_events.append(event)

    mock_context.write_event_to_stream.side_effect = capture_event

    # Call the streaming method
    await agent._get_streaming_response(
        mock_context, [ChatMessage(role="user", content="test")], []
    )

    # Verify AgentStream events contain thinking_delta
    agent_streams = [event for event in stream_events if isinstance(event, AgentStream)]
    assert len(agent_streams) == 3  # 3 deltas from mock

    # Check that thinking deltas are passed through correctly
    assert agent_streams[0].thinking_delta == ""
    assert agent_streams[1].thinking_delta == "I need to think about this..."
    assert agent_streams[2].thinking_delta == " Let me consider the options."

    # Verify other fields are still correct
    assert agent_streams[0].delta == "Hello"
    assert agent_streams[1].delta == " there"
    assert agent_streams[2].delta == "!"


@pytest.mark.asyncio
async def test_codeact_agent_comprehensive_thinking_streaming():
    """Comprehensive test: CodeActAgent streams thinking_delta correctly."""

    def mock_code_execute(code: str):
        return {"output": "executed"}

    mock_llm = MockThinkingLLM()
    agent = CodeActAgent(
        llm=mock_llm, code_execute_fn=mock_code_execute, streaming=True
    )

    # Mock context to capture stream events
    mock_context = AsyncMock(spec=Context)
    stream_events = []

    def capture_event(event):
        stream_events.append(event)

    mock_context.write_event_to_stream.side_effect = capture_event

    # Call the streaming method
    await agent._get_streaming_response(
        mock_context, [ChatMessage(role="user", content="test")], []
    )

    # Verify AgentStream events contain thinking_delta
    agent_streams = [event for event in stream_events if isinstance(event, AgentStream)]
    assert len(agent_streams) == 3  # 3 deltas from mock

    # Check that thinking deltas are passed through correctly
    assert agent_streams[0].thinking_delta == ""
    assert agent_streams[1].thinking_delta == "I need to think about this..."
    assert agent_streams[2].thinking_delta == " Let me consider the options."


@pytest.mark.asyncio
async def test_react_agent_comprehensive_thinking_streaming():
    """Comprehensive test: ReActAgent streams thinking_delta correctly."""
    mock_llm = MockThinkingLLM()
    agent = ReActAgent(llm=mock_llm, streaming=True)

    # Mock context to capture stream events
    mock_context = AsyncMock(spec=Context)
    stream_events = []

    def capture_event(event):
        stream_events.append(event)

    mock_context.write_event_to_stream.side_effect = capture_event

    # Call the streaming method
    await agent._get_streaming_response(
        mock_context, [ChatMessage(role="user", content="test")]
    )

    # Verify AgentStream events contain thinking_delta
    agent_streams = [event for event in stream_events if isinstance(event, AgentStream)]
    assert len(agent_streams) == 3  # 3 deltas from mock

    # Check that thinking deltas are passed through correctly
    assert agent_streams[0].thinking_delta == ""
    assert agent_streams[1].thinking_delta == "I need to think about this..."
    assert agent_streams[2].thinking_delta == " Let me consider the options."


@pytest.mark.asyncio
async def test_agents_handle_missing_thinking_delta():
    """Test all agents handle LLMs without thinking_delta gracefully."""

    class MockNonThinkingLLM(MockLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(is_function_calling_model=True)

        async def astream_chat_with_tools(
            self, tools: List[Any], chat_history: List[ChatMessage], **kwargs: Any
        ) -> AsyncGenerator[ChatResponse, None]:
            async def _gen():
                yield ChatResponse(
                    message=ChatMessage(role="assistant", content="Hello!"),
                    delta="Hello!",
                    additional_kwargs={},  # No thinking_delta
                    raw={"message": {"content": "Hello!"}},
                )

            return _gen()

        def get_tool_calls_from_response(
            self, response: ChatResponse, error_on_no_tool_call: bool = False
        ):
            """Mock method for getting tool calls from response."""
            return []

    # Test FunctionAgent
    mock_llm = MockNonThinkingLLM()
    agent = FunctionAgent(llm=mock_llm, streaming=True)
    mock_context = AsyncMock(spec=Context)
    stream_events = []
    mock_context.write_event_to_stream.side_effect = lambda event: stream_events.append(
        event
    )

    await agent._get_streaming_response(
        mock_context, [ChatMessage(role="user", content="test")], []
    )
    agent_streams = [event for event in stream_events if isinstance(event, AgentStream)]
    assert len(agent_streams) == 1
    assert agent_streams[0].thinking_delta is None  # Should default to None
