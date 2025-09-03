from llama_index.core.agent.workflow.workflow_events import AgentStream


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


def test_agent_stream_default_thinking_delta():
    """Test AgentStream defaults thinking_delta to empty string."""
    stream = AgentStream(
        delta="Hello", response="Hello there", current_agent_name="test_agent"
    )

    assert stream.thinking_delta == ""


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

    # should default to empty string
    response_without_thinking = ChatResponse(
        message=ChatMessage(role="assistant", content="Hello"),
        delta="Hello",
        additional_kwargs={},
    )

    thinking_delta = response_without_thinking.additional_kwargs.get(
        "thinking_delta", ""
    )
    assert thinking_delta == ""
