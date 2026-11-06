"""Embeddings."""

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ToolCallBlock,
)
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks.token_counting import TokenCountingHandler

TEST_PAYLOAD = {"chunks": ["one"], "formatted_prompt": "two", "response": "three"}
TEST_ID = "my id"


def test_on_event_start() -> None:
    """Test event start."""
    handler = TokenCountingHandler()

    event_id = handler.on_event_start(
        CBEventType.LLM, payload=TEST_PAYLOAD, event_id=TEST_ID
    )

    assert event_id == TEST_ID

    event_id = handler.on_event_start(
        CBEventType.EMBEDDING, payload=TEST_PAYLOAD, event_id=TEST_ID
    )

    assert event_id == TEST_ID
    assert len(handler.llm_token_counts) == 0
    assert len(handler.embedding_token_counts) == 0


def test_on_event_end() -> None:
    """Test event end."""
    handler = TokenCountingHandler()

    handler.on_event_end(CBEventType.LLM, payload=TEST_PAYLOAD, event_id=TEST_ID)

    assert len(handler.llm_token_counts) == 1
    assert len(handler.embedding_token_counts) == 0

    handler.on_event_end(CBEventType.EMBEDDING, payload=TEST_PAYLOAD, event_id=TEST_ID)

    assert len(handler.llm_token_counts) == 1
    assert len(handler.embedding_token_counts) == 1

    assert handler.embedding_token_counts[0].total_token_count == 1
    assert handler.llm_token_counts[0].total_token_count == 2

    # test actual counts
    # LLM should be two (prompt plus response)
    # Embedding should be one (single token chunk)
    assert handler.total_llm_token_count == 2
    assert handler.total_embedding_token_count == 1


def _completion_token_count(message: ChatMessage) -> int:
    """Completion tokens estimated for a response with no usage data."""
    handler = TokenCountingHandler()
    handler.on_event_end(
        CBEventType.LLM,
        payload={
            EventPayload.MESSAGES: [
                ChatMessage(role="user", content="Weather in Paris?")
            ],
            EventPayload.RESPONSE: ChatResponse(message=message),
        },
        event_id=TEST_ID,
    )
    assert len(handler.llm_token_counts) == 1
    return handler.llm_token_counts[0].completion_token_count


def test_tool_call_completion_counts_tool_call() -> None:
    """A tool-call-only completion must count the tool name/arguments.

    Regression test for #24016: str(response) renders such a reply as the
    literal "assistant: None" (3 tokens), the same as an empty reply.
    """
    arguments = (
        '{"city": "Paris", "unit": "celsius", "days": 3, '
        '"include": ["humidity", "wind", "uv_index"]}'
    )
    tool_call_message = ChatMessage(
        role="assistant",
        blocks=[
            ToolCallBlock(
                tool_call_id="call_1",
                tool_name="get_weather",
                tool_kwargs=arguments,
            )
        ],
    )
    empty_message = ChatMessage(role="assistant")

    tool_call_count = _completion_token_count(tool_call_message)
    empty_count = _completion_token_count(empty_message)

    assert tool_call_count > empty_count
    assert tool_call_count > 3
