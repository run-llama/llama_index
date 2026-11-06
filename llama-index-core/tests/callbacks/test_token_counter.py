"""Embeddings."""

from llama_index.core.callbacks.schema import CBEventType
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


def test_on_event_end_llm_tool_call_fallback() -> None:
    """
    A tool-call reply has no text content; the usage-less fallback must
    still count and record the generated calls instead of "assistant: None"
    (https://github.com/run-llama/llama_index/issues/24016).
    """
    from openai.types.chat import ChatCompletionMessage

    from llama_index.core.base.llms.types import ChatMessage, ChatResponse
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
    from llama_index.llms.openai.utils import from_openai_message

    handler = TokenCountingHandler()
    arguments = (
        '{"city": "Paris", "unit": "celsius", "days": 3, '
        '"include": ["humidity", "wind", "uv_index"]}'
    )
    openai_message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather_forecast", "arguments": arguments},
            }
        ],
    )
    message = from_openai_message(openai_message, modalities=["text"])

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

    event = handler.llm_token_counts[0]
    assert "get_weather_forecast" in event.completion
    assert event.completion_token_count > len("assistant: None")
