"""Embeddings."""

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks.token_counting import (
    TokenCountingHandler,
    get_tokens_from_response,
)

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


PROVIDER_TOKEN_COUNTS = {"prompt_tokens": 1200, "completion_tokens": 345}


@pytest.mark.parametrize(
    "raw",
    [
        # Bedrock Converse: usage present, but with camelCase keys
        {"usage": {"inputTokens": 1200, "outputTokens": 345, "totalTokens": 1545}},
        # Bedrock Converse streaming metadata chunk: no top-level usage
        {"metadata": {"usage": {"inputTokens": 1200, "outputTokens": 345}}},
        None,
    ],
)
def test_get_tokens_from_response_falls_back_to_additional_kwargs(raw) -> None:
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="Paris."),
        raw=raw,
        additional_kwargs=PROVIDER_TOKEN_COUNTS,
    )

    assert get_tokens_from_response(response) == (1200, 345)


def test_get_tokens_from_response_prefers_raw_usage() -> None:
    response = CompletionResponse(
        text="Paris.",
        raw={"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        additional_kwargs=PROVIDER_TOKEN_COUNTS,
    )

    assert get_tokens_from_response(response) == (10, 2)


def test_on_event_end_uses_token_counts_from_additional_kwargs() -> None:
    handler = TokenCountingHandler()
    response = ChatResponse(
        message=ChatMessage(role="assistant", content="Paris."),
        raw={"usage": {"inputTokens": 1200, "outputTokens": 345}},
        additional_kwargs=PROVIDER_TOKEN_COUNTS,
    )

    handler.on_event_end(
        CBEventType.LLM,
        payload={
            EventPayload.MESSAGES: [ChatMessage(role="user", content="Capital?")],
            EventPayload.RESPONSE: response,
        },
        event_id=TEST_ID,
    )

    assert handler.prompt_llm_token_count == 1200
    assert handler.completion_llm_token_count == 345
