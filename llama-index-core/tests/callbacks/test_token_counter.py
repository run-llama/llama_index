"""Embeddings."""

from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks.token_counting import (
    TokenCountingHandler,
    get_llm_token_counts,
)
from llama_index.core.utilities.token_counting import TokenCounter

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


def test_get_tokens_from_response_with_explicit_none_usage_subfield() -> None:
    """A provider reporting a usage sub-field as explicit None should not crash."""

    class FakeCompletion:
        text = "hello"
        raw = {
            "usage": {"prompt_tokens": None, "completion_tokens": 5, "total_tokens": 5}
        }
        additional_kwargs = {}

    payload = {EventPayload.PROMPT: "hi", EventPayload.COMPLETION: FakeCompletion()}

    event = get_llm_token_counts(TokenCounter(), payload, event_id="x")

    # prompt_tokens falls back to being estimated from the prompt string
    # instead of staying None, and completion_tokens is read as reported.
    assert event.prompt_token_count != 0
    assert event.completion_token_count == 5
