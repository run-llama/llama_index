"""Embeddings."""

from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks.token_counting import TokenCountingHandler
from llama_index.core.llms import ChatMessage

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
    assert handler.llm_token_counts[0].total_token_count == 1

    # test actual counts
    # LLM should be one: the payload carries a prompt but no completion, and
    # EventPayload.RESPONSE is only read for the messages-based branch
    # Embedding should be one (single token chunk)
    assert handler.total_llm_token_count == 1
    assert handler.total_embedding_token_count == 1


def test_missing_completion_is_not_counted_as_a_token() -> None:
    """A payload with no completion should not be charged for the string "None"."""
    handler = TokenCountingHandler()

    handler.on_event_end(
        CBEventType.LLM,
        payload={EventPayload.PROMPT: "two", EventPayload.COMPLETION: None},
        event_id=TEST_ID,
    )

    event = handler.llm_token_counts[0]
    assert event.completion == ""
    assert event.completion_token_count == 0
    assert event.total_token_count == event.prompt_token_count


def test_missing_response_is_not_counted_as_a_token() -> None:
    """The messages branch should not be charged for the string "None" either."""
    handler = TokenCountingHandler()

    handler.on_event_end(
        CBEventType.LLM,
        payload={
            EventPayload.MESSAGES: [ChatMessage(role="user", content="hi")],
            EventPayload.RESPONSE: None,
        },
        event_id=TEST_ID,
    )

    event = handler.llm_token_counts[0]
    assert event.completion == ""
    assert event.completion_token_count == 0
