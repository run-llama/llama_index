"""Embeddings."""

from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks.token_counting import (
    TokenCountingHandler,
    get_llm_token_counts,
    get_tokens_from_response,
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


def test_null_prompt_tokens_fall_back_to_estimated_count() -> None:
    """A null prompt token count should not crash token counting."""

    class FakeCompletion:
        text = "hello"
        raw = {
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": 5,
                "total_tokens": 5,
            }
        }
        additional_kwargs = {}

    payload = {
        EventPayload.PROMPT: "hi",
        EventPayload.COMPLETION: FakeCompletion(),
    }

    result = get_llm_token_counts(TokenCounter(), payload, event_id="x")

    assert result.prompt_token_count > 0
    assert result.completion_token_count == 5
    assert result.total_token_count == (
        result.prompt_token_count + result.completion_token_count
    )


def test_null_completion_tokens_fall_back_to_estimated_count() -> None:
    """A null completion token count should use the existing text fallback."""

    class FakeCompletion:
        text = "hello"
        raw = {"usage": {"prompt_tokens": 3, "completion_tokens": None}}
        additional_kwargs = {}

    result = get_llm_token_counts(
        TokenCounter(),
        {
            EventPayload.PROMPT: "hi",
            EventPayload.COMPLETION: FakeCompletion(),
        },
    )

    assert result.prompt_token_count == 3
    assert result.completion_token_count > 0


def test_valid_usage_counts_are_preserved() -> None:
    """Valid provider-reported token counts are returned unchanged."""

    class FakeCompletion:
        raw = {"usage": {"prompt_tokens": 3, "completion_tokens": 5}}
        additional_kwargs = {}

    assert get_tokens_from_response(FakeCompletion()) == (3, 5)


def test_null_preferred_usage_fields_use_alternatives_from_model() -> None:
    """Null preferred fields fall through to alternate fields on model usage."""

    class FakeUsage:
        def model_dump(self):
            return {
                "prompt_tokens": None,
                "input_tokens": 7,
                "completion_tokens": None,
                "output_tokens": 9,
            }

    class FakeCompletion:
        raw = {"usage": FakeUsage()}
        additional_kwargs = {}

    assert get_tokens_from_response(FakeCompletion()) == (7, 9)


def test_missing_usage_information_returns_zero_counts() -> None:
    """Missing or entirely null usage follows the existing empty-usage behavior."""

    class FakeCompletion:
        raw = {}
        additional_kwargs = {}

    assert get_tokens_from_response(FakeCompletion()) == (0, 0)

    class FakeNullUsageCompletion:
        raw = {"usage": None}
        additional_kwargs = {}

    assert get_tokens_from_response(FakeNullUsageCompletion()) == (0, 0)

    class FakeNullCountsCompletion:
        raw = {"usage": {"prompt_tokens": None, "completion_tokens": None}}
        additional_kwargs = {}

    assert get_tokens_from_response(FakeNullCountsCompletion()) == (0, 0)

    result = get_llm_token_counts(
        TokenCounter(),
        {
            EventPayload.PROMPT: "hi",
            EventPayload.COMPLETION: FakeNullCountsCompletion(),
        },
    )
    assert result.prompt_token_count > 0
    assert result.completion_token_count > 0


def test_null_usage_uses_additional_kwargs_fallback() -> None:
    """A null usage object keeps the supported additional_kwargs fallback."""

    class FakeCompletion:
        raw = {"usage": None}
        additional_kwargs = {"prompt_tokens": 4, "completion_tokens": 6}

    assert get_tokens_from_response(FakeCompletion()) == (4, 6)
