import pytest
from unittest.mock import Mock
from types import SimpleNamespace

from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks.token_counting import TokenCountingHandler
from llama_index.core.llms import CompletionResponse


def test_token_budget_enforcement():
    """Test that the TokenCountingHandler enforces the budget in on_event_end."""
    # 1. Create a Mock Tokenizer
    # We configure it so every time it runs, it returns a list of 5 tokens.
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = [1, 2, 3, 4, 5]  # Always 5 tokens

    # 2. Setup Handler with Budget of 15
    handler = TokenCountingHandler(tokenizer=mock_tokenizer, token_budget=15)

    # 3. Create a valid CompletionResponse object
    # The handler expects an object with a .raw attribute, not just a string.
    response_object = CompletionResponse(text="generated text")

    # 4. First Event: Safe
    # Prompt (5 tokens) + Completion (5 tokens) = 10 tokens total.
    # Budget is 15. This should pass.
    handler.on_event_end(
        event_type=CBEventType.LLM,
        payload={
            EventPayload.PROMPT: "input prompt",
            EventPayload.COMPLETION: response_object,
        },
    )

    # 5. Second Event: Unsafe
    # This adds another 10 tokens. Total = 20.
    # Budget is 15. This should CRASH.
    with pytest.raises(ValueError) as excinfo:
        handler.on_event_end(
            event_type=CBEventType.LLM,
            payload={
                EventPayload.PROMPT: "input prompt",
                EventPayload.COMPLETION: response_object,
            },
        )

    # 6. Verify Error
    assert "Token budget exceeded" in str(excinfo.value)
    assert "Limit: 15" in str(excinfo.value)


def test_token_budget_via_callback_manager():
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = [1, 2, 3, 4, 5]

    handler = TokenCountingHandler(tokenizer=mock_tokenizer, token_budget=15)
    cm = CallbackManager([handler])

    # minimal object that satisfies your get_tokens_from_response fallback behavior
    resp = SimpleNamespace(raw=None, additional_kwargs={})

    # first event -> 10 tokens (5 prompt + 5 completion) OK
    cm.on_event_end(
        CBEventType.LLM,
        payload={EventPayload.PROMPT: "p", EventPayload.COMPLETION: resp},
    )

    # second event -> total 20 tokens => should exceed
    with pytest.raises(ValueError):
        cm.on_event_end(
            CBEventType.LLM,
            payload={EventPayload.PROMPT: "p", EventPayload.COMPLETION: resp},
        )
