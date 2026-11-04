import pytest

import llama_index.core
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
    TextBlock,
    ThinkingBlock,
)
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.callbacks.token_counting import TokenCountingHandler
from llama_index.core.utilities.token_counting import TokenCounter


@pytest.fixture(autouse=True)
def single_token_global_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llama_index.core, "global_tokenizer", lambda text: [text] if text else []
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (ChatMessage(role="user", content="hello world"), 18),
        (
            ChatMessage(
                role="assistant",
                blocks=[TextBlock(text="hello"), TextBlock(text="world")],
            ),
            22,
        ),
        (ChatMessage(role=MessageRole.TOOL, content="result"), 11),
        (
            ChatMessage(
                role="assistant",
                blocks=[ThinkingBlock(num_tokens=7), TextBlock(text="answer")],
            ),
            25,
        ),
    ],
)
@pytest.mark.asyncio
async def test_custom_tokenizer_counts_message_blocks(
    message: ChatMessage, expected: int
) -> None:
    counter = TokenCounter(tokenizer=list)

    assert counter.estimate_tokens_in_messages([message]) == expected
    assert await counter.aestimate_tokens_in_messages([message]) == expected


@pytest.mark.asyncio
async def test_default_tokenizer_and_empty_messages() -> None:
    counter = TokenCounter()
    messages = [ChatMessage(role="user", content="hello world")]

    assert counter.estimate_tokens_in_messages(messages) == 5
    assert await counter.aestimate_tokens_in_messages(messages) == 5
    assert counter.estimate_tokens_in_messages([]) == 0
    assert await counter.aestimate_tokens_in_messages([]) == 0


def test_callback_fallback_uses_custom_tokenizer() -> None:
    handler = TokenCountingHandler(tokenizer=list)

    handler.on_event_end(
        CBEventType.LLM,
        payload={
            EventPayload.MESSAGES: [ChatMessage(role="user", content="hello world")],
            EventPayload.RESPONSE: ChatResponse(message=ChatMessage(content="ok")),
        },
    )

    assert handler.prompt_llm_token_count == 18


def test_callback_preserves_provider_usage() -> None:
    handler = TokenCountingHandler(tokenizer=list)

    handler.on_event_end(
        CBEventType.LLM,
        payload={
            EventPayload.MESSAGES: [ChatMessage(role="user", content="hello world")],
            EventPayload.RESPONSE: ChatResponse(
                message=ChatMessage(content="ok"),
                raw={"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
            ),
        },
    )

    assert handler.prompt_llm_token_count == 7
    assert handler.completion_llm_token_count == 3
