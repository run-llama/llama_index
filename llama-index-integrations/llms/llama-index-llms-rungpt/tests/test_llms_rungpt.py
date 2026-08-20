import asyncio
from unittest.mock import patch

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
)
from llama_index.llms.rungpt import RunGptLLM


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in RunGptLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def test_acomplete_offloads_blocking_call_to_thread():
    llm = RunGptLLM()
    with (
        patch.object(
            RunGptLLM, "complete", return_value=CompletionResponse(text="hi")
        ) as mock_complete,
        patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy,
    ):
        response = asyncio.run(llm.acomplete("prompt"))

    # complete() issues a blocking requests.post; acomplete must run it off the loop.
    spy.assert_called_once()
    mock_complete.assert_called_once()
    assert response.text == "hi"


def test_achat_offloads_blocking_call_to_thread():
    llm = RunGptLLM()
    reply = ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="hi"))
    with (
        patch.object(RunGptLLM, "chat", return_value=reply) as mock_chat,
        patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy,
    ):
        response = asyncio.run(
            llm.achat([ChatMessage(role=MessageRole.USER, content="q")])
        )

    spy.assert_called_once()
    mock_chat.assert_called_once()
    assert response.message.content == "hi"
