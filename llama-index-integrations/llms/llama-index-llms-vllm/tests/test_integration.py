import asyncio
import os

from llama_index.core.base.llms.types import MessageRole, ChatMessage, ChatResponse
from llama_index.llms.vllm import VllmServer
from llama_index.core.base.llms.types import CompletionResponse


def test_completion():
    remote = create_vllm()
    completion = remote.complete("When AI hype is over?")
    assert (
        isinstance(completion, CompletionResponse)
        and isinstance(completion.text, str)
        and len(completion.text) > 0
    )


def test_acompletion():
    remote = create_vllm()
    completion = asyncio.run(remote.acomplete("When AI hype is over?"))
    assert (
        isinstance(completion, CompletionResponse)
        and isinstance(completion.text, str)
        and len(completion.text) > 0
    )


def create_vllm():
    return VllmServer(
        api_url=os.environ.get("VLLM", "http://localhost:8000/generate"),
        max_new_tokens=123,
    )


def test_chat():
    remote = create_vllm()
    from llama_index.core.base.llms.types import ChatMessage

    chat = remote.chat(
        [ChatMessage(content="When AI hype is over?", role=MessageRole.USER)]
    )
    assert (
        isinstance(chat.message, ChatMessage)
        and chat.message.role == MessageRole.ASSISTANT
        and isinstance(chat.message.content, str)
        and len(chat.message.content) > 0
    )


def test_achat():
    remote = create_vllm()
    from llama_index.core.base.llms.types import ChatMessage

    chat = asyncio.run(
        remote.achat(
            [ChatMessage(content="When AI hype is over?", role=MessageRole.USER)]
        )
    )
    assert (
        isinstance(chat.message, ChatMessage)
        and chat.message.role == MessageRole.ASSISTANT
        and isinstance(chat.message.content, str)
        and len(chat.message.content) > 0
    )


def test_stream_completion():
    remote = create_vllm()
    prompt = "When AI hype is over?"
    completion = list(remote.stream_complete(prompt))[-1]
    assert isinstance(completion, CompletionResponse)
    assert completion.text.count(prompt) == 1
    print(completion)


def test_astream_completion():
    remote = create_vllm()
    prompt = "When AI hype is over?"

    async def concat2():
        return [c async for c in await remote.astream_complete(prompt)]

    completion = asyncio.run(concat2())[-1]
    assert isinstance(completion, CompletionResponse)
    assert completion.text.count(prompt) == 1
    print(completion)


def test_stream_chat():
    remote = create_vllm()
    prompt = "When AI hype is over?"
    chat = list(
        remote.stream_chat([ChatMessage(content=prompt, role=MessageRole.USER)])
    )[-1]
    assert isinstance(chat, ChatResponse)
    assert isinstance(chat.message, ChatMessage)
    assert chat.message.role == MessageRole.ASSISTANT
    assert chat.message.content.count(prompt) == 1
    print(chat)


def test_astream_chat():
    remote = create_vllm()
    prompt = "When AI hype is over?"

    async def concat2():
        return [
            c
            async for c in await remote.astream_chat(
                [ChatMessage(content=prompt, role=MessageRole.USER)]
            )
        ]

    chat = asyncio.run(concat2())[-1]
    assert isinstance(chat, ChatResponse)
    assert isinstance(chat.message, ChatMessage)
    assert chat.message.role == MessageRole.ASSISTANT
    assert chat.message.content.count(prompt) == 1
    print(chat)
