import asyncio
import os

from llama_index.core.base.llms.types import MessageRole, ChatMessage, ChatResponse

from llama_index.core.base.llms.types import CompletionResponse


def remote_vllm():
    from llama_index.llms.vllm import VllmServer

    return VllmServer(
        api_url=os.environ.get("VLLM", "http://localhost:8000/generate"),
    )


def local_vllm():
    from llama_index.llms.vllm import Vllm

    return Vllm(
        model="facebook/opt-350m",
    )


# replace to local_vllm(), it requires vllm installed and fail ..stream.. tests due to Not Implemented
vllm = remote_vllm()


def test_completion():
    completion = vllm.complete("When AI hype is over?")
    assert (
        isinstance(completion, CompletionResponse)
        and isinstance(completion.text, str)
        and len(completion.text) > 0
    )


def test_acompletion():
    completion = asyncio.run(vllm.acomplete("When AI hype is over?"))
    assert (
        isinstance(completion, CompletionResponse)
        and isinstance(completion.text, str)
        and len(completion.text) > 0
    )


def test_chat():
    from llama_index.core.base.llms.types import ChatMessage

    chat = vllm.chat(
        [ChatMessage(content="When AI hype is over?", role=MessageRole.USER)]
    )
    assert (
        isinstance(chat.message, ChatMessage)
        and chat.message.role == MessageRole.ASSISTANT
        and isinstance(chat.message.content, str)
        and len(chat.message.content) > 0
    )


def test_achat():
    from llama_index.core.base.llms.types import ChatMessage

    chat = asyncio.run(
        vllm.achat(
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
    prompt = "When AI hype is over?"
    completion = list(vllm.stream_complete(prompt))[-1]
    assert isinstance(completion, CompletionResponse)
    assert completion.text.count(prompt) == 1
    print(completion)


def test_astream_completion():
    prompt = "When AI hype is over?"

    async def concat2():
        return [c async for c in await vllm.astream_complete(prompt)]

    completion = asyncio.run(concat2())[-1]
    assert isinstance(completion, CompletionResponse)
    assert completion.text.count(prompt) == 1
    print(completion)


def test_stream_chat():
    prompt = "When AI hype is over?"
    chat = list(vllm.stream_chat([ChatMessage(content=prompt, role=MessageRole.USER)]))[
        -1
    ]
    assert isinstance(chat, ChatResponse)
    assert isinstance(chat.message, ChatMessage)
    assert chat.message.role == MessageRole.ASSISTANT
    assert chat.message.content.count(prompt) == 1
    print(chat)


def test_astream_chat():
    prompt = "When AI hype is over?"

    async def concat2():
        return [
            c
            async for c in await vllm.astream_chat(
                [ChatMessage(content=prompt, role=MessageRole.USER)]
            )
        ]

    chat = asyncio.run(concat2())[-1]
    assert isinstance(chat, ChatResponse)
    assert isinstance(chat.message, ChatMessage)
    assert chat.message.role == MessageRole.ASSISTANT
    assert chat.message.content.count(prompt) == 1
    print(chat)
