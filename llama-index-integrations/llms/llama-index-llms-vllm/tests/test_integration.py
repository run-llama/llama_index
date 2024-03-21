import asyncio
import os

import pytest
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


@pytest.mark.skip(reason="requires remotely running `vllm.entrypoints.api_server`")
class TestVllmIntegration:
    # replace to local_vllm(), it requires vllm installed and fail ..stream.. tests due to Not Implemented
    vllm = remote_vllm()

    def test_completion(self):
        completion = self.vllm.complete("When AI hype is over?")
        assert (
            isinstance(completion, CompletionResponse)
            and isinstance(completion.text, str)
            and len(completion.text) > 0
        )

    def test_acompletion(self):
        completion = asyncio.run(self.vllm.acomplete("When AI hype is over?"))
        assert (
            isinstance(completion, CompletionResponse)
            and isinstance(completion.text, str)
            and len(completion.text) > 0
        )

    def test_chat(self):
        from llama_index.core.base.llms.types import ChatMessage

        chat = self.vllm.chat(
            [ChatMessage(content="When AI hype is over?", role=MessageRole.USER)]
        )
        assert (
            isinstance(chat.message, ChatMessage)
            and chat.message.role == MessageRole.ASSISTANT
            and isinstance(chat.message.content, str)
            and len(chat.message.content) > 0
        )

    def test_achat(self):
        from llama_index.core.base.llms.types import ChatMessage

        chat = asyncio.run(
            self.vllm.achat(
                [ChatMessage(content="When AI hype is over?", role=MessageRole.USER)]
            )
        )
        assert (
            isinstance(chat.message, ChatMessage)
            and chat.message.role == MessageRole.ASSISTANT
            and isinstance(chat.message.content, str)
            and len(chat.message.content) > 0
        )

    def test_stream_completion(self):
        prompt = "When AI hype is over?"
        completion = list(self.vllm.stream_complete(prompt))[-1]
        assert isinstance(completion, CompletionResponse)
        assert completion.text.count(prompt) == 1
        print(completion)

    def test_astream_completion(self):
        prompt = "When AI hype is over?"

        async def concat():
            return [c async for c in await self.vllm.astream_complete(prompt)]

        completion = asyncio.run(concat())[-1]
        assert isinstance(completion, CompletionResponse)
        assert completion.text.count(prompt) == 1
        print(completion)

    def test_stream_chat(self):
        prompt = "When AI hype is over?"
        chat = list(
            self.vllm.stream_chat([ChatMessage(content=prompt, role=MessageRole.USER)])
        )[-1]
        assert isinstance(chat, ChatResponse)
        assert isinstance(chat.message, ChatMessage)
        assert chat.message.role == MessageRole.ASSISTANT
        assert chat.message.content.count(prompt) == 1
        print(chat)

    def test_astream_chat(self):
        prompt = "When AI hype is over?"

        async def concat():
            return [
                c
                async for c in await self.vllm.astream_chat(
                    [ChatMessage(content=prompt, role=MessageRole.USER)]
                )
            ]

        chat = asyncio.run(concat())[-1]
        assert isinstance(chat, ChatResponse)
        assert isinstance(chat.message, ChatMessage)
        assert chat.message.role == MessageRole.ASSISTANT
        assert chat.message.content.count(prompt) == 1
        print(chat)
