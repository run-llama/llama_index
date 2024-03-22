import asyncio
import os

from llama_index.core.base.llms.types import MessageRole, ChatMessage, ChatResponse

from llama_index.core.base.llms.types import CompletionResponse
from vcr.record_mode import RecordMode
from vcr.unittest import VCRTestCase


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


prompt = "When AI hype is over?"


class TestVllmIntegration(VCRTestCase):
    # replace to local_vllm(), it requires vllm installed and fail ..stream.. tests due to Not Implemented
    vllm = remote_vllm()

    # drop it to record new cassete files
    def _get_vcr_kwargs(self, **kwargs):
        return {"record_mode": RecordMode.NONE}

    def test_completion(self):
        completion = self.vllm.complete(prompt)
        assert isinstance(completion, CompletionResponse)
        assert isinstance(completion.text, str)
        assert len(completion.text) > 0
        assert prompt in completion.text

    def test_acompletion(self):
        completion = asyncio.run(self.vllm.acomplete(prompt))
        assert isinstance(completion, CompletionResponse)
        assert isinstance(completion.text, str)
        assert len(completion.text) > 0
        assert prompt in completion.text

    def test_chat(self):
        chat = self.vllm.chat([ChatMessage(content=prompt, role=MessageRole.USER)])
        assert isinstance(chat.message, ChatMessage)
        assert chat.message.role == MessageRole.ASSISTANT
        assert isinstance(chat.message.content, str)
        assert len(chat.message.content) > 0
        assert prompt in chat.message.content

    def test_achat(self):
        chat = asyncio.run(
            self.vllm.achat([ChatMessage(content=prompt, role=MessageRole.USER)])
        )
        assert isinstance(chat.message, ChatMessage)
        assert chat.message.role == MessageRole.ASSISTANT
        assert isinstance(chat.message.content, str)
        assert len(chat.message.content) > 0
        assert prompt in chat.message.content

    def test_stream_completion(self):
        stream = list(self.vllm.stream_complete(prompt))
        first = stream[0]
        completion = stream[-1]
        for i in first, completion:
            assert isinstance(i, CompletionResponse)
            assert i.text.count(prompt) == 1
        assert completion.text.count(first.text) == 1
        assert first.text.count(completion.text) == 0

    def test_astream_completion(self):
        async def concat():
            return [c async for c in await self.vllm.astream_complete(prompt)]

        stream = asyncio.run(concat())
        first = stream[0]
        assert isinstance(first, CompletionResponse)
        assert first.text.count(prompt) == 1, "no repeats please"
        completion = stream[-1]
        assert isinstance(completion, CompletionResponse)
        assert completion.text.count(first.text) == 1, "no repeats please"
        assert completion.text not in first.text

    def test_stream_chat(self):
        stream = list(
            self.vllm.stream_chat([ChatMessage(content=prompt, role=MessageRole.USER)])
        )
        first = stream[0]
        chat = stream[-1]
        for i in first, chat:
            assert isinstance(i, ChatResponse)
            assert isinstance(i.message, ChatMessage)
            assert i.message.role == MessageRole.ASSISTANT
            assert i.message.content.count(prompt) == 1
        assert chat.message.content.count(first.message.content) == 1
        assert chat.message.content not in first.message.content, "not equal"

    def test_astream_chat(self):
        async def concat():
            return [
                c
                async for c in await self.vllm.astream_chat(
                    [ChatMessage(content=prompt, role=MessageRole.USER)]
                )
            ]

        stream = asyncio.run(concat())
        first = stream[0]
        chat = stream[-1]
        for i in first, chat:
            assert isinstance(i, ChatResponse)
            assert isinstance(i.message, ChatMessage)
            assert i.message.role == MessageRole.ASSISTANT
            assert i.message.content.count(prompt) == 1
        assert chat.message.content.count(first.message.content) == 1
        assert chat.message.content not in first.message.content, "not equal"
