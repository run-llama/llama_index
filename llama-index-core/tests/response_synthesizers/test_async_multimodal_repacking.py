import asyncio
from typing import Any, AsyncGenerator

import pytest

from llama_index.core.base.llms.types import ChatMessage, ImageBlock
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.response.utils import aget_response_text
from llama_index.core.response_synthesizers.compact_and_refine import CompactAndRefine
from llama_index.core.response_synthesizers.refine import Refine


@pytest.fixture()
def message_chunks() -> list[ChatMessage]:
    png_1px_b64 = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    return [
        ChatMessage(content="context information"),
        ChatMessage(blocks=[ImageBlock(image=png_1px_b64)]),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("synthesizer_cls", [Refine, CompactAndRefine])
@pytest.mark.parametrize("prev_response", [None, "previous answer"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_content_processing_uses_caller_loop(
    monkeypatch: pytest.MonkeyPatch,
    message_chunks: list[ChatMessage],
    synthesizer_cls: type[Refine],
    prev_response: str | None,
    streaming: bool,
) -> None:
    loop = asyncio.get_running_loop()
    original_estimate_tokens = ImageBlock.aestimate_tokens
    estimation_calls = 0

    async def aestimate_tokens(block: ImageBlock, *args: Any, **kwargs: Any) -> int:
        nonlocal estimation_calls
        estimation_calls += 1
        # Async content processing may depend on work scheduled on the caller's loop.
        ready: asyncio.Future[None] = loop.create_future()
        loop.call_soon_threadsafe(ready.set_result, None)
        await ready
        return await original_estimate_tokens(block, *args, **kwargs)

    monkeypatch.setattr(ImageBlock, "aestimate_tokens", aestimate_tokens)
    llm = MockLLMWithChatMemoryOfLastCall(max_tokens=3, is_chat_model=True)
    synthesizer = synthesizer_cls(llm=llm, multimodal=True, streaming=streaming)

    response = await synthesizer.aget_response_from_messages(
        query_str="test", message_chunks=message_chunks, prev_response=prev_response
    )
    if streaming:
        assert isinstance(response, AsyncGenerator)
        response = await aget_response_text(response)

    assert response == "text text text"
    assert estimation_calls > 0
    assert llm.last_chat_messages is not None
    assert any(
        isinstance(block, ImageBlock)
        for message in llm.last_chat_messages
        for block in message.blocks
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("synthesizer_cls", [Refine, CompactAndRefine])
async def test_cancelling_synthesis_cancels_content_processing(
    monkeypatch: pytest.MonkeyPatch,
    message_chunks: list[ChatMessage],
    synthesizer_cls: type[Refine],
) -> None:
    loop = asyncio.get_running_loop()
    original_estimate_tokens = ImageBlock.aestimate_tokens
    estimation_started = False
    estimation_cancelled = False

    async def aestimate_tokens(block: ImageBlock, *args: Any, **kwargs: Any) -> int:
        nonlocal estimation_started, estimation_cancelled
        if not estimation_started:
            estimation_started = True
            loop.call_soon_threadsafe(synthesis_task.cancel)
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                estimation_cancelled = True
                raise
        return await original_estimate_tokens(block, *args, **kwargs)

    monkeypatch.setattr(ImageBlock, "aestimate_tokens", aestimate_tokens)
    llm = MockLLMWithChatMemoryOfLastCall(max_tokens=3, is_chat_model=True)
    synthesizer = synthesizer_cls(llm=llm, multimodal=True)
    synthesis_task = asyncio.create_task(
        synthesizer.aget_response_from_messages(
            query_str="test", message_chunks=message_chunks
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await synthesis_task

    assert estimation_cancelled
    assert llm.last_called_chat_function == []
