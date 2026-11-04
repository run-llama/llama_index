import asyncio
from typing import Any, Awaitable, Callable, List, Optional

import pytest
from pydantic import ValidationError

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory.memory import BaseMemoryBlock, Memory


class CallbackMemoryBlock(BaseMemoryBlock[str]):
    reader: Callable[..., Awaitable[str]]

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **kwargs: Any
    ) -> str:
        return await self.reader(messages, **kwargs)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        pass


@pytest.mark.parametrize("concurrency", [0, -1])
def test_read_concurrency_must_be_positive(concurrency: int) -> None:
    with pytest.raises(ValidationError, match="memory_blocks_concurrency"):
        Memory.from_defaults(memory_blocks_concurrency=concurrency)


def test_from_defaults_preserves_read_concurrency() -> None:
    assert Memory.from_defaults().memory_blocks_concurrency == 1
    assert (
        Memory.from_defaults(memory_blocks_concurrency=3).memory_blocks_concurrency == 3
    )


@pytest.mark.asyncio
async def test_default_reads_stop_before_next_block_on_failure() -> None:
    calls = []

    async def failing_reader(*args: Any, **kwargs: Any) -> str:
        calls.append("first")
        raise RuntimeError("read failed")

    async def later_reader(*args: Any, **kwargs: Any) -> str:
        calls.append("second")
        return "later content"

    memory = Memory.from_defaults(
        memory_blocks=[
            CallbackMemoryBlock(name="second", priority=1, reader=later_reader),
            CallbackMemoryBlock(name="first", priority=2, reader=failing_reader),
        ]
    )
    with pytest.raises(RuntimeError, match="read failed"):
        await memory._get_memory_blocks_content([])
    assert calls == ["first"]


@pytest.mark.asyncio
async def test_reads_overlap_with_limit_and_preserve_priority_order() -> None:
    names = ["fourth", "first", "third", "second"]
    priorities = {"first": 4, "second": 3, "third": 2, "fourth": 1}
    started = {name: asyncio.Event() for name in names}
    release = {name: asyncio.Event() for name in names}
    received = {}
    active = 0
    peak = 0

    def reader_for(name: str) -> Callable[..., Awaitable[str]]:
        async def reader(messages: List[ChatMessage], **kwargs: Any) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            received[name] = (messages, kwargs)
            started[name].set()
            try:
                await release[name].wait()
                return name
            finally:
                active -= 1

        return reader

    memory = Memory.from_defaults(
        session_id="concurrent-session",
        memory_blocks_concurrency=2,
        memory_blocks=[
            CallbackMemoryBlock(
                name=name, priority=priorities[name], reader=reader_for(name)
            )
            for name in names
        ],
    )
    history = [ChatMessage(role="user", content="Previous message")]
    task = asyncio.create_task(
        memory._get_memory_blocks_content(history, input="New message", context="value")
    )
    try:
        await asyncio.wait_for(started["first"].wait(), timeout=5)
        await asyncio.wait_for(started["second"].wait(), timeout=5)
        assert not started["third"].is_set()
        assert not started["fourth"].is_set()

        release["second"].set()
        await asyncio.wait_for(started["third"].wait(), timeout=5)
        assert not started["fourth"].is_set()
        release["third"].set()
        await asyncio.wait_for(started["fourth"].wait(), timeout=5)
        release["fourth"].set()
        release["first"].set()
        result = await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert peak == 2
    assert active == 0
    assert list(result.items()) == [(name, name) for name in priorities]
    assert len(history) == 1
    for messages, kwargs in received.values():
        assert [message.content for message in messages] == [
            "Previous message",
            "New message",
        ]
        assert kwargs == {"session_id": "concurrent-session", "context": "value"}


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_content", [False, True])
async def test_failed_read_cancels_and_awaits_other_reads(
    invalid_content: bool,
) -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def waiting_reader(*args: Any, **kwargs: Any) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
            return "unreachable"
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()

    async def failing_reader(*args: Any, **kwargs: Any) -> Any:
        await started.wait()
        if invalid_content:
            return {"unsupported": "content"}
        raise RuntimeError("read failed")

    memory = Memory.from_defaults(
        memory_blocks_concurrency=2,
        memory_blocks=[
            CallbackMemoryBlock(name="waiting", reader=waiting_reader),
            CallbackMemoryBlock(name="failing", reader=failing_reader),
        ],
    )
    expected_error = ValueError if invalid_content else RuntimeError
    with pytest.raises(expected_error):
        await asyncio.wait_for(memory._get_memory_blocks_content([]), timeout=5)
    assert cleaned_up.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_child", [False, True])
async def test_cancelled_retrieval_awaits_active_reads(cancel_child: bool) -> None:
    started = [asyncio.Event(), asyncio.Event()]
    cleaned_up = [asyncio.Event(), asyncio.Event()]
    cancel = asyncio.Event()

    def reader_for(index: int) -> Callable[..., Awaitable[str]]:
        async def reader(*args: Any, **kwargs: Any) -> str:
            started[index].set()
            try:
                if cancel_child and index == 0:
                    await cancel.wait()
                    raise asyncio.CancelledError
                await asyncio.Event().wait()
                return "unreachable"
            finally:
                if index == 1:
                    await asyncio.sleep(0)
                cleaned_up[index].set()

        return reader

    memory = Memory.from_defaults(
        memory_blocks_concurrency=2,
        memory_blocks=[
            CallbackMemoryBlock(name=str(index), reader=reader_for(index))
            for index in range(2)
        ],
    )
    task = asyncio.create_task(memory._get_memory_blocks_content([]))
    try:
        for event in started:
            await asyncio.wait_for(event.wait(), timeout=5)
        if cancel_child:
            cancel.set()
        else:
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert all(event.is_set() for event in cleaned_up)
