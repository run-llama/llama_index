import asyncio

import pytest

from typing import Any, Generator, List, Optional

from llama_index.core.base.llms.types import (
    ChatMessage,
    DocumentBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.memory.memory import BaseMemoryBlock, Memory
from llama_index.core.storage.chat_store.sql import MessageStatus, SQLAlchemyChatStore


@pytest.fixture()
def memory(event_loop: asyncio.AbstractEventLoop) -> Generator[Memory, None, None]:
    """Create a basic memory instance and close its database after each test."""
    memory = Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_user",
    )
    yield memory
    assert isinstance(memory.sql_store, SQLAlchemyChatStore)
    if memory.sql_store._async_engine is not None:
        event_loop.run_until_complete(memory.sql_store._async_engine.dispose())


@pytest.mark.asyncio
async def test_initialization(memory):
    """Test that memory initializes correctly."""
    assert memory.token_limit == 1000
    assert memory.token_flush_size == 700
    assert memory.session_id == "test_user"


@pytest.mark.asyncio
async def test_estimate_token_count_text(memory):
    """Test token counting for text."""
    message = ChatMessage(role="user", content="Test message")
    count = memory._estimate_token_count(message)
    assert count == len(memory.tokenizer_fn("Test message"))


@pytest.mark.asyncio
async def test_estimate_token_count_image(memory):
    """Test token counting for images."""
    block = ImageBlock(url="http://example.com/image.jpg")
    message = ChatMessage(role="user", blocks=[block])
    count = memory._estimate_token_count(message)
    assert count == memory.image_token_size_estimate


@pytest.mark.asyncio
async def test_estimate_token_count_video(memory):
    """Test token counting for images."""
    block = VideoBlock(url="http://example.com/video.mp4")
    message = ChatMessage(role="user", blocks=[block])
    count = memory._estimate_token_count(message)
    assert count == memory.video_token_size_estimate


@pytest.mark.asyncio
async def test_estimate_token_count_audio(memory):
    """Test token counting for audio."""
    block = AudioBlock(url="http://example.com/audio.mp3")
    message = ChatMessage(role="user", blocks=[block])
    count = memory._estimate_token_count(message)
    assert count == memory.audio_token_size_estimate


@pytest.mark.asyncio
async def test_estimate_token_count_document(memory):
    """Test token counting for a document uses the fixed estimate."""
    block = DocumentBlock(
        url="http://example.com/doc.pdf", document_mimetype="application/pdf"
    )
    message = ChatMessage(role="user", blocks=[block])
    count = memory._estimate_token_count(message)
    assert count == memory.document_token_size_estimate


@pytest.mark.asyncio
async def test_manage_queue_under_limit(memory):
    """Test queue management when under token limit."""
    # Set up a case where we're under the token limit
    chat_messages = [ChatMessage(role="user", content="Short message")]

    await memory.aput_messages(chat_messages)
    cur_messages = await memory.aget()
    assert len(cur_messages) == 1
    assert cur_messages[0].content == "Short message"


@pytest.mark.asyncio
async def test_manage_queue_over_limit(memory):
    """Test queue management when over token limit."""
    # Set up a case where we're over the token limit
    chat_messages = [
        ChatMessage(role="user", content="x " * 500),
        ChatMessage(role="assistant", content="y " * 500),
        ChatMessage(role="user", content="z " * 500),
    ]

    # This will exceed the token limit and flush 700 tokens (two messages)
    await memory.aput_messages(chat_messages)

    cur_messages = await memory.aget()
    assert len(cur_messages) == 1
    assert "z " in cur_messages[0].content


@pytest.mark.asyncio
async def test_aput(memory):
    """Test adding a message."""
    message = ChatMessage(role="user", content="New message")

    await memory.aput(message)

    # Should add the message to the store
    messages = await memory.aget()
    assert len(messages) == 1
    assert messages[0].content == "New message"


@pytest.mark.asyncio
async def test_aput_messages(memory):
    """Test adding multiple messages."""
    messages = [
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Response 1"),
    ]

    await memory.aput_messages(messages)

    # Should add the messages to the store
    messages = await memory.aget()
    assert len(messages) == 2
    assert messages[0].content == "Message 1"
    assert messages[1].content == "Response 1"


@pytest.mark.asyncio
async def test_aset(memory):
    """Test setting the chat history."""
    messages = [
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Response 1"),
    ]

    await memory.aset(messages)

    # Should set the messages in the store
    messages = await memory.aget()
    assert len(messages) == 2
    assert messages[0].content == "Message 1"
    assert messages[1].content == "Response 1"


@pytest.mark.asyncio
async def test_aget_all(memory):
    """Test getting all messages."""
    await memory.aput_messages(
        [
            ChatMessage(role="user", content="Message 1"),
            ChatMessage(role="assistant", content="Response 1"),
        ]
    )
    messages = await memory.aget_all(status=MessageStatus.ACTIVE)

    # Should get all messages from the store
    assert len(messages) == 2
    assert messages[0].content == "Message 1"
    assert messages[1].content == "Response 1"


@pytest.mark.asyncio
async def test_areset(memory):
    """Test resetting the memory."""
    await memory.aput(ChatMessage(role="user", content="New message"))
    await memory.areset(status=MessageStatus.ACTIVE)

    # Should delete messages from the store
    messages = await memory.aget()
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_manage_queue_first_message_must_be_user():
    """
    Test that after flushing, the first message in queue is always a user message.

    This tests the edge case where token limits are set low enough that
    flushing could leave only an assistant message, which would break
    providers like Amazon Bedrock that require user message first.
    """
    # Create memory with very low limits to trigger the edge case
    # token_limit * chat_history_token_ratio = 100 * 0.5 = 50 tokens for chat history
    memory = Memory(
        token_limit=100,
        token_flush_size=50,
        chat_history_token_ratio=0.5,
        session_id="test_first_message_user",
    )

    # Create messages where user message is large and assistant message is small
    # This simulates a tool call scenario where the tool returns a lot of content
    # After flush, only the small assistant message might remain
    chat_messages = [
        ChatMessage(
            role="user", content="x " * 100
        ),  # Large user message (~100 tokens)
        ChatMessage(
            role="assistant", content="ok"
        ),  # Small assistant message (~1 token)
    ]

    await memory.aput_messages(chat_messages)

    cur_messages = await memory.aget()

    # The queue should not be empty
    assert len(cur_messages) > 0, "Queue should not be empty after flush"

    # The first message MUST be a user message
    assert cur_messages[0].role == "user", (
        f"First message must be 'user', but got '{cur_messages[0].role}'. "
        "This would break providers like Amazon Bedrock."
    )


@pytest.mark.asyncio
async def test_manage_queue_preserves_conversation_turn():
    """Test that flushing preserves at least one complete conversation turn."""
    memory = Memory(
        token_limit=200,
        token_flush_size=100,
        chat_history_token_ratio=0.5,
        session_id="test_preserve_turn",
    )

    # Multiple conversation turns
    chat_messages = [
        ChatMessage(role="user", content="a " * 50),
        ChatMessage(role="assistant", content="b " * 50),
        ChatMessage(role="user", content="c " * 50),
        ChatMessage(role="assistant", content="d " * 50),
    ]

    await memory.aput_messages(chat_messages)

    cur_messages = await memory.aget()

    # Should have at least one complete turn (user + assistant)
    assert len(cur_messages) >= 2, (
        "Should preserve at least one complete conversation turn"
    )

    # First message must be user
    assert cur_messages[0].role == "user"

    # Verify alternating pattern
    for i in range(len(cur_messages) - 1):
        if cur_messages[i].role == "user":
            assert cur_messages[i + 1].role in ("assistant", "tool")


@pytest.mark.asyncio
async def test_manage_queue_with_tool_messages():
    """
    Test that flushing correctly handles tool calling scenarios.

    In tool calling, the message sequence is:
    user → assistant (tool_call) → tool → assistant

    The recovery logic should keep the complete turn together.
    """
    memory = Memory(
        token_limit=150,
        token_flush_size=80,
        chat_history_token_ratio=0.5,
        session_id="test_tool_calling",
    )

    # Simulate a tool calling scenario
    chat_messages = [
        ChatMessage(role="user", content="a " * 40),  # ~40 tokens
        ChatMessage(role="assistant", content="b " * 20),  # ~20 tokens (with tool_call)
        ChatMessage(role="tool", content="c " * 20),  # ~20 tokens
        ChatMessage(role="assistant", content="d " * 20),  # ~20 tokens (final response)
    ]

    await memory.aput_messages(chat_messages)

    cur_messages = await memory.aget()

    # Should preserve at least the user message
    assert len(cur_messages) > 0, "Queue should not be empty"
    assert cur_messages[0].role == "user", "First message must be user"

    # If we have tool messages, they should be preceded by assistant
    for i, msg in enumerate(cur_messages):
        if msg.role == "tool":
            assert i > 0, "Tool message should not be first"
            # Tool messages should come after an assistant message
            assert cur_messages[i - 1].role in ("assistant", "tool"), (
                "Tool message should follow assistant or another tool"
            )


@pytest.mark.asyncio
async def test_manage_queue_only_tool_message_remaining():
    """
    Test edge case where only a tool message would remain after flush.

    This can happen with very low token limits. The recovery should
    find the preceding user message and keep the complete turn.
    """
    memory = Memory(
        token_limit=80,
        token_flush_size=40,
        chat_history_token_ratio=0.5,  # Effective limit: 40 tokens
        session_id="test_only_tool",
    )

    # Large user message, small tool response
    chat_messages = [
        ChatMessage(role="user", content="x " * 50),  # ~50 tokens
        ChatMessage(role="assistant", content="call"),  # ~1 token
        ChatMessage(role="tool", content="result"),  # ~1 token
    ]

    await memory.aput_messages(chat_messages)

    cur_messages = await memory.aget()

    # The queue should either:
    # 1. Have a complete turn starting with user, OR
    # 2. Be empty (if no recovery possible)
    if len(cur_messages) > 0:
        assert cur_messages[0].role == "user", (
            f"First message must be 'user', got '{cur_messages[0].role}'"
        )


class _RecordingBlock(BaseMemoryBlock[str]):
    """Test block that records every batch pushed from short-term memory."""

    received: List[List[str]] = Field(default_factory=list)
    fail_next: bool = False

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        return ""

    async def _aput(self, messages: List[ChatMessage]) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("transient block failure")
        self.received.append([m.content or "" for m in messages])


class _OrderCheckingBlock(BaseMemoryBlock[str]):
    """Test block that checks the flushed messages are still active when called."""

    sql_store: Any = None
    saw_active: List[bool] = Field(default_factory=list)

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        return ""

    async def _aput(self, messages: List[ChatMessage]) -> None:
        session_id = messages[0].additional_kwargs["session_id"]
        active = await self.sql_store.get_messages(
            session_id, status=MessageStatus.ACTIVE
        )
        active_contents = {m.content for m in active}
        self.saw_active.append(all(m.content in active_contents for m in messages))


@pytest.mark.asyncio
async def test_manage_queue_block_failure_keeps_messages_active() -> None:
    """A failing memory block must not cause flushed messages to be archived."""
    block = _RecordingBlock(name="rec", fail_next=True)
    memory = Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_block_failure",
        memory_blocks=[block],
    )
    chat_messages = [
        ChatMessage(role="user", content="x " * 500),
        ChatMessage(role="assistant", content="y " * 500),
        ChatMessage(role="user", content="z " * 500),
    ]

    with pytest.raises(RuntimeError, match="transient block failure"):
        await memory.aput_messages(chat_messages)

    # Nothing may be archived: the block never accepted the messages, so they
    # must stay active for a later flush to retry the delivery.
    active = await memory.aget_all(status=MessageStatus.ACTIVE)
    archived = await memory.aget_all(status=MessageStatus.ARCHIVED)
    assert {m.content for m in active} == {m.content for m in chat_messages}
    assert len(archived) == 0
    assert block.received == []


@pytest.mark.asyncio
async def test_manage_queue_blocks_run_before_archiving() -> None:
    """Memory blocks receive flushed messages while they are still active."""
    memory = Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_flush_ordering",
    )
    block = _OrderCheckingBlock(name="order", sql_store=memory.sql_store)
    memory.memory_blocks.append(block)

    await memory.aput_messages(
        [
            ChatMessage(role="user", content="x " * 500),
            ChatMessage(role="assistant", content="y " * 500),
            ChatMessage(role="user", content="z " * 500),
        ]
    )

    # Every batch the block received was still active at delivery time.
    assert block.saw_active
    assert all(block.saw_active)

    # After a successful flush the delivered messages are archived.
    archived = await memory.aget_all(status=MessageStatus.ARCHIVED)
    assert len(archived) > 0


@pytest.mark.asyncio
async def test_manage_queue_retries_block_delivery_after_failure() -> None:
    """After a block failure, the next flush redelivers the backlog (at-least-once)."""
    block = _RecordingBlock(name="rec_retry", fail_next=True)
    memory = Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_block_retry",
        memory_blocks=[block],
    )
    chat_messages = [
        ChatMessage(role="user", content="x " * 500),
        ChatMessage(role="assistant", content="y " * 500),
        ChatMessage(role="user", content="z " * 500),
    ]

    with pytest.raises(RuntimeError, match="transient block failure"):
        await memory.aput_messages(chat_messages)
    assert block.received == []

    # The block has recovered; adding another message triggers a new flush.
    await memory.aput(ChatMessage(role="assistant", content="w " * 500))

    # Everything that ends up archived must have been delivered to the block.
    archived = await memory.aget_all(status=MessageStatus.ARCHIVED)
    delivered = {content for batch in block.received for content in batch}
    assert len(archived) > 0
    assert all(m.content in delivered for m in archived)

    # And nothing was lost: every message is still either active or archived.
    active = await memory.aget_all(status=MessageStatus.ACTIVE)
    remaining = {m.content for m in active} | {m.content for m in archived}
    assert remaining == {m.content for m in chat_messages} | {"w " * 500}


@pytest.mark.asyncio
async def test_manage_queue_retries_only_failed_blocks() -> None:
    """A sibling failure must not redeliver an already acknowledged batch."""
    recorder = _RecordingBlock(name="recorder")
    retrying = _RecordingBlock(name="retrying", fail_next=True)
    memory = Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        memory_blocks=[recorder, retrying],
    )
    messages = [
        ChatMessage(role="user", content="x " * 500),
        ChatMessage(role="assistant", content="y " * 500),
        ChatMessage(role="user", content="z " * 500),
    ]
    batch = [message.content for message in messages[:2]]

    with pytest.raises(RuntimeError, match="transient block failure"):
        await memory.aput_messages(messages)
    assert recorder.received == [batch]
    assert retrying.received == []
    assert await memory.aget_all(status=MessageStatus.ARCHIVED) == []

    await memory._manage_queue()

    assert recorder.received == [batch]
    assert retrying.received == [batch]
    assert [
        message.content
        for message in await memory.aget_all(status=MessageStatus.ARCHIVED)
    ] == batch


@pytest.fixture()
def waterfall_messages() -> List[ChatMessage]:
    return [
        ChatMessage(role="user", content="x " * 500),
        ChatMessage(role="assistant", content="y " * 500),
        ChatMessage(role="user", content="z " * 500),
    ]


class _WaitingBlock(_RecordingBlock):
    started: asyncio.Event = Field(default_factory=asyncio.Event)
    release: asyncio.Event = Field(default_factory=asyncio.Event)
    cancelled: bool = False

    async def _aput(self, messages: List[ChatMessage]) -> None:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        await super()._aput(messages)


@pytest.mark.asyncio
async def test_manage_queue_waits_for_siblings_after_failure(
    memory: Memory, waterfall_messages: List[ChatMessage]
) -> None:
    retrying = _RecordingBlock(name="retrying", fail_next=True)
    waiting = _WaitingBlock(name="waiting")
    memory.memory_blocks = [retrying, waiting]
    flush = asyncio.create_task(memory.aput_messages(waterfall_messages))
    await asyncio.wait_for(waiting.started.wait(), timeout=5)
    assert not flush.done()
    waiting.release.set()
    with pytest.raises(RuntimeError, match="transient block failure"):
        await asyncio.wait_for(flush, timeout=5)

    await memory._manage_queue()
    batch = [message.content for message in waterfall_messages[:2]]
    assert waiting.received == [batch]
    assert retrying.received == [batch]


@pytest.mark.asyncio
async def test_manage_queue_serializes_overlapping_puts(
    memory: Memory, waterfall_messages: List[ChatMessage]
) -> None:
    waiting = _WaitingBlock(name="waiting")
    memory.memory_blocks = [waiting]
    first = asyncio.create_task(memory.aput_messages(waterfall_messages))
    await asyncio.wait_for(waiting.started.wait(), timeout=5)
    new_message = ChatMessage(role="assistant", content="w " * 500)
    second = asyncio.create_task(memory.aput(new_message))
    await asyncio.sleep(0)
    waiting.release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    assert waiting.received == [[message.content for message in waterfall_messages[:2]]]
    assert [
        message.content
        for message in await memory.aget_all(status=MessageStatus.ACTIVE)
    ] == [waterfall_messages[-1].content, new_message.content]


@pytest.mark.asyncio
async def test_manage_queue_cancellation_retains_acknowledgements(
    memory: Memory, waterfall_messages: List[ChatMessage]
) -> None:
    recorder = _RecordingBlock(name="recorder")
    waiting = _WaitingBlock(name="waiting")
    memory.memory_blocks = [recorder, waiting]
    flush = asyncio.create_task(memory.aput_messages(waterfall_messages))
    await asyncio.wait_for(waiting.started.wait(), timeout=5)
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    assert waiting.cancelled
    assert await memory.aget_all(status=MessageStatus.ARCHIVED) == []

    waiting.release.set()
    await memory._manage_queue()
    batch = [message.content for message in waterfall_messages[:2]]
    assert recorder.received == [batch]
    assert waiting.received == [batch]


@pytest.mark.asyncio
async def test_manage_queue_retries_original_batch_before_new_messages(
    memory: Memory, waterfall_messages: List[ChatMessage]
) -> None:
    recorder = _RecordingBlock(name="recorder")
    retrying = _RecordingBlock(name="retrying", fail_next=True)
    memory.memory_blocks = [recorder, retrying]
    with pytest.raises(RuntimeError, match="transient block failure"):
        await memory.aput_messages(waterfall_messages)

    # A pending batch must complete even if the new limit would not trigger a
    # flush, or the new flush size would select a different conversation boundary.
    memory.token_limit = 5000
    memory.token_flush_size = 3000
    new_message = ChatMessage(role="assistant", content="w " * 500)
    await memory.aput(new_message)
    batch = [message.content for message in waterfall_messages[:2]]
    assert recorder.received == [batch]
    assert retrying.received == [batch]
    assert [
        message.content
        for message in await memory.aget_all(status=MessageStatus.ACTIVE)
    ] == [waterfall_messages[-1].content, new_message.content]


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_manage_queue_recovers_archive_failure_without_redelivery(
    memory: Memory,
    waterfall_messages: List[ChatMessage],
    monkeypatch: pytest.MonkeyPatch,
    committed: bool,
    cancelled: bool,
) -> None:
    recorder = _RecordingBlock(name="recorder")
    memory.memory_blocks = [recorder]
    archive = type(memory.sql_store).archive_oldest_messages
    calls = 0
    started = asyncio.Event()

    async def fail_once(store: Any, key: str, n: int) -> List[ChatMessage]:
        nonlocal calls
        calls += 1
        if calls > 1:
            return await archive(store, key, n)
        if committed:
            await archive(store, key, n)
        started.set()
        if cancelled:
            await asyncio.Event().wait()
        raise RuntimeError("archive failure")

    monkeypatch.setattr(type(memory.sql_store), "archive_oldest_messages", fail_once)
    flush = asyncio.create_task(memory.aput_messages(waterfall_messages))
    await asyncio.wait_for(started.wait(), timeout=5)
    if cancelled:
        flush.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flush
    else:
        with pytest.raises(RuntimeError, match="archive failure"):
            await flush

    memory.token_limit = 5000
    # The same content in later messages must not be mistaken for the batch
    # whose archive committed before the store reported a failure.
    await memory.aput_messages(waterfall_messages[:2])
    batch = [message.content for message in waterfall_messages[:2]]
    assert recorder.received == [batch]
    assert calls == (1 if committed else 2)
    assert [
        message.content
        for message in await memory.aget_all(status=MessageStatus.ARCHIVED)
    ] == batch
    assert [
        message.content
        for message in await memory.aget_all(status=MessageStatus.ACTIVE)
    ] == [waterfall_messages[-1].content, *batch]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["set", "reset", "reset_active"])
async def test_manage_queue_discards_pending_batch_when_history_is_replaced(
    memory: Memory, waterfall_messages: List[ChatMessage], operation: str
) -> None:
    recorder = _RecordingBlock(name="recorder")
    retrying = _RecordingBlock(name="retrying", fail_next=True)
    memory.memory_blocks = [recorder, retrying]
    with pytest.raises(RuntimeError, match="transient block failure"):
        await memory.aput_messages(waterfall_messages)

    if operation == "set":
        await memory.aset(waterfall_messages)
    else:
        await memory.areset(
            status=MessageStatus.ACTIVE if operation == "reset_active" else None
        )
        await memory.aput_messages(waterfall_messages)
    await memory._manage_queue()
    batch = [message.content for message in waterfall_messages[:2]]
    assert recorder.received == [batch, batch]
    assert retrying.received == [batch]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["set", "reset"])
@pytest.mark.parametrize("committed", [False, True])
async def test_manage_queue_requires_failed_history_replacement_to_be_retried(
    memory: Memory,
    waterfall_messages: List[ChatMessage],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    committed: bool,
) -> None:
    recorder = _RecordingBlock(name="recorder")
    retrying = _RecordingBlock(name="retrying", fail_next=True)
    memory.memory_blocks = [recorder, retrying]
    with pytest.raises(RuntimeError, match="transient block failure"):
        await memory.aput_messages(waterfall_messages)

    method = "set_messages" if operation == "set" else "delete_messages"
    original = getattr(type(memory.sql_store), method)
    calls = 0

    async def fail_once(store: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls != 1 or committed:
            await original(store, *args, **kwargs)
        if calls == 1:
            raise RuntimeError("history replacement failure")

    monkeypatch.setattr(type(memory.sql_store), method, fail_once)

    async def replace_history() -> None:
        if operation == "set":
            await memory.aset(waterfall_messages)
        else:
            await memory.areset(status=MessageStatus.ACTIVE)

    with pytest.raises(RuntimeError, match="history replacement failure"):
        await replace_history()
    with pytest.raises(RuntimeError, match="Chat history replacement did not complete"):
        await memory._manage_queue()
    active = await memory.aget_all(status=MessageStatus.ACTIVE)
    with pytest.raises(RuntimeError, match="Chat history replacement did not complete"):
        await memory.aput(ChatMessage(role="user", content="must not be added"))
    assert await memory.aget_all(status=MessageStatus.ACTIVE) == active
    assert len(recorder.received) == 1
    assert retrying.received == []

    await replace_history()
    if operation == "reset":
        await memory.aput_messages(waterfall_messages)
    await memory._manage_queue()
    batch = [message.content for message in waterfall_messages[:2]]
    assert recorder.received == [batch, batch]
    assert retrying.received == [batch]


@pytest.mark.asyncio
@pytest.mark.parametrize("archive_committed", [False, True])
@pytest.mark.parametrize("delete_committed", [False, True])
async def test_manage_queue_preserves_pending_delivery_when_resetting_archives_fails(
    memory: Memory,
    waterfall_messages: List[ChatMessage],
    monkeypatch: pytest.MonkeyPatch,
    archive_committed: bool,
    delete_committed: bool,
) -> None:
    recorder = _RecordingBlock(name="recorder")
    memory.memory_blocks = [recorder]
    # An existing archive makes deleting archived history change the baseline.
    await memory.sql_store.add_message(
        memory.session_id,
        ChatMessage(role="user", content="previous archive"),
        status=MessageStatus.ARCHIVED,
    )
    archive = type(memory.sql_store).archive_oldest_messages
    delete = type(memory.sql_store).delete_messages
    archive_calls = 0

    async def fail_archive_once(store: Any, key: str, n: int) -> List[ChatMessage]:
        nonlocal archive_calls
        archive_calls += 1
        if archive_calls > 1:
            return await archive(store, key, n)
        if archive_committed:
            await archive(store, key, n)
        raise RuntimeError("archive failure")

    async def fail_delete(
        store: Any, key: str, status: Optional[MessageStatus] = None
    ) -> None:
        if delete_committed:
            await delete(store, key, status=status)
        raise RuntimeError("delete failure")

    monkeypatch.setattr(
        type(memory.sql_store), "archive_oldest_messages", fail_archive_once
    )
    with pytest.raises(RuntimeError, match="archive failure"):
        await memory.aput_messages(waterfall_messages)
    monkeypatch.setattr(type(memory.sql_store), "delete_messages", fail_delete)
    with pytest.raises(RuntimeError, match="delete failure"):
        await memory.areset(status=MessageStatus.ARCHIVED)

    await memory._manage_queue()
    batch = [message.content for message in waterfall_messages[:2]]
    assert recorder.received == [batch]
    assert archive_calls == (1 if archive_committed else 2)
    assert [
        message.content
        for message in await memory.aget_all(status=MessageStatus.ACTIVE)
    ] == [waterfall_messages[-1].content]


@pytest.mark.asyncio
async def test_manage_queue_delivers_later_identical_messages_as_a_new_batch(
    memory: Memory, waterfall_messages: List[ChatMessage]
) -> None:
    recorder = _RecordingBlock(name="recorder")
    retrying = _RecordingBlock(name="retrying", fail_next=True)
    memory.memory_blocks = [recorder, retrying]
    with pytest.raises(RuntimeError, match="transient block failure"):
        await memory.aput_messages(waterfall_messages)

    next_response = ChatMessage(role="assistant", content="w " * 500)
    await memory.aput_messages([next_response, *waterfall_messages])
    batch = [message.content for message in waterfall_messages[:2]]
    expected = [batch, [waterfall_messages[-1].content, next_response.content], batch]
    assert recorder.received == expected
    assert retrying.received == expected
    assert [
        message.content
        for message in await memory.aget_all(status=MessageStatus.ARCHIVED)
    ] == [content for delivered in expected for content in delivered]
