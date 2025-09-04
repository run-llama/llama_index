import pytest

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, AudioBlock
from llama_index.core.memory.memory import Memory
from llama_index.core.storage.chat_store.sql import MessageStatus


@pytest.fixture()
def memory():
    """Create a basic memory instance for testing."""
    return Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_user",
    )


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
async def test_estimate_token_count_audio(memory):
    """Test token counting for audio."""
    block = AudioBlock(url="http://example.com/audio.mp3")
    message = ChatMessage(role="user", blocks=[block])
    count = memory._estimate_token_count(message)
    assert count == memory.audio_token_size_estimate


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
