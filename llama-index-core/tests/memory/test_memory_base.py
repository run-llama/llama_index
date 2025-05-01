import pytest
from typing import List, Any, Optional

from llama_index.core.base.llms.types import ChatMessage, TextBlock, ImageBlock, AudioBlock
from llama_index.core.memory.memory import Memory, BaseMemoryBlock
from llama_index.core.storage.chat_store.sql import MessageStatus


class TestMemoryBlock(BaseMemoryBlock[str]):
    """Simple memory block for testing."""

    get_result: str = "Test memory content"
    put_called: bool = False

    async def _aget(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        return self.get_result

    async def _aput(self, messages: List[ChatMessage]) -> None:
        self.put_called = True


class TruncatableMemoryBlock(BaseMemoryBlock[str]):
    """Memory block that supports truncation for testing."""

    get_result: str = "Some truncatable content that can be reduced"
    truncate_return: Optional[str] = "Truncated content"
    put_called: bool = False

    async def _aget(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        return self.get_result

    async def _aput(self, messages: List[ChatMessage]) -> None:
        self.put_called = True

    async def atruncate(self, content: str, tokens_to_truncate: int) -> Optional[str]:
        return self.truncate_return


class ChatMessagesMemoryBlock(BaseMemoryBlock[List[ChatMessage]]):
    """Memory block that returns chat messages."""

    get_result: List[ChatMessage] = [
        ChatMessage(role="user", content="Test user message from memory block")
    ]
    put_called: bool = False

    async def _aget(self, messages: List[ChatMessage], **kwargs: Any) -> List[ChatMessage]:
        return self.get_result

    async def _aput(self, messages: List[ChatMessage]) -> None:
        self.put_called = True



@pytest.fixture()
def basic_memory():
    """Create a basic memory instance for testing."""
    return Memory(
        token_limit=1000,
        token_flush_size=700,
        user_id="test_user",
    )

@pytest.fixture()
def memory_with_blocks():
    """Create a memory instance with memory blocks."""
    return Memory(
        token_limit=1000,
        token_flush_size=700,
        user_id="test_user",
        memory_blocks=[
            TestMemoryBlock(name="test_block", priority=0),
            TruncatableMemoryBlock(name="truncatable_block", priority=1),
            ChatMessagesMemoryBlock(name="chat_messages_block", priority=2),
        ],
    )

@pytest.mark.asyncio
async def test_initialization(basic_memory):
    """Test that memory initializes correctly."""
    assert basic_memory.token_limit == 1000
    assert basic_memory.token_flush_size == 700
    assert basic_memory.user_id == "test_user"

@pytest.mark.asyncio
async def test_estimate_token_count_text(basic_memory):
    """Test token counting for text."""
    message = ChatMessage(role="user", content="Test message")
    count = basic_memory._estimate_token_count(message)
    assert count == len(basic_memory.tokenizer_fn("Test message"))

@pytest.mark.asyncio
async def test_estimate_token_count_image(basic_memory):
    """Test token counting for images."""
    block = ImageBlock(url="http://example.com/image.jpg")
    message = ChatMessage(role="user", blocks=[block])
    count = basic_memory._estimate_token_count(message)
    assert count == basic_memory.image_token_size_estimate

@pytest.mark.asyncio
async def test_estimate_token_count_audio(basic_memory):
    """Test token counting for audio."""
    block = AudioBlock(url="http://example.com/audio.mp3")
    message = ChatMessage(role="user", blocks=[block])
    count = basic_memory._estimate_token_count(message)
    assert count == basic_memory.audio_token_size_estimate

@pytest.mark.asyncio
async def test_get_memory_blocks_content(memory_with_blocks):
    """Test getting content from memory blocks."""
    chat_history = [
        ChatMessage(role="user", content="Test message for blocks")
    ]

    content = await memory_with_blocks._get_memory_blocks_content(chat_history)

    assert "test_block" in content
    assert content["test_block"] == "Test memory content"
    assert "truncatable_block" in content
    assert "chat_messages_block" in content

@pytest.mark.asyncio
async def test_truncate_memory_blocks(memory_with_blocks):
    """Test truncation of memory blocks when token limit is exceeded."""
    content_per_block = {
        "test_block": "Test memory content",  # Priority 0 - never truncate
        "truncatable_block": "Some truncatable content that can be reduced",  # Priority 1
        "chat_messages_block": [ChatMessage(role="user", content="Test user message from memory block")]  # Priority 2
    }

    memory_blocks_tokens = 500  # Simulate token count
    chat_history_tokens = 600  # Simulate token count

    # Total (1100) exceeds token limit (1000), so we need to truncate 100 tokens
    truncated = await memory_with_blocks._truncate_memory_blocks(
        content_per_block, memory_blocks_tokens, chat_history_tokens
    )

    # Priority 0 block should never be truncated
    assert "test_block" in truncated

    # All other blocks should be truncated
    assert "truncatable_block" not in truncated
    assert "chat_messages_block" not in truncated

@pytest.mark.asyncio
async def test_format_memory_blocks(memory_with_blocks):
    """Test formatting memory blocks into template data and chat messages."""
    content_per_block = {
        "test_block": "Test memory content",
        "chat_messages_block": [ChatMessage(role="user", content="Test user message from memory block")]
    }

    formatted_blocks, chat_messages = await memory_with_blocks._format_memory_blocks(content_per_block)

    # Text content should be formatted as text blocks
    assert len(formatted_blocks) == 1
    block_name, blocks = formatted_blocks[0]
    assert block_name == "test_block"
    assert len(blocks) == 1
    assert blocks[0].text == "Test memory content"

    # Chat messages should be returned separately
    assert len(chat_messages) == 1
    assert chat_messages[0].role == "user"
    assert chat_messages[0].content == "Test user message from memory block"

@pytest.mark.asyncio
async def test_insert_memory_content(memory_with_blocks):
    """Test inserting memory content into chat history."""
    chat_history = [
        ChatMessage(role="user", content="Original user message"),
        ChatMessage(role="assistant", content="Original assistant response"),
    ]

    blocks = [TextBlock(text="User content1"), TextBlock(text="User content2")]
    chat_message_data = [ChatMessage(role="user", content="Memory block user message")]

    result = memory_with_blocks._insert_memory_content(
        chat_history, blocks, chat_message_data
    )

    # Should have all original messages plus new ones
    assert len(result) == 4  # 2 original + 1 user content + 1 chat message data

    # Our content should be inserted into the system message
    assert result[0].role == "system"
    assert any(block.text == "User content1" for block in result[0].blocks)
    assert any(block.text == "User content2" for block in result[0].blocks)

@pytest.mark.asyncio
async def test_aget(memory_with_blocks):
    """Test getting messages with memory blocks included."""
    result = await memory_with_blocks.aget()

    # Should have processed messages with memory blocks
    assert len(result) > 0

    # First message should have memory block content
    first_msg = result[0]
    assert first_msg is not None


@pytest.mark.asyncio
async def test_manage_queue_under_limit(basic_memory):
    """Test queue management when under token limit."""
    # Set up a case where we're under the token limit
    chat_messages = [
        ChatMessage(role="user", content="Short message")
    ]

    await basic_memory.aput_messages(chat_messages)
    cur_messages = await basic_memory.aget()
    assert len(cur_messages) == 1
    assert cur_messages[0].content == "Short message"

@pytest.mark.asyncio
async def test_manage_queue_over_limit(memory_with_blocks):
    """Test queue management when over token limit."""
    # Create a long message history that exceeds token limit
    # This will cause manage_queue to run and manage the FIFO queue
    long_messages = [
        ChatMessage(role="user", content="x " * 800),
        ChatMessage(role="assistant", content="y " * 800),
        ChatMessage(role="user", content="z " * 800),
    ]

    await memory_with_blocks.aput_messages(long_messages)


    messages = await memory_with_blocks.aget()
    assert len(messages) == 3

@pytest.mark.asyncio
async def test_aput(basic_memory):
    """Test adding a message."""
    message = ChatMessage(role="user", content="New message")

    await basic_memory.aput(message)

    # Should add the message to the store
    messages = await basic_memory.aget()
    assert len(messages) == 1
    assert messages[0].content == "New message"

@pytest.mark.asyncio
async def test_aput_messages(basic_memory):
    """Test adding multiple messages."""
    messages = [
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Response 1"),
    ]

    await basic_memory.aput_messages(messages)

    # Should add the messages to the store
    messages = await basic_memory.aget()
    assert len(messages) == 2
    assert messages[0].content == "Message 1"
    assert messages[1].content == "Response 1"

@pytest.mark.asyncio
async def test_aset(basic_memory):
    """Test setting the chat history."""
    messages = [
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Response 1"),
    ]

    await basic_memory.aset(messages)

    # Should set the messages in the store
    messages = await basic_memory.aget()
    assert len(messages) == 2
    assert messages[0].content == "Message 1"
    assert messages[1].content == "Response 1"

@pytest.mark.asyncio
async def test_aget_all(basic_memory):
    """Test getting all messages."""
    await basic_memory.aput_messages([
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Response 1"),
    ])
    messages = await basic_memory.aget_all(status=MessageStatus.ACTIVE)

    # Should get all messages from the store
    assert len(messages) == 2
    assert messages[0].content == "Message 1"
    assert messages[1].content == "Response 1"

@pytest.mark.asyncio
async def test_areset(basic_memory):
    """Test resetting the memory."""
    await basic_memory.aput(ChatMessage(role="user", content="New message"))
    await basic_memory.areset(status=MessageStatus.ACTIVE)

    # Should delete messages from the store
    messages = await basic_memory.aget()
    assert len(messages) == 0
