import pytest
from typing import List

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ContentBlock
from llama_index.core.memory.memory_blocks.static import StaticMemoryBlock


@pytest.fixture
def sample_messages():
    """Create sample chat messages."""
    return [
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
        ChatMessage(role=MessageRole.ASSISTANT, content="I'm doing well, thanks for asking!"),
        ChatMessage(role=MessageRole.USER, content="What's the weather like today?"),
    ]


@pytest.mark.asyncio
async def test_initialization_with_string():
    """Test initialization of StaticMemoryBlock with a string."""
    static_text = "This is some static content"
    memory_block = StaticMemoryBlock(static_content=static_text)

    assert memory_block.name == "StaticContent"
    assert len(memory_block.static_content) == 1
    assert isinstance(memory_block.static_content[0], TextBlock)
    assert memory_block.static_content[0].text == static_text


@pytest.mark.asyncio
async def test_initialization_with_content_blocks():
    """Test initialization of StaticMemoryBlock with a list of ContentBlock."""
    content_blocks = [
        TextBlock(text="First block"),
        TextBlock(text="Second block")
    ]
    memory_block = StaticMemoryBlock(static_content=content_blocks, name="CustomName")

    assert memory_block.name == "CustomName"
    assert len(memory_block.static_content) == 2
    assert memory_block.static_content[0].text == "First block"
    assert memory_block.static_content[1].text == "Second block"


@pytest.mark.asyncio
async def test_aget_returns_static_content():
    """Test that aget method returns the static content."""
    static_text = "This is static content for testing"
    memory_block = StaticMemoryBlock(static_content=static_text)

    result = await memory_block.aget()

    assert len(result) == 1
    assert result[0].text == static_text


@pytest.mark.asyncio
async def test_aget_ignores_messages(sample_messages):
    """Test that aget method returns the same content regardless of messages."""
    static_text = "Fixed content that doesn't change"
    memory_block = StaticMemoryBlock(static_content=static_text)

    result_with_messages = await memory_block.aget(messages=sample_messages)
    result_without_messages = await memory_block.aget()

    assert result_with_messages == result_without_messages
    assert result_with_messages[0].text == static_text


@pytest.mark.asyncio
async def test_aput_does_nothing(sample_messages):
    """Test that aput method is a no-op and doesn't change the static content."""
    static_text = "Unchanging content"
    memory_block = StaticMemoryBlock(static_content=static_text)

    # Get the content before calling aput
    content_before = await memory_block.aget()

    # Call aput with sample messages
    await memory_block.aput(sample_messages)

    # Get the content after calling aput
    content_after = await memory_block.aget()

    # Verify content hasn't changed
    assert content_before == content_after
    assert content_before[0].text == static_text
