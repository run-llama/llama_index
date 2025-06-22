import pytest
from typing import List, Any, Optional, Union

from llama_index.core.base.llms.types import ChatMessage, TextBlock, ContentBlock
from llama_index.core.memory.memory import Memory, BaseMemoryBlock, InsertMethod


class TextMemoryBlock(BaseMemoryBlock[str]):
    """Memory block that returns text content."""

    async def _aget(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        return "Simple text content from TextMemoryBlock"

    async def _aput(self, messages: List[ChatMessage]) -> None:
        # Just a no-op for testing
        pass


class ContentBlocksMemoryBlock(BaseMemoryBlock[List[ContentBlock]]):
    """Memory block that returns content blocks."""

    async def _aget(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> List[ContentBlock]:
        return [
            TextBlock(text="Text block 1"),
            TextBlock(text="Text block 2"),
        ]

    async def _aput(self, messages: List[ChatMessage]) -> None:
        # Just a no-op for testing
        pass

    async def atruncate(
        self, content: List[ContentBlock], tokens_to_truncate: int
    ) -> Optional[List[ContentBlock]]:
        # Simple truncation - remove last block
        if not content:
            return None
        return content[:-1]


class ChatMessagesMemoryBlock(BaseMemoryBlock[List[ChatMessage]]):
    """Memory block that returns chat messages."""

    async def _aget(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> List[ChatMessage]:
        return [
            ChatMessage(role="user", content="Historical user message"),
            ChatMessage(role="assistant", content="Historical assistant response"),
        ]

    async def _aput(self, messages: List[ChatMessage]) -> None:
        # Just a no-op for testing
        pass


class ComplexMemoryBlock(BaseMemoryBlock[Union[str, List[ContentBlock]]]):
    """Memory block that can return different types based on input."""

    mode: str = "text"  # Can be "text" or "blocks"

    async def _aget(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> Union[str, List[ContentBlock]]:
        if self.mode == "text":
            return "Text content from ComplexMemoryBlock"
        else:
            return [
                TextBlock(text="Complex block 1"),
                TextBlock(text="Complex block 2"),
            ]

    async def _aput(self, messages: List[ChatMessage]) -> None:
        # Just a no-op for testing
        pass


class ParameterizedMemoryBlock(BaseMemoryBlock[str]):
    """Memory block that uses parameters passed to aget."""

    async def _aget(self, messages: List[ChatMessage], **kwargs: Any) -> str:
        # Use parameters passed to aget
        parameter = kwargs.get("test_parameter", "default")
        return f"Parameter value: {parameter}"

    async def _aput(self, messages: List[ChatMessage]) -> None:
        # Just a no-op for testing
        pass


@pytest.fixture()
def memory_with_blocks():
    """Set up memory with different types of memory blocks."""
    return Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_blocks",
        memory_blocks=[
            TextMemoryBlock(name="text_block", priority=1),
            ContentBlocksMemoryBlock(name="content_blocks", priority=2),
            ChatMessagesMemoryBlock(name="chat_messages", priority=3),
            ComplexMemoryBlock(name="complex_block", priority=4),
            ParameterizedMemoryBlock(name="param_block", priority=5),
        ],
    )


@pytest.mark.asyncio
async def test_text_memory_block(memory_with_blocks):
    """Test text memory block integration."""
    # Get the memory block content
    content = await memory_with_blocks._get_memory_blocks_content([])

    # Verify text block content
    assert "text_block" in content
    assert content["text_block"] == "Simple text content from TextMemoryBlock"

    # Format content for insertion
    formatted_blocks, _ = await memory_with_blocks._format_memory_blocks(
        {"text_block": content["text_block"]}
    )

    # Check formatting
    assert len(formatted_blocks) == 1
    block_name, content = formatted_blocks[0]

    assert block_name == "text_block"
    assert len(content) == 1
    assert content[0].text == "Simple text content from TextMemoryBlock"


@pytest.mark.asyncio
async def test_content_blocks_memory_block(memory_with_blocks):
    """Test content blocks memory block integration."""
    # Get the memory block content
    content = await memory_with_blocks._get_memory_blocks_content([])

    # Verify content blocks block content
    assert "content_blocks" in content
    assert len(content["content_blocks"]) == 2
    assert content["content_blocks"][0].text == "Text block 1"
    assert content["content_blocks"][1].text == "Text block 2"

    # Format content for insertion
    formatted_blocks, _ = await memory_with_blocks._format_memory_blocks(
        {"content_blocks": content["content_blocks"]}
    )

    # Check formatting
    assert len(formatted_blocks) == 1
    block_name, content = formatted_blocks[0]

    assert block_name == "content_blocks"
    assert len(content) == 2
    assert content[0].text == "Text block 1"
    assert content[1].text == "Text block 2"


@pytest.mark.asyncio
async def test_chat_messages_memory_block(memory_with_blocks):
    """Test chat messages memory block integration."""
    # Get the memory block content
    content = await memory_with_blocks._get_memory_blocks_content([])

    # Verify chat messages block content
    assert "chat_messages" in content
    assert len(content["chat_messages"]) == 2
    assert content["chat_messages"][0].role == "user"
    assert content["chat_messages"][0].content == "Historical user message"
    assert content["chat_messages"][1].role == "assistant"
    assert content["chat_messages"][1].content == "Historical assistant response"

    # Format content for insertion
    formatted_blocks, chat_messages = await memory_with_blocks._format_memory_blocks(
        {"chat_messages": content["chat_messages"]}
    )

    # Chat messages should be returned directly
    assert len(chat_messages) == 2
    assert chat_messages[0].role == "user"
    assert chat_messages[0].content == "Historical user message"
    assert chat_messages[1].role == "assistant"
    assert chat_messages[1].content == "Historical assistant response"


@pytest.mark.asyncio
async def test_complex_memory_block_text_mode(memory_with_blocks):
    """Test complex memory block in text mode."""
    # Set complex block to text mode
    for block in memory_with_blocks.memory_blocks:
        if isinstance(block, ComplexMemoryBlock):
            block.mode = "text"
            break

    # Get the memory block content
    content = await memory_with_blocks._get_memory_blocks_content([])

    # Verify complex block content in text mode
    assert "complex_block" in content
    assert content["complex_block"] == "Text content from ComplexMemoryBlock"


@pytest.mark.asyncio
async def test_complex_memory_block_blocks_mode(memory_with_blocks):
    """Test complex memory block in blocks mode."""
    # Set complex block to blocks mode
    for block in memory_with_blocks.memory_blocks:
        if isinstance(block, ComplexMemoryBlock):
            block.mode = "blocks"
            break

    # Get the memory block content
    content = await memory_with_blocks._get_memory_blocks_content([])

    # Verify complex block content in blocks mode
    assert "complex_block" in content
    assert len(content["complex_block"]) == 2
    assert content["complex_block"][0].text == "Complex block 1"
    assert content["complex_block"][1].text == "Complex block 2"


@pytest.mark.asyncio
async def test_parameterized_memory_block(memory_with_blocks):
    """Test memory block that accepts parameters."""
    # Get memory block content with parameter
    content = await memory_with_blocks._get_memory_blocks_content(
        [], test_parameter="custom_value"
    )

    # Verify parameter was passed through
    assert "param_block" in content
    assert content["param_block"] == "Parameter value: custom_value"

    # Try with default parameter
    content = await memory_with_blocks._get_memory_blocks_content([])
    assert content["param_block"] == "Parameter value: default"


@pytest.mark.asyncio
async def test_truncation_of_content_blocks(memory_with_blocks):
    """Test truncation of content blocks."""
    # Get memory blocks content
    content = await memory_with_blocks._get_memory_blocks_content([])
    content_blocks = content["content_blocks"]

    # Get the memory block for truncation
    content_block = next(
        (
            block
            for block in memory_with_blocks.memory_blocks
            if isinstance(block, ContentBlocksMemoryBlock)
        ),
        None,
    )
    assert content_block is not None

    # Test truncation
    truncated = await content_block.atruncate(content_blocks, 100)
    assert len(truncated) == 1  # Should have truncated to one block
    assert truncated[0].text == "Text block 1"


@pytest.mark.asyncio
async def test_memory_with_all_block_types(memory_with_blocks):
    """Test getting all memory block types together."""
    # Set up block content
    chat_history = [ChatMessage(role="user", content="Current message")]

    # Get final messages with memory blocks included
    messages = await memory_with_blocks.aget()

    # Should have properly inserted all memory content
    assert len(messages) > 0

    # Should have a system message with memory blocks
    system_messages = [msg for msg in messages if msg.role == "system"]
    assert len(system_messages) > 0

    # The system message should contain content from our blocks
    system_content = system_messages[0].blocks[0].text
    assert "Simple text content from TextMemoryBlock" in system_content
    assert "Text block 1" in system_content
    assert "Text block 2" in system_content

    # Should also include direct chat messages from ChatMessagesMemoryBlock
    user_historical = [
        msg for msg in messages if msg.content == "Historical user message"
    ]
    assert len(user_historical) > 0


@pytest.mark.asyncio
async def test_insert_method_setting():
    """Test that insert_method is respected for blocks."""
    # Create blocks with different insert methods
    system_block = TextMemoryBlock(
        name="system_block",
        priority=1,
    )
    user_block = TextMemoryBlock(
        name="user_block",
        priority=2,
    )

    # Create memory with user insert method
    memory = Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_insert_methods",
        insert_method=InsertMethod.USER,
        memory_blocks=[system_block, user_block],
    )

    # Insert a user message
    await memory.aput(ChatMessage(role="user", content="Test message!"))

    # Get messages
    messages = await memory.aget()

    # Should have both system and user messages with appropriate content
    system_msgs = [msg for msg in messages if msg.role == "system"]
    user_msgs = [msg for msg in messages if msg.role == "user"]

    assert len(user_msgs) > 0
    assert len(system_msgs) == 0

    # Create memory with system insert method
    memory = Memory(
        token_limit=1000,
        token_flush_size=700,
        chat_history_token_ratio=0.9,
        session_id="test_insert_methods",
        insert_method=InsertMethod.SYSTEM,
        memory_blocks=[system_block, user_block],
    )

    # Get messages
    messages = await memory.aget()

    # Should have both system and user messages with appropriate content
    system_msgs = [msg for msg in messages if msg.role == "system"]
    user_msgs = [msg for msg in messages if msg.role == "user"]

    assert len(user_msgs) == 0
    assert len(system_msgs) > 0
