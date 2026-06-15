import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    TextBlock,
    ThinkingBlock,
)
from llama_index.core.chat_engine.types import (
    StreamingAgentChatResponse,
    set_streamed_message_text,
)
from llama_index.core.memory import ChatMemoryBuffer


def test_set_streamed_message_text_single_text_block():
    message = ChatMessage(role="assistant", blocks=[TextBlock(text="old")])
    set_streamed_message_text(message, "new")
    assert message.content == "new"
    assert len(message.blocks) == 1


def test_set_streamed_message_text_no_blocks():
    message = ChatMessage(role="assistant", blocks=[])
    set_streamed_message_text(message, "new")
    assert message.content == "new"


def test_set_streamed_message_text_preserves_non_text_blocks():
    """A thinking block alongside text must survive and keep its position."""
    message = ChatMessage(
        role="assistant",
        blocks=[ThinkingBlock(content="reasoning"), TextBlock(text="old")],
    )
    set_streamed_message_text(message, "new")
    assert message.content == "new"
    assert [type(b) for b in message.blocks] == [ThinkingBlock, TextBlock]
    assert message.blocks[0].content == "reasoning"


def test_set_streamed_message_text_consolidates_multiple_text_blocks():
    message = ChatMessage(
        role="assistant",
        blocks=[TextBlock(text="a"), TextBlock(text="b")],
    )
    set_streamed_message_text(message, "final")
    assert message.content == "final"
    assert sum(isinstance(b, TextBlock) for b in message.blocks) == 1


def test_set_streamed_message_text_appends_when_no_text_block():
    message = ChatMessage(role="assistant", blocks=[ThinkingBlock(content="r")])
    set_streamed_message_text(message, "answer")
    assert message.content == "answer"
    assert [type(b) for b in message.blocks] == [ThinkingBlock, TextBlock]


def test_write_response_to_history_handles_multi_block_message():
    """Regression for #21679: writing a multi-block message must not raise."""
    final_message = ChatMessage(
        role="assistant",
        blocks=[ThinkingBlock(content="reasoning"), TextBlock(text="hello")],
    )

    def gen():
        yield ChatResponse(message=final_message, delta="hello")

    response = StreamingAgentChatResponse(chat_stream=gen())
    memory = ChatMemoryBuffer.from_defaults()

    response.write_response_to_history(memory)

    history = memory.get_all()
    assert len(history) == 1
    assert history[0].content == "hello"
    assert any(isinstance(b, ThinkingBlock) for b in history[0].blocks)


@pytest.mark.asyncio
async def test_awrite_response_to_history_handles_multi_block_message():
    """Regression for #21679 on the async path used by astream_chat."""
    final_message = ChatMessage(
        role="assistant",
        blocks=[ThinkingBlock(content="reasoning"), TextBlock(text="hello there")],
    )

    async def agen():
        yield ChatResponse(message=final_message, delta="hello there")

    response = StreamingAgentChatResponse(achat_stream=agen())
    memory = ChatMemoryBuffer.from_defaults()

    await response.awrite_response_to_history(memory)

    history = memory.get_all()
    assert len(history) == 1
    assert history[0].content == "hello there"
    assert any(isinstance(b, ThinkingBlock) for b in history[0].blocks)
