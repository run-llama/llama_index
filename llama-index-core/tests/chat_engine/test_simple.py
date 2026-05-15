import gc
import asyncio
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    MessageRole,
    TextBlock,
)
from typing import Any, AsyncGenerator
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.mock import MockLLM
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.core.chat_engine.types import StreamingAgentChatResponse


def test_simple_chat_engine() -> None:
    engine = SimpleChatEngine.from_defaults()

    engine.reset()
    response = engine.chat("Test message 1")
    assert str(response) == "user: Test message 1\nassistant: "

    response = engine.chat("Test message 2")
    assert (
        str(response)
        == "user: Test message 1\nassistant: user: Test message 1\nassistant: \n"
        "user: Test message 2\nassistant: "
    )

    engine.reset()
    response = engine.chat("Test message 3")
    assert str(response) == "user: Test message 3\nassistant: "


def test_simple_chat_engine_with_init_history() -> None:
    engine = SimpleChatEngine.from_defaults(
        chat_history=[
            ChatMessage(role=MessageRole.USER, content="test human message"),
            ChatMessage(role=MessageRole.ASSISTANT, content="test ai message"),
        ],
    )

    response = engine.chat("new human message")
    assert (
        str(response) == "user: test human message\nassistant: test ai message\n"
        "user: new human message\nassistant: "
    )


@pytest.mark.asyncio
async def test_simple_chat_engine_astream():
    engine = SimpleChatEngine.from_defaults()
    response = await engine.astream_chat("Hello World!")

    num_iters = 0
    async for response_part in response.async_response_gen():
        num_iters += 1

    assert num_iters > 10

    assert "Hello World!" in response.unformatted_response
    assert len(engine.chat_history) == 2

    response = await engine.astream_chat("What is the capital of the moon?")

    num_iters = 0
    async for _ in response.async_response_gen():
        num_iters += 1

    assert num_iters > 10
    assert "Hello World!" in response.unformatted_response
    assert "What is the capital of the moon?" in response.unformatted_response


def test_simple_chat_engine_astream_exception_handling():
    """
    Test that an exception thrown while retrieving the streamed LLM response gets bubbled up to the user.
    Also tests that the non-retrieved exception does not remain in an task that was not awaited leading to
    a 'Task exception was never retrieved' message during garbage collection.
    """

    class ExceptionThrownInTest(Exception):
        pass

    class ExceptionMockLLM(MockLLM):
        """Raises an exception while streaming back the mocked LLM response."""

        @classmethod
        def class_name(cls) -> str:
            return "ExceptionMockLLM"

        @llm_completion_callback()
        def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponseGen:
            def gen_prompt() -> CompletionResponseGen:
                for ch in prompt:
                    yield CompletionResponse(
                        text=prompt,
                        delta=ch,
                    )
                raise ExceptionThrownInTest("Exception thrown for testing purposes")

            return gen_prompt()

    async def async_part():
        engine = SimpleChatEngine.from_defaults(
            llm=ExceptionMockLLM(), memory=ChatMemoryBuffer.from_defaults()
        )
        response = await engine.astream_chat("Hello World!")

        with pytest.raises(ExceptionThrownInTest):
            async for response_part in response.async_response_gen():
                pass

    not_retrieved_exception = False

    def custom_exception_handler(loop, context):
        if context.get("message") == "Task exception was never retrieved":
            nonlocal not_retrieved_exception
            not_retrieved_exception = True

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(custom_exception_handler)
    result = loop.run_until_complete(async_part())
    loop.close()
    gc.collect()
    if not_retrieved_exception:
        pytest.fail(
            "Exception was not correctly handled - ended up in asyncio cleanup performed during garbage collection"
        )


# ---------------------------------------------------------------------------
# Regression tests for GitHub issue #21679
# ValueError when ChatMessage contains multiple blocks in write_response_to_history
# ---------------------------------------------------------------------------


def _make_multi_block_message() -> ChatMessage:
    """Build an assistant message with two blocks (e.g. reasoning + text)."""
    from llama_index.core.base.llms.types import ThinkingBlock

    return ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[
            ThinkingBlock(thinking="some reasoning", signature="sig"),
            TextBlock(text="final answer"),
        ],
    )


def _make_single_text_message(text: str = "hello") -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, blocks=[TextBlock(text=text)])


@pytest.mark.asyncio
async def test_awrite_response_to_history_multi_block_does_not_raise():
    """awrite_response_to_history must not raise ValueError when the final
    assistant message contains multiple blocks (e.g. reasoning + text).
    Before the fix, setting ChatMessage.content on a multi-block message
    raised: ValueError: ChatMessage contains multiple blocks, use 'ChatMessage.blocks' instead."""
    memory = ChatMemoryBuffer.from_defaults()
    multi_block_msg = _make_multi_block_message()

    async def _stream() -> AsyncGenerator[ChatResponse, None]:
        yield ChatResponse(message=multi_block_msg, delta="final answer")

    streaming_response = StreamingAgentChatResponse(achat_stream=_stream())
    streaming_response._ensure_async_setup()

    # Must not raise ValueError
    await streaming_response.awrite_response_to_history(memory)

    history = memory.get_all()
    assert any(m.role == MessageRole.ASSISTANT for m in history)


@pytest.mark.asyncio
async def test_awrite_response_to_history_single_block_still_updates_content():
    """For single-TextBlock messages the content override (partial stream
    handling) must still work correctly after the fix."""
    memory = ChatMemoryBuffer.from_defaults()
    partial_text_msg = _make_single_text_message("partial")

    async def _stream() -> AsyncGenerator[ChatResponse, None]:
        yield ChatResponse(message=partial_text_msg, delta="full text")

    streaming_response = StreamingAgentChatResponse(achat_stream=_stream())
    streaming_response._ensure_async_setup()

    await streaming_response.awrite_response_to_history(memory)

    history = memory.get_all()
    assistant_msgs = [m for m in history if m.role == MessageRole.ASSISTANT]
    assert len(assistant_msgs) == 1
    # Content is overwritten by final_text accumulated from deltas
    assert assistant_msgs[0].content == "full text"
