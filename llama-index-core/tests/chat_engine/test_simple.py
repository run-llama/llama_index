import gc
import asyncio
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    CompletionResponseGen,
)
from typing import Any
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.mock import MockLLM
import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.simple import SimpleChatEngine


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


@pytest.mark.asyncio()
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
    """Test that an exception thrown while retrieving the streamed LLM response gets bubbled up to the user.
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
