import time

import pytest

from llama_index.core import MockEmbedding
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.chat_engine.context import (
    ContextChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import Document, QueryBundle

SYSTEM_PROMPT = "Talk like a pirate."


@pytest.fixture()
def chat_engine() -> ContextChatEngine:
    index = VectorStoreIndex.from_documents(
        [Document.example()], embed_model=MockEmbedding(embed_dim=3)
    )
    retriever = index.as_retriever()
    return ContextChatEngine.from_defaults(
        retriever, llm=MockLLM(), system_prompt=SYSTEM_PROMPT
    )


def test_chat(chat_engine: ContextChatEngine):
    response = chat_engine.chat("Hello World!")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2

    response = chat_engine.chat("What is the capital of the moon?")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert "What is the capital of the moon?" in str(response)
    assert len(chat_engine.chat_history) == 4

    chat_engine.reset()
    q = QueryBundle("Hello World through QueryBundle")
    response = chat_engine.chat(q)
    assert str(q) in str(response)
    assert len(chat_engine.chat_history) == 2
    assert str(q) in str(chat_engine.chat_history[0])


def test_chat_stream(chat_engine: ContextChatEngine):
    response = chat_engine.stream_chat("Hello World!")

    num_iters = 0
    for _ in response.response_gen:
        num_iters += 1

    assert num_iters > 10
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2

    response = chat_engine.stream_chat("What is the capital of the moon?")

    num_iters = 0
    for _ in response.response_gen:
        num_iters += 1

    assert num_iters > 10
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert "What is the capital of the moon?" in str(response)
    assert len(chat_engine.chat_history) == 4

    chat_engine.reset()
    q = QueryBundle("Hello World through QueryBundle")
    response = chat_engine.stream_chat(q)
    num_iters = 0
    for _ in response.response_gen:
        num_iters += 1
    assert num_iters > 10
    assert str(q) in str(response)
    assert len(chat_engine.chat_history) == 2
    assert str(q) in str(chat_engine.chat_history[0])


@pytest.mark.asyncio
async def test_achat(chat_engine: ContextChatEngine):
    response = await chat_engine.achat("Hello World!")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2

    response = await chat_engine.achat("What is the capital of the moon?")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert "What is the capital of the moon?" in str(response)
    assert len(chat_engine.chat_history) == 4

    chat_engine.reset()
    q = QueryBundle("Hello World through QueryBundle")
    response = await chat_engine.achat(q)
    assert str(q) in str(response)
    assert len(chat_engine.chat_history) == 2
    assert str(q) in str(chat_engine.chat_history[0])


@pytest.mark.asyncio
async def test_chat_astream(chat_engine: ContextChatEngine):
    response = await chat_engine.astream_chat("Hello World!")

    num_iters = 0
    async for _ in response.async_response_gen():
        num_iters += 1

    assert num_iters > 10
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2

    response = await chat_engine.astream_chat("What is the capital of the moon?")

    num_iters = 0
    async for _ in response.async_response_gen():
        num_iters += 1

    assert num_iters > 10
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert "What is the capital of the moon?" in str(response)
    assert len(chat_engine.chat_history) == 4

    chat_engine.reset()
    q = QueryBundle("Hello World through QueryBundle")
    response = await chat_engine.astream_chat(q)
    num_iters = 0
    async for _ in response.async_response_gen():
        num_iters += 1
    assert num_iters > 10
    assert str(q) in str(response)
    assert len(chat_engine.chat_history) == 2
    assert str(q) in str(chat_engine.chat_history[0])


def test_stream_chat_memory_not_lost_on_incomplete_consumption(
    chat_engine: ContextChatEngine,
):
    # Use ChatMemoryBuffer to avoid per-event-loop aiosqlite isolation
    # when the background thread writes memory.
    chat_engine._memory = ChatMemoryBuffer.from_defaults()
    response = chat_engine.stream_chat("Hello World!")
    assert len(chat_engine.chat_history) == 1
    assert chat_engine.chat_history[0].role == MessageRole.USER
    assert "Hello World!" in str(chat_engine.chat_history[0].content)
    for i, _ in enumerate(response.response_gen):
        if i >= 2:
            break
    deadline = time.time() + 2.0
    while not response.is_done and time.time() < deadline:
        time.sleep(0.01)
    assert response.is_done
    assert len(chat_engine.chat_history) == 2
    assert chat_engine.chat_history[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_astream_chat_memory_not_lost_on_incomplete_consumption(
    chat_engine: ContextChatEngine,
):
    response = await chat_engine.astream_chat("Hello World!")
    assert len(chat_engine.chat_history) == 1
    assert chat_engine.chat_history[0].role == MessageRole.USER
    assert "Hello World!" in str(chat_engine.chat_history[0].content)
    i = 0
    async for _ in response.async_response_gen():
        i += 1
        if i >= 2:
            break
    assert response.awrite_response_to_history_task is not None
    await response.awrite_response_to_history_task
    assert len(chat_engine.chat_history) == 2
    assert chat_engine.chat_history[1].role == MessageRole.ASSISTANT


def test_stream_chat_history_write_completes_on_early_exit(
    chat_engine: ContextChatEngine,
):
    chat_engine._memory = ChatMemoryBuffer.from_defaults()
    response = chat_engine.stream_chat("Hello World!")
    assert len(chat_engine.chat_history) == 1
    gen = response.response_gen
    for i, _ in enumerate(gen):
        if i >= 2:
            break
    gen.close()
    assert len(chat_engine.chat_history) == 2
    assert chat_engine.chat_history[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_astream_chat_history_write_completes_on_early_exit(
    chat_engine: ContextChatEngine,
):
    chat_engine._memory = ChatMemoryBuffer.from_defaults()
    response = await chat_engine.astream_chat("Hello World!")
    assert len(chat_engine.chat_history) == 1
    i = 0
    gen = response.async_response_gen()
    async for _ in gen:
        i += 1
        if i >= 2:
            break
    await gen.aclose()
    assert len(chat_engine.chat_history) == 2
    assert chat_engine.chat_history[1].role == MessageRole.ASSISTANT
