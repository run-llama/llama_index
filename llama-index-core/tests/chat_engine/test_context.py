import pytest

from llama_index.core import MockEmbedding
from llama_index.core.chat_engine.context import (
    ContextChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms.mock import MockLLM
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


def test_chat_stream_partial_consumption(chat_engine: ContextChatEngine):
    """Verify that memory is persisted even if the stream is not fully consumed."""
    response = chat_engine.stream_chat("My name is Alice.")

    # Only consume a few tokens, then stop (simulating client disconnect).
    for i, _ in enumerate(response.response_gen):
        if i == 2:
            break

    # Both the user message and the partial assistant response must be saved.
    assert len(chat_engine.chat_history) == 2
    assert chat_engine.chat_history[0].role.value == "user"
    assert "Alice" in str(chat_engine.chat_history[0].content)


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


@pytest.mark.asyncio
async def test_chat_astream_partial_consumption(chat_engine: ContextChatEngine):
    """Verify that memory is persisted even if the async stream is not fully consumed."""
    response = await chat_engine.astream_chat("My name is Alice.")

    # Only consume a few tokens, then stop (simulating client disconnect).
    count = 0
    async for _ in response.async_response_gen():
        count += 1
        if count == 2:
            break

    # Both the user message and the partial assistant response must be saved.
    assert len(chat_engine.chat_history) == 2
    assert chat_engine.chat_history[0].role.value == "user"
    assert "Alice" in str(chat_engine.chat_history[0].content)
