import time
from typing import List

import pytest

from llama_index.core import MockEmbedding
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.chat_engine.condense_plus_context import (
    CondensePlusContextChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import Document, NodeWithScore, QueryBundle

SYSTEM_PROMPT = "Talk like a pirate."


@pytest.fixture()
def chat_engine() -> CondensePlusContextChatEngine:
    index = VectorStoreIndex.from_documents(
        [Document.example()], embed_model=MockEmbedding(embed_dim=3)
    )
    retriever = index.as_retriever()
    return CondensePlusContextChatEngine.from_defaults(
        retriever, llm=MockLLM(), system_prompt=SYSTEM_PROMPT
    )


def test_chat(chat_engine: CondensePlusContextChatEngine):
    response = chat_engine.chat("Hello World!")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2

    response = chat_engine.chat("What is the capital of the moon?")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert "What is the capital of the moon?" in str(response)
    assert len(chat_engine.chat_history) == 4


def test_chat_stream(chat_engine: CondensePlusContextChatEngine):
    response = chat_engine.stream_chat("Hello World!")

    num_iters = 0
    for _ in response.response_gen:
        num_iters += 1

    assert num_iters == 1
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2

    response = chat_engine.stream_chat("What is the capital of the moon?")

    num_iters = 0
    for _ in response.response_gen:
        num_iters += 1

    assert num_iters == 1
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert "What is the capital of the moon?" in str(response)
    assert len(chat_engine.chat_history) == 4


def test_stream_chat_memory_not_lost_on_incomplete_consumption(
    chat_engine: CondensePlusContextChatEngine,
):
    # Use ChatMemoryBuffer to avoid per-event-loop aiosqlite isolation
    # when the background thread writes memory.
    chat_engine._memory = ChatMemoryBuffer.from_defaults()
    response = chat_engine.stream_chat("Hello World!")
    assert len(chat_engine.chat_history) >= 1
    assert chat_engine.chat_history[0].role == MessageRole.USER
    assert "Hello World!" in str(chat_engine.chat_history[0].content)
    for _ in response.response_gen:
        break
    deadline = time.time() + 2.0
    while not response.is_done and time.time() < deadline:
        time.sleep(0.01)
    assert response.is_done
    assert len(chat_engine.chat_history) == 2
    assert chat_engine.chat_history[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_astream_chat_memory_not_lost_on_incomplete_consumption(
    chat_engine: CondensePlusContextChatEngine,
):
    response = await chat_engine.astream_chat("Hello World!")
    assert len(chat_engine.chat_history) == 1
    assert chat_engine.chat_history[0].role == MessageRole.USER
    assert "Hello World!" in str(chat_engine.chat_history[0].content)
    async for _ in response.async_response_gen():
        break
    assert response.awrite_response_to_history_task is not None
    await response.awrite_response_to_history_task
    assert len(chat_engine.chat_history) == 2
    assert chat_engine.chat_history[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_achat(chat_engine: CondensePlusContextChatEngine):
    response = await chat_engine.achat("Hello World!")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2

    response = await chat_engine.achat("What is the capital of the moon?")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert "What is the capital of the moon?" in str(response)
    assert len(chat_engine.chat_history) == 4


@pytest.mark.asyncio
async def test_chat_astream(chat_engine: CondensePlusContextChatEngine):
    response = await chat_engine.astream_chat("Hello World!")

    num_iters = 0
    async for _ in response.async_response_gen():
        num_iters += 1

    assert num_iters == 1
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2

    response = await chat_engine.astream_chat("What is the capital of the moon?")

    num_iters = 0
    async for _ in response.async_response_gen():
        num_iters += 1

    assert num_iters == 1
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert "What is the capital of the moon?" in str(response)
    assert len(chat_engine.chat_history) == 4


class _EmptyRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return []

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return []


def _empty_context_engine(fallback_to_llm: bool) -> CondensePlusContextChatEngine:
    return CondensePlusContextChatEngine.from_defaults(
        _EmptyRetriever(),
        llm=MockLLM(),
        system_prompt=SYSTEM_PROMPT,
        fallback_to_llm=fallback_to_llm,
    )


def test_empty_retrieval_returns_empty_response_by_default():
    chat_engine = _empty_context_engine(fallback_to_llm=False)

    assert str(chat_engine.chat("Hello World!")) == "Empty Response"
    assert (
        "".join(_empty_context_engine(False).stream_chat("Hello World!").response_gen)
        == "Empty Response"
    )


def test_empty_retrieval_calls_llm_when_fallback_enabled():
    chat_engine = _empty_context_engine(fallback_to_llm=True)
    response = chat_engine.chat("Hello World!")

    # MockLLM echoes the prompt, so this only passes if the LLM was really called
    assert "Empty Response" not in str(response)
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2


@pytest.mark.asyncio
async def test_empty_retrieval_astream_chat_with_fallback():
    chat_engine = _empty_context_engine(fallback_to_llm=True)
    stream = await chat_engine.astream_chat("Hello World!")
    response = "".join([token async for token in stream.async_response_gen()])

    assert "Empty Response" not in response
    assert SYSTEM_PROMPT in response

    default_engine = _empty_context_engine(fallback_to_llm=False)
    default_stream = await default_engine.astream_chat("Hello World!")
    assert (
        "".join([token async for token in default_stream.async_response_gen()])
        == "Empty Response"
    )
