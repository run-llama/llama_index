import time
from typing import Any, List, Optional

import pytest

from llama_index.core import MockEmbedding
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.chat_engine.condense_plus_context import (
    CondensePlusContextChatEngine,
)
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
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
    chat_engine: CondensePlusContextChatEngine,
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


def test_condense_plus_context_custom_callback_manager() -> None:
    """Test that custom callback_manager passed to from_defaults is respected."""

    class CustomTraceHandler(BaseCallbackHandler):
        def __init__(self) -> None:
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
            self.traces: list[tuple[str, str | None]] = []

        def on_event_start(self, *args: Any, **kwargs: Any) -> str:
            return ""

        def on_event_end(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start_trace(self, trace_id: str | None = None) -> None:
            self.traces.append(("start", trace_id))

        def end_trace(
            self, trace_id: str | None = None, trace_map: dict | None = None
        ) -> None:
            self.traces.append(("end", trace_id))

    class DummyPostprocessor(BaseNodePostprocessor):
        def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
        ) -> List[NodeWithScore]:
            return nodes

    index = VectorStoreIndex.from_documents(
        [Document.example()], embed_model=MockEmbedding(embed_dim=3)
    )
    retriever = index.as_retriever()

    # 1. Passed explicitly as callback_manager
    handler = CustomTraceHandler()
    custom_cb = CallbackManager([handler])
    postprocessor = DummyPostprocessor()
    engine = CondensePlusContextChatEngine.from_defaults(
        retriever,
        llm=MockLLM(),
        memory=ChatMemoryBuffer.from_defaults(),
        node_postprocessors=[postprocessor],
        callback_manager=custom_cb,
    )
    assert engine.callback_manager is custom_cb
    assert postprocessor.callback_manager is custom_cb
    engine.chat("Hello World!")
    assert ("start", "chat") in handler.traces
    assert ("end", "chat") in handler.traces

    # 2. Passed via **kwargs
    handler_kwargs = CustomTraceHandler()
    custom_cb_kwargs = CallbackManager([handler_kwargs])
    postprocessor_kwargs = DummyPostprocessor()
    engine_kwargs = CondensePlusContextChatEngine.from_defaults(
        retriever,
        llm=MockLLM(),
        memory=ChatMemoryBuffer.from_defaults(),
        node_postprocessors=[postprocessor_kwargs],
        **{"callback_manager": custom_cb_kwargs},
    )
    assert engine_kwargs.callback_manager is custom_cb_kwargs
    assert postprocessor_kwargs.callback_manager is custom_cb_kwargs
    engine_kwargs.chat("Hello again!")
    assert ("start", "chat") in handler_kwargs.traces
    assert ("end", "chat") in handler_kwargs.traces
