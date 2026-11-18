import time
from typing import Any, List, Optional

import pytest

from llama_index.core import MockEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.embeddings import MockMultiModalEmbedding
from llama_index.core.chat_engine.multi_modal_context import (
    MultiModalContextChatEngine,
)
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import Document, ImageDocument, NodeWithScore, QueryBundle
from llama_index.core.llms import TextBlock, ImageBlock, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMode

SYSTEM_PROMPT = "Talk like a pirate."


@pytest.fixture()
def chat_engine() -> MultiModalContextChatEngine:
    # Base64 string for a 1×1 transparent PNG
    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    img = ImageDocument(image=base64_str, metadata={"file_name": "tiny.png"})
    embed_model_text = MockEmbedding(embed_dim=3)
    embed_model_image = MockMultiModalEmbedding(embed_dim=3)
    index = MultiModalVectorStoreIndex.from_documents(
        [Document.example(), img],
        image_embed_model=embed_model_image,
        embed_model=embed_model_text,
    )
    fixture = index.as_chat_engine(
        similarity_top_k=2,
        image_similarity_top_k=1,
        chat_mode=ChatMode.CONTEXT,
        llm=MockLLMWithChatMemoryOfLastCall(),
        system_prompt=SYSTEM_PROMPT,
    )
    assert isinstance(fixture, MultiModalContextChatEngine)
    return fixture


def test_chat(chat_engine: MultiModalContextChatEngine):
    response = chat_engine.chat("Hello World!")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2
    assert len(response.source_nodes) == 2  # one image and one text
    assert len(response.sources) == 1
    assert response.sources[0].tool_name == "retriever"
    assert len(response.sources[0].raw_output["image_nodes"]) == 1
    assert len(response.sources[0].raw_output["text_nodes"]) == 1

    llm = chat_engine._multi_modal_llm
    assert len(llm.last_chat_messages) == 2  # system prompt and user message
    assert (
        len(llm.last_chat_messages[1].blocks) == 2
    )  # user message consisting of text block containing text context and query and image block
    assert (
        isinstance(llm.last_chat_messages[1].blocks[0], ImageBlock)
        and isinstance(llm.last_chat_messages[1].blocks[1], TextBlock)
    ) or (
        isinstance(llm.last_chat_messages[1].blocks[0], TextBlock)
        and isinstance(llm.last_chat_messages[1].blocks[1], ImageBlock)
    )
    assert "chat" in llm.last_called_chat_function

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


def test_chat_stream(chat_engine: MultiModalContextChatEngine):
    response = chat_engine.stream_chat("Hello World!")

    num_iters = 0
    for _ in response.response_gen:
        num_iters += 1

    assert num_iters > 10
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2
    assert len(response.source_nodes) == 2  # one image and one text
    assert len(response.sources) == 1
    assert response.sources[0].tool_name == "retriever"
    assert len(response.sources[0].raw_output["image_nodes"]) == 1
    assert len(response.sources[0].raw_output["text_nodes"]) == 1

    llm = chat_engine._multi_modal_llm
    assert len(llm.last_chat_messages) == 2  # system prompt and user message
    assert (
        len(llm.last_chat_messages[1].blocks) == 2
    )  # user message consisting of text block containing text context and query and image block
    assert (
        isinstance(llm.last_chat_messages[1].blocks[0], ImageBlock)
        and isinstance(llm.last_chat_messages[1].blocks[1], TextBlock)
    ) or (
        isinstance(llm.last_chat_messages[1].blocks[0], TextBlock)
        and isinstance(llm.last_chat_messages[1].blocks[1], ImageBlock)
    )
    assert "stream_chat" in llm.last_called_chat_function

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
async def test_achat(chat_engine: MultiModalContextChatEngine):
    response = await chat_engine.achat("Hello World!")
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2
    assert len(response.source_nodes) == 2  # one image and one text
    assert len(response.sources) == 1
    assert response.sources[0].tool_name == "retriever"
    assert len(response.sources[0].raw_output["image_nodes"]) == 1
    assert len(response.sources[0].raw_output["text_nodes"]) == 1

    llm = chat_engine._multi_modal_llm
    assert len(llm.last_chat_messages) == 2  # system prompt and user message
    assert (
        len(llm.last_chat_messages[1].blocks) == 2
    )  # user message consisting of text block containing text context and query and image block
    assert (
        isinstance(llm.last_chat_messages[1].blocks[0], ImageBlock)
        and isinstance(llm.last_chat_messages[1].blocks[1], TextBlock)
    ) or (
        isinstance(llm.last_chat_messages[1].blocks[0], TextBlock)
        and isinstance(llm.last_chat_messages[1].blocks[1], ImageBlock)
    )
    assert "achat" in llm.last_called_chat_function

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
async def test_chat_astream(chat_engine: MultiModalContextChatEngine):
    response = await chat_engine.astream_chat("Hello World!")

    num_iters = 0
    async for _ in response.async_response_gen():
        num_iters += 1

    assert num_iters > 10
    assert SYSTEM_PROMPT in str(response)
    assert "Hello World!" in str(response)
    assert len(chat_engine.chat_history) == 2
    assert len(response.source_nodes) == 2  # one image and one text
    assert len(response.sources) == 1
    assert response.sources[0].tool_name == "retriever"
    assert len(response.sources[0].raw_output["image_nodes"]) == 1
    assert len(response.sources[0].raw_output["text_nodes"]) == 1

    llm = chat_engine._multi_modal_llm
    assert len(llm.last_chat_messages) == 2  # system prompt and user message
    assert (
        len(llm.last_chat_messages[1].blocks) == 2
    )  # user message consisting of text block containing text context and query and image block
    assert (
        isinstance(llm.last_chat_messages[1].blocks[0], ImageBlock)
        and isinstance(llm.last_chat_messages[1].blocks[1], TextBlock)
    ) or (
        isinstance(llm.last_chat_messages[1].blocks[0], TextBlock)
        and isinstance(llm.last_chat_messages[1].blocks[1], ImageBlock)
    )
    assert "astream_chat" in llm.last_called_chat_function

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
    chat_engine: MultiModalContextChatEngine,
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
    chat_engine: MultiModalContextChatEngine,
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


def test_multi_modal_context_custom_callback_manager() -> None:
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

    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    img = ImageDocument(image=base64_str, metadata={"file_name": "tiny.png"})
    embed_model_text = MockEmbedding(embed_dim=3)
    embed_model_image = MockMultiModalEmbedding(embed_dim=3)
    index = MultiModalVectorStoreIndex.from_documents(
        [Document.example(), img],
        image_embed_model=embed_model_image,
        embed_model=embed_model_text,
    )
    retriever = index.as_retriever(similarity_top_k=2, image_similarity_top_k=1)

    # 1. Passed explicitly as callback_manager
    handler = CustomTraceHandler()
    custom_cb = CallbackManager([handler])
    postprocessor = DummyPostprocessor()
    engine = MultiModalContextChatEngine.from_defaults(
        retriever,
        multi_modal_llm=MockLLMWithChatMemoryOfLastCall(),
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
    engine_kwargs = MultiModalContextChatEngine.from_defaults(
        retriever,
        multi_modal_llm=MockLLMWithChatMemoryOfLastCall(),
        memory=ChatMemoryBuffer.from_defaults(),
        node_postprocessors=[postprocessor_kwargs],
        **{"callback_manager": custom_cb_kwargs},
    )
    assert engine_kwargs.callback_manager is custom_cb_kwargs
    assert postprocessor_kwargs.callback_manager is custom_cb_kwargs
    engine_kwargs.chat("Hello again!")
    assert ("start", "chat") in handler_kwargs.traces
    assert ("end", "chat") in handler_kwargs.traces
