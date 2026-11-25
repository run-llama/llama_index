import time
from typing import List, Optional

import pytest

from llama_index.core import MockEmbedding
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms.mock import MockLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import Document, NodeWithScore, QueryBundle, TextNode

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
    chat_engine: ContextChatEngine,
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


def test_stream_chat_history_write_completes_on_early_exit(
    chat_engine: ContextChatEngine,
):
    chat_engine._memory = ChatMemoryBuffer.from_defaults()
    response = chat_engine.stream_chat("Hello World!")
    assert len(chat_engine.chat_history) >= 1
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


class _TrackingPostprocessor(BaseNodePostprocessor):
    called_sync: bool = False
    called_async: bool = False
    last_query_bundle: Optional[QueryBundle] = None

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        self.called_sync = True
        self.last_query_bundle = query_bundle
        return nodes

    async def _apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        self.called_async = True
        self.last_query_bundle = query_bundle
        return nodes


class _DummyRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return [NodeWithScore(node=TextNode(text="dummy"))]

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)


@pytest.mark.asyncio
async def test_aget_nodes_calls_async_postprocessor() -> None:
    postprocessor = _TrackingPostprocessor()
    engine = ContextChatEngine.from_defaults(
        retriever=_DummyRetriever(),
        llm=MockLLM(),
        node_postprocessors=[postprocessor],
    )

    qb = QueryBundle("test", custom_embedding_strs=["custom"])
    await engine._aget_nodes(qb)

    assert postprocessor.called_async is True
    assert postprocessor.called_sync is False
    assert postprocessor.last_query_bundle is qb
    assert isinstance(postprocessor.last_query_bundle.query_str, str)


def test_get_nodes_preserves_query_bundle_and_non_empty_chat_memory() -> None:
    postprocessor = _TrackingPostprocessor()
    memory = ChatMemoryBuffer.from_defaults()
    engine = ContextChatEngine.from_defaults(
        retriever=_DummyRetriever(),
        llm=MockLLM(),
        memory=memory,
        node_postprocessors=[postprocessor],
    )

    engine.chat("First turn")
    qb = QueryBundle("Second turn via QueryBundle", custom_embedding_strs=["emb"])
    response = engine.chat(qb)
    assert "Second turn via QueryBundle" in str(response)
    assert postprocessor.called_sync is True
    assert postprocessor.last_query_bundle is qb
    assert postprocessor.last_query_bundle.query_str == "Second turn via QueryBundle"


def test_response_synthesizer_extracts_custom_llm_system_role() -> None:
    from llama_index.core.base.llms.types import LLMMetadata

    class _DeveloperRoleLLM(MockLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(system_role=MessageRole.DEVELOPER)

    llm = _DeveloperRoleLLM()
    engine = ContextChatEngine.from_defaults(
        retriever=_DummyRetriever(),
        llm=llm,
        system_prompt="Custom developer system instruction.",
    )
    synthesizer = engine._get_response_synthesizer(chat_history=[])
    qa_messages = synthesizer._text_qa_template.message_templates
    system_role_messages = [m for m in qa_messages if m.role == MessageRole.DEVELOPER]
    assert len(system_role_messages) == 1
    assert "Custom developer system instruction." in str(
        system_role_messages[0].content
    )
