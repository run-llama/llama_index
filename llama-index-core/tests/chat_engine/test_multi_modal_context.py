"""Tests for MultiModalContextChatEngine, MockMultiModalEmbedding, and
MockLLMWithChatMemoryOfLastCall."""

from typing import List

import pytest

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.chat_engine.multi_modal_context import MultiModalContextChatEngine
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.embeddings.mock_embed_model import MockMultiModalEmbedding
from llama_index.core.llms.mock import MockLLMWithChatMemoryOfLastCall
from llama_index.core.schema import (
    ImageNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)

# ---------------------------------------------------------------------------
# Minimal stub retriever
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful multi-modal assistant."


class _MockRetriever(BaseRetriever):
    """Retriever that always returns a fixed list of nodes."""

    def __init__(self, nodes: List[NodeWithScore]) -> None:
        self._fixed_nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return list(self._fixed_nodes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm() -> MockLLMWithChatMemoryOfLastCall:
    return MockLLMWithChatMemoryOfLastCall()


@pytest.fixture()
def text_nodes() -> List[NodeWithScore]:
    return [
        NodeWithScore(node=TextNode(text="Paris is the capital of France."), score=0.9),
        NodeWithScore(node=TextNode(text="The Eiffel Tower is in Paris."), score=0.8),
    ]


@pytest.fixture()
def image_nodes() -> List[NodeWithScore]:
    return [
        NodeWithScore(
            node=ImageNode(image_url="http://example.com/eiffel.jpg"), score=0.7
        ),
    ]


@pytest.fixture()
def chat_engine(
    mock_llm: MockLLMWithChatMemoryOfLastCall,
    text_nodes: List[NodeWithScore],
    image_nodes: List[NodeWithScore],
) -> MultiModalContextChatEngine:
    retriever = _MockRetriever(text_nodes + image_nodes)
    return MultiModalContextChatEngine.from_defaults(
        retriever=retriever,
        multi_modal_llm=mock_llm,
        system_prompt=SYSTEM_PROMPT,
    )


# ===========================================================================
# MockMultiModalEmbedding
# ===========================================================================


def test_mock_multi_modal_embedding_text() -> None:
    emb = MockMultiModalEmbedding(embed_dim=8)
    assert emb._get_text_embedding("hello") == [0.5] * 8
    assert emb._get_query_embedding("query") == [0.5] * 8


def test_mock_multi_modal_embedding_image() -> None:
    emb = MockMultiModalEmbedding(embed_dim=4)
    vec = emb._get_image_embedding("fake/path/to/image.jpg")
    assert vec == [0.5] * 4


def test_mock_multi_modal_embedding_class_name() -> None:
    emb = MockMultiModalEmbedding(embed_dim=2)
    assert emb.class_name() == "MockMultiModalEmbedding"


@pytest.mark.asyncio
async def test_mock_multi_modal_embedding_async() -> None:
    emb = MockMultiModalEmbedding(embed_dim=3)
    assert await emb._aget_text_embedding("hello") == [0.5] * 3
    assert await emb._aget_query_embedding("q") == [0.5] * 3
    assert await emb._aget_image_embedding("fake.png") == [0.5] * 3


# ===========================================================================
# MockLLMWithChatMemoryOfLastCall
# ===========================================================================


def test_mock_llm_metadata(mock_llm: MockLLMWithChatMemoryOfLastCall) -> None:
    assert mock_llm.metadata.model_name == "mock-multi-modal"
    assert mock_llm.class_name() == "MockLLMWithChatMemoryOfLastCall"


def test_mock_llm_chat_records_messages(
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    msgs = [ChatMessage(content="hello", role=MessageRole.USER)]
    response = mock_llm.chat(msgs)
    assert mock_llm.last_messages_received == msgs
    assert response.message.content == "mock multi-modal response"


def test_mock_llm_chat_overwrites_previous_messages(
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    msgs1 = [ChatMessage(content="first", role=MessageRole.USER)]
    msgs2 = [ChatMessage(content="second", role=MessageRole.USER)]
    mock_llm.chat(msgs1)
    mock_llm.chat(msgs2)
    assert mock_llm.last_messages_received == msgs2


def test_mock_llm_stream_chat_records_messages_after_consumption(
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    msgs = [ChatMessage(content="stream test", role=MessageRole.USER)]
    # stream_chat is a generator function; body runs on first next()
    chunks = list(mock_llm.stream_chat(msgs))
    assert mock_llm.last_messages_received == msgs
    assert len(chunks) == 1
    assert chunks[0].delta == "mock multi-modal response"


@pytest.mark.asyncio
async def test_mock_llm_achat_records_messages(
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    msgs = [ChatMessage(content="async hello", role=MessageRole.USER)]
    response = await mock_llm.achat(msgs)
    assert mock_llm.last_messages_received == msgs
    assert response.message.content == "mock multi-modal response"


@pytest.mark.asyncio
async def test_mock_llm_astream_chat_records_messages(
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    msgs = [ChatMessage(content="async stream", role=MessageRole.USER)]
    stream = await mock_llm.astream_chat(msgs)
    # messages are captured when astream_chat is awaited (not a generator func)
    assert mock_llm.last_messages_received == msgs
    chunks = [chunk async for chunk in stream]
    assert len(chunks) == 1
    assert chunks[0].delta == "mock multi-modal response"


# ===========================================================================
# MultiModalContextChatEngine — chat()
# ===========================================================================


def test_chat_returns_response(
    chat_engine: MultiModalContextChatEngine,
) -> None:
    response = chat_engine.chat("What is in the image?")
    assert response.response == "mock multi-modal response"
    assert len(chat_engine.chat_history) == 2


def test_chat_message_contains_text_block_with_context(
    chat_engine: MultiModalContextChatEngine,
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    chat_engine.chat("Describe what you see.")
    user_msg = mock_llm.last_messages_received[-1]
    text_blocks = [b for b in user_msg.blocks if isinstance(b, TextBlock)]
    assert len(text_blocks) == 1
    combined = text_blocks[0].text
    assert "Paris is the capital of France." in combined
    assert "Describe what you see." in combined


def test_chat_message_contains_image_block(
    chat_engine: MultiModalContextChatEngine,
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    chat_engine.chat("What is shown?")
    user_msg = mock_llm.last_messages_received[-1]
    image_blocks = [b for b in user_msg.blocks if isinstance(b, ImageBlock)]
    assert len(image_blocks) == 1
    assert "eiffel.jpg" in str(image_blocks[0].url)


def test_chat_prefix_system_message_is_first(
    chat_engine: MultiModalContextChatEngine,
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    chat_engine.chat("Hello")
    messages = mock_llm.last_messages_received
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[0].content == SYSTEM_PROMPT


def test_chat_history_grows_across_turns(
    chat_engine: MultiModalContextChatEngine,
) -> None:
    chat_engine.chat("First question")
    assert len(chat_engine.chat_history) == 2
    chat_engine.chat("Second question")
    assert len(chat_engine.chat_history) == 4


def test_reset_clears_history(chat_engine: MultiModalContextChatEngine) -> None:
    chat_engine.chat("Something")
    assert len(chat_engine.chat_history) == 2
    chat_engine.reset()
    assert len(chat_engine.chat_history) == 0


def test_chat_with_explicit_history_overrides_memory(
    chat_engine: MultiModalContextChatEngine,
) -> None:
    explicit_history = [
        ChatMessage(content="Earlier question", role=MessageRole.USER),
        ChatMessage(content="Earlier answer", role=MessageRole.ASSISTANT),
    ]
    chat_engine.chat("New question", chat_history=explicit_history)
    # 2 explicit prior + 2 new from this turn
    assert len(chat_engine.chat_history) == 4


def test_chat_text_only_no_image_blocks(
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    """When the retriever returns no image nodes, no ImageBlocks appear."""
    retriever = _MockRetriever(
        [NodeWithScore(node=TextNode(text="Some text."), score=1.0)]
    )
    engine = MultiModalContextChatEngine.from_defaults(
        retriever=retriever,
        multi_modal_llm=mock_llm,
    )
    engine.chat("Tell me about the text.")
    user_msg = mock_llm.last_messages_received[-1]
    image_blocks = [b for b in user_msg.blocks if isinstance(b, ImageBlock)]
    assert len(image_blocks) == 0


def test_from_defaults_raises_without_multimodal_llm() -> None:
    """from_defaults raises ValueError when given a plain LLM."""
    from llama_index.core.llms.mock import MockLLM

    retriever = _MockRetriever([])
    with pytest.raises(ValueError, match="MultiModalLLM"):
        MultiModalContextChatEngine.from_defaults(
            retriever=retriever,
            multi_modal_llm=MockLLM(),  # type: ignore[arg-type]
        )


def test_from_defaults_raises_with_both_system_prompt_and_prefix_messages(
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    retriever = _MockRetriever([])
    with pytest.raises(ValueError, match="system_prompt"):
        MultiModalContextChatEngine.from_defaults(
            retriever=retriever,
            multi_modal_llm=mock_llm,
            system_prompt="sys",
            prefix_messages=[ChatMessage(content="x", role=MessageRole.SYSTEM)],
        )


# ===========================================================================
# MultiModalContextChatEngine — stream_chat()
# ===========================================================================


def test_stream_chat_yields_delta(chat_engine: MultiModalContextChatEngine) -> None:
    response = chat_engine.stream_chat("Describe this.")
    tokens = list(response.response_gen)
    assert tokens == ["mock multi-modal response"]


def test_stream_chat_updates_history_after_consumption(
    chat_engine: MultiModalContextChatEngine,
) -> None:
    response = chat_engine.stream_chat("Stream question")
    list(response.response_gen)  # consume to trigger memory write
    assert len(chat_engine.chat_history) == 2


def test_stream_chat_records_messages_after_consumption(
    chat_engine: MultiModalContextChatEngine,
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    response = chat_engine.stream_chat("Stream this.")
    list(response.response_gen)  # generator body executes on consumption
    user_msg = mock_llm.last_messages_received[-1]
    text_blocks = [b for b in user_msg.blocks if isinstance(b, TextBlock)]
    assert len(text_blocks) == 1
    assert "Stream this." in text_blocks[0].text


def test_stream_chat_contains_image_block_after_consumption(
    chat_engine: MultiModalContextChatEngine,
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    response = chat_engine.stream_chat("Stream image.")
    list(response.response_gen)
    user_msg = mock_llm.last_messages_received[-1]
    image_blocks = [b for b in user_msg.blocks if isinstance(b, ImageBlock)]
    assert len(image_blocks) == 1


# ===========================================================================
# MultiModalContextChatEngine — achat()
# ===========================================================================


@pytest.mark.asyncio
async def test_achat_returns_response(
    chat_engine: MultiModalContextChatEngine,
) -> None:
    response = await chat_engine.achat("Async question")
    assert response.response == "mock multi-modal response"
    assert len(chat_engine.chat_history) == 2


@pytest.mark.asyncio
async def test_achat_message_has_text_and_image_blocks(
    chat_engine: MultiModalContextChatEngine,
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    await chat_engine.achat("Async describe")
    user_msg = mock_llm.last_messages_received[-1]
    text_blocks = [b for b in user_msg.blocks if isinstance(b, TextBlock)]
    image_blocks = [b for b in user_msg.blocks if isinstance(b, ImageBlock)]
    assert len(text_blocks) == 1
    assert len(image_blocks) == 1
    assert "Async describe" in text_blocks[0].text


@pytest.mark.asyncio
async def test_achat_history_grows(chat_engine: MultiModalContextChatEngine) -> None:
    await chat_engine.achat("First async")
    assert len(chat_engine.chat_history) == 2
    await chat_engine.achat("Second async")
    assert len(chat_engine.chat_history) == 4


# ===========================================================================
# MultiModalContextChatEngine — astream_chat()
# ===========================================================================


@pytest.mark.asyncio
async def test_astream_chat_yields_delta(
    chat_engine: MultiModalContextChatEngine,
) -> None:
    response = await chat_engine.astream_chat("Async stream question")
    tokens = [tok async for tok in response.async_response_gen()]
    assert tokens == ["mock multi-modal response"]


@pytest.mark.asyncio
async def test_astream_chat_updates_history_after_consumption(
    chat_engine: MultiModalContextChatEngine,
) -> None:
    response = await chat_engine.astream_chat("Async stream")
    async for _ in response.async_response_gen():
        pass
    assert len(chat_engine.chat_history) == 2


@pytest.mark.asyncio
async def test_astream_chat_messages_captured_before_consumption(
    chat_engine: MultiModalContextChatEngine,
    mock_llm: MockLLMWithChatMemoryOfLastCall,
) -> None:
    # astream_chat is a regular async def (not a generator func), so messages
    # are stored when the coroutine is awaited, before consuming the stream.
    await chat_engine.astream_chat("Async stream before consume")
    assert len(mock_llm.last_messages_received) > 0
    user_msg = mock_llm.last_messages_received[-1]
    # blocks[0] is always the TextBlock (context + question)
    text_blocks = [b for b in user_msg.blocks if isinstance(b, TextBlock)]
    assert len(text_blocks) == 1
    assert "Async stream before consume" in text_blocks[0].text


# ===========================================================================
# MultiModalVectorStoreIndex.as_chat_engine() wiring
# ===========================================================================


def test_as_chat_engine_context_mode_returns_multimodal_engine() -> None:
    """as_chat_engine(ChatMode.CONTEXT) returns a MultiModalContextChatEngine."""
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

    mm_llm = MockLLMWithChatMemoryOfLastCall()
    mm_embed = MockMultiModalEmbedding(embed_dim=3)
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mm_embed,
        image_embed_model=mm_embed,
    )
    engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        multi_modal_llm=mm_llm,
    )
    assert isinstance(engine, MultiModalContextChatEngine)


def test_as_chat_engine_context_mode_raises_with_plain_llm() -> None:
    """as_chat_engine(ChatMode.CONTEXT) raises ValueError for a plain text LLM."""
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex
    from llama_index.core.llms.mock import MockLLM

    mm_embed = MockMultiModalEmbedding(embed_dim=3)
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mm_embed,
        image_embed_model=mm_embed,
    )
    with pytest.raises(ValueError, match="MultiModalLLM"):
        index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            llm=MockLLM(),
        )


def test_as_chat_engine_context_llm_param_accepts_multimodal_llm() -> None:
    """Passing a MultiModalLLM via the llm= kwarg also works."""
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

    mm_llm = MockLLMWithChatMemoryOfLastCall()
    mm_embed = MockMultiModalEmbedding(embed_dim=3)
    index = MultiModalVectorStoreIndex(
        nodes=[],
        embed_model=mm_embed,
        image_embed_model=mm_embed,
    )
    # Passing via llm= should also work if it's a MultiModalLLM instance
    engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        llm=mm_llm,
    )
    assert isinstance(engine, MultiModalContextChatEngine)
