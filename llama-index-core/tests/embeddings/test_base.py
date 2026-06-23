"""Embeddings."""

from typing import Any, List
from unittest.mock import patch
import pytest

from llama_index.core.base.embeddings.base import (
    EmbeddingResponse,
    SimilarityMode,
    _unpack_embedding,
    _unpack_embeddings,
    mean_agg,
)
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.instrumentation.events.embedding import EmbeddingEndEvent


def mock_get_text_embedding(text: str) -> List[float]:
    """Mock get text embedding."""
    # assume dimensions are 5
    if text == "Hello world.":
        return [1, 0, 0, 0, 0]
    elif text == "This is a test.":
        return [0, 1, 0, 0, 0]
    elif text == "This is another test.":
        return [0, 0, 1, 0, 0]
    elif text == "This is a test v2.":
        return [0, 0, 0, 1, 0]
    elif text == "This is a test v3.":
        return [0, 0, 0, 0, 1]
    elif text == "This is bar test.":
        return [0, 0, 1, 0, 0]
    elif text == "Hello world backup.":
        # this is used when "Hello world." is deleted.
        return [1, 0, 0, 0, 0]
    else:
        raise ValueError("Invalid text for `mock_get_text_embedding`.")


def mock_get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Mock get text embeddings."""
    return [mock_get_text_embedding(text) for text in texts]


@patch.object(MockEmbedding, "_get_text_embedding", side_effect=mock_get_text_embedding)
@patch.object(
    MockEmbedding, "_get_text_embeddings", side_effect=mock_get_text_embeddings
)
def test_get_text_embeddings(
    _mock_get_text_embeddings: Any, _mock_get_text_embedding: Any
) -> None:
    """Test get queued text embeddings."""
    embed_model = MockEmbedding(embed_dim=8)
    texts_to_embed = []
    for i in range(8):
        texts_to_embed.append("Hello world.")
    for i in range(8):
        texts_to_embed.append("This is a test.")
    for i in range(4):
        texts_to_embed.append("This is another test.")
    for i in range(4):
        texts_to_embed.append("This is a test v2.")

    result_embeddings = embed_model.get_text_embedding_batch(texts_to_embed)
    for i in range(8):
        assert result_embeddings[i] == [1, 0, 0, 0, 0]
    for i in range(8, 16):
        assert result_embeddings[i] == [0, 1, 0, 0, 0]
    for i in range(16, 20):
        assert result_embeddings[i] == [0, 0, 1, 0, 0]
    for i in range(20, 24):
        assert result_embeddings[i] == [0, 0, 0, 1, 0]


def test_embedding_similarity() -> None:
    """Test embedding similarity."""
    embed_model = MockEmbedding(embed_dim=3)
    text_embedding = [3.0, 4.0, 0.0]
    query_embedding = [0.0, 1.0, 0.0]
    cosine = embed_model.similarity(query_embedding, text_embedding)
    assert cosine == 0.8


def test_embedding_similarity_euclidean() -> None:
    embed_model = MockEmbedding(embed_dim=2)
    query_embedding = [1.0, 0.0]
    text1_embedding = [0.0, 1.0]  # further from query_embedding distance=1.414
    text2_embedding = [1.0, 1.0]  # closer to query_embedding distance=1.0
    euclidean_similarity1 = embed_model.similarity(
        query_embedding, text1_embedding, mode=SimilarityMode.EUCLIDEAN
    )
    euclidean_similarity2 = embed_model.similarity(
        query_embedding, text2_embedding, mode=SimilarityMode.EUCLIDEAN
    )
    assert euclidean_similarity1 < euclidean_similarity2


def test_mean_agg() -> None:
    """Test mean aggregation for embeddings."""
    embedding_0 = [3.0, 4.0, 0.0]
    embedding_1 = [0.0, 1.0, 0.0]
    output = mean_agg([embedding_0, embedding_1])
    assert output == [1.5, 2.5, 0.0]


def test_mean_agg_empty_list() -> None:
    """Test mean aggregation raises ValueError for empty list."""
    with pytest.raises(ValueError, match="No embeddings to aggregate"):
        mean_agg([])


# --- EmbeddingResponse / unpack helpers ---


def test_unpack_embedding_plain_list() -> None:
    """Plain List[float] unpacks to (embedding, None)."""
    emb, tc = _unpack_embedding([0.1, 0.2])
    assert emb == [0.1, 0.2]
    assert tc is None


def test_unpack_embedding_response() -> None:
    """EmbeddingResponse unpacks to (embedding, token_count)."""
    resp = EmbeddingResponse(embedding=[0.1, 0.2], token_count=42)
    emb, tc = _unpack_embedding(resp)
    assert emb == [0.1, 0.2]
    assert tc == 42


def test_unpack_embeddings_plain_list() -> None:
    """Batch of plain lists produces (embeddings, None)."""
    embs, tc = _unpack_embeddings([[0.1], [0.2]])
    assert embs == [[0.1], [0.2]]
    assert tc is None


def test_unpack_embeddings_with_token_counts() -> None:
    """Batch of EmbeddingResponse sums token counts."""
    batch = [
        EmbeddingResponse(embedding=[0.1], token_count=10),
        EmbeddingResponse(embedding=[0.2], token_count=None),
        EmbeddingResponse(embedding=[0.3], token_count=5),
    ]
    embs, tc = _unpack_embeddings(batch)
    assert embs == [[0.1], [0.2], [0.3]]
    assert tc == 15


def test_embedding_end_event_token_count_default() -> None:
    """EmbeddingEndEvent.token_count defaults to None."""
    event = EmbeddingEndEvent(chunks=["hello"], embeddings=[[0.1]])
    assert event.token_count is None


def test_embedding_end_event_token_count_set() -> None:
    """EmbeddingEndEvent accepts a token_count value."""
    event = EmbeddingEndEvent(chunks=["hello"], embeddings=[[0.1]], token_count=42)
    assert event.token_count == 42


def test_embedding_response_surfaces_token_count_in_event() -> None:
    """
    When a subclass returns EmbeddingResponse, the dispatched
    EmbeddingEndEvent carries the token_count.
    """
    import llama_index.core.instrumentation as instrument
    from llama_index.core.instrumentation.event_handlers import BaseEventHandler

    captured: List[EmbeddingEndEvent] = []

    class Capture(BaseEventHandler):
        @classmethod
        def class_name(cls) -> str:
            return "Capture"

        def handle(self, event: Any, **kwargs: Any) -> None:
            if isinstance(event, EmbeddingEndEvent):
                captured.append(event)

    handler = Capture()
    root = instrument.get_dispatcher()
    root.add_event_handler(handler)

    try:

        class TokenReportingEmbedding(MockEmbedding):
            def _get_text_embedding(self, text: str) -> EmbeddingResponse:
                vec = [0.0] * self.embed_dim
                return EmbeddingResponse(embedding=vec, token_count=99)

            async def _aget_query_embedding(self, query: str) -> EmbeddingResponse:
                return self._get_text_embedding(query)

            def _get_query_embedding(self, query: str) -> EmbeddingResponse:
                return self._get_text_embedding(query)

        model = TokenReportingEmbedding(embed_dim=5)
        result = model.get_text_embedding("test")
        assert result == [0.0] * 5
        assert len(captured) == 1
        assert captured[0].token_count == 99
    finally:
        root.event_handlers.remove(handler)


def test_plain_embedding_still_works() -> None:
    """
    Existing integrations returning plain List[float] continue to work
    and produce token_count=None in the event.
    """
    import llama_index.core.instrumentation as instrument
    from llama_index.core.instrumentation.event_handlers import BaseEventHandler

    captured: List[EmbeddingEndEvent] = []

    class Capture(BaseEventHandler):
        @classmethod
        def class_name(cls) -> str:
            return "Capture"

        def handle(self, event: Any, **kwargs: Any) -> None:
            if isinstance(event, EmbeddingEndEvent):
                captured.append(event)

    handler = Capture()
    root = instrument.get_dispatcher()
    root.add_event_handler(handler)

    try:
        model = MockEmbedding(embed_dim=5)
        result = model.get_text_embedding("test")
        assert isinstance(result, list)
        assert len(captured) == 1
        assert captured[0].token_count is None
    finally:
        root.event_handlers.remove(handler)
