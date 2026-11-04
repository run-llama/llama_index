import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from mistralai.client.models import EmbeddingResponse

from llama_index.embeddings.mistralai import MistralAIEmbedding


@pytest.fixture()
def embedding_response() -> EmbeddingResponse:
    return EmbeddingResponse.model_validate(
        {
            "id": "embedding-response",
            "object": "list",
            "model": "mistral-embed",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
            "data": [{"index": 0, "object": "embedding", "embedding": [0.1, 0.2]}],
        }
    )


@pytest.mark.parametrize("kind", ["text", "query", "text_batch"])
def test_async_embedding_uses_async_client(
    monkeypatch: pytest.MonkeyPatch, embedding_response: EmbeddingResponse, kind: str
) -> None:
    embedder = MistralAIEmbedding(api_key="placeholder-key")
    sync_create = Mock(return_value=embedding_response)
    async_create = AsyncMock(return_value=embedding_response)
    monkeypatch.setattr(embedder._client.embeddings, "create", sync_create)
    monkeypatch.setattr(embedder._client.embeddings, "create_async", async_create)

    if kind == "text_batch":
        result = asyncio.run(embedder.aget_text_embedding_batch(["hello"]))
        assert result == [[0.1, 0.2]]
    else:
        result = asyncio.run(getattr(embedder, f"aget_{kind}_embedding")("hello"))
        assert result == [0.1, 0.2]

    async_create.assert_awaited_once_with(model="mistral-embed", inputs=["hello"])
    sync_create.assert_not_called()


@pytest.mark.parametrize("kind", ["text", "query", "text_batch"])
def test_sync_embedding_uses_sync_client(
    monkeypatch: pytest.MonkeyPatch, embedding_response: EmbeddingResponse, kind: str
) -> None:
    embedder = MistralAIEmbedding(api_key="placeholder-key")
    sync_create = Mock(return_value=embedding_response)
    monkeypatch.setattr(embedder._client.embeddings, "create", sync_create)

    if kind == "text_batch":
        assert embedder.get_text_embedding_batch(["hello"]) == [[0.1, 0.2]]
    else:
        assert getattr(embedder, f"get_{kind}_embedding")("hello") == [0.1, 0.2]

    sync_create.assert_called_once_with(model="mistral-embed", inputs=["hello"])
