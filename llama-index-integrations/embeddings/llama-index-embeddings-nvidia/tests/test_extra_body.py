from unittest.mock import AsyncMock, Mock, patch

import pytest

from llama_index.embeddings.nvidia import NVIDIAEmbedding


class MockEmbeddingResponse:
    def __init__(self) -> None:
        self.data = [Mock(embedding=[1.0, 2.0, 3.0], index=0)]


@pytest.fixture()
def mock_clients():
    mock_response = MockEmbeddingResponse()
    with (
        patch("llama_index.embeddings.nvidia.base.OpenAI") as mock_openai_cls,
        patch("llama_index.embeddings.nvidia.base.AsyncOpenAI") as mock_async_cls,
    ):
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        mock_aclient = Mock()
        mock_aclient.embeddings.create = AsyncMock(return_value=mock_response)
        mock_async_cls.return_value = mock_aclient

        yield mock_client, mock_aclient


@pytest.mark.parametrize(
    ("method_name", "arg", "input_type"),
    [
        ("get_query_embedding", "nvidia", "query"),
        ("get_text_embedding", "nvidia", "passage"),
        ("get_text_embedding_batch", ["nvidia"], "passage"),
    ],
)
def test_sync_extra_body(mock_clients, method_name, arg, input_type) -> None:
    mock_client, _ = mock_clients
    embedder = NVIDIAEmbedding(api_key="BOGUS", dimensions=256)

    getattr(embedder, method_name)(arg)

    extra_body = mock_client.embeddings.create.call_args.kwargs["extra_body"]
    assert extra_body == {
        "input_type": input_type,
        "truncate": "NONE",
        "dimensions": 256,
    }


@pytest.mark.parametrize(
    ("method_name", "arg", "input_type"),
    [
        ("aget_query_embedding", "nvidia", "query"),
        ("aget_text_embedding", "nvidia", "passage"),
        ("aget_text_embedding_batch", ["nvidia"], "passage"),
    ],
)
@pytest.mark.asyncio
async def test_async_extra_body(mock_clients, method_name, arg, input_type) -> None:
    _, mock_aclient = mock_clients
    embedder = NVIDIAEmbedding(api_key="BOGUS", dimensions=256)

    await getattr(embedder, method_name)(arg)

    extra_body = mock_aclient.embeddings.create.call_args.kwargs["extra_body"]
    assert extra_body == {
        "input_type": input_type,
        "truncate": "NONE",
        "dimensions": 256,
    }
