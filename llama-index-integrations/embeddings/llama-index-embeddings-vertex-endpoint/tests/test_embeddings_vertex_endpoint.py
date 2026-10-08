import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.vertex_endpoint import VertexEndpointEmbedding


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in VertexEndpointEmbedding.__mro__]
    assert BaseEmbedding.__name__ in names_of_base_classes


@pytest.mark.parametrize("method", ["aget_query_embedding", "aget_text_embedding"])
def test_async_single_embedding(method, monkeypatch):
    client = SimpleNamespace(
        predict_async=AsyncMock(
            return_value=SimpleNamespace(predictions=[[[0.1, 0.2]]])
        )
    )
    monkeypatch.setattr(
        "llama_index.embeddings.vertex_endpoint.base.aiplatform.Endpoint",
        lambda **kwargs: client,
    )
    emb = VertexEndpointEmbedding(
        endpoint_id="test-endpoint",
        project_id="test-project",
        location="us-central1",
        model_kwargs={"test_parameter": "value"},
        endpoint_kwargs={"use_dedicated_endpoint": True},
        timeout=15.0,
    )

    assert asyncio.run(getattr(emb, method)("hello\nworld")) == [0.1, 0.2]
    client.predict_async.assert_awaited_once_with(
        instances=[{"inputs": "hello world"}],
        parameters={"test_parameter": "value"},
        use_dedicated_endpoint=True,
        timeout=15.0,
    )
