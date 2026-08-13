import json

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.embeddings.nebius.base import DEFAULT_API_BASE


def test_embedding_class() -> None:
    embedding = NebiusEmbedding(model_name="BAAI/bge-en-icl", api_key="test-key")
    assert isinstance(embedding, BaseEmbedding)


def test_embedding_uses_token_factory_endpoint_and_openai_request_shape() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == f"{DEFAULT_API_BASE}/embeddings"
        assert request.headers["authorization"] == "Bearer test-key"

        payload = json.loads(request.content)
        assert payload == {
            "input": ["Everyone loves justice"],
            "model": "BAAI/bge-en-icl",
            "encoding_format": "base64",
        }

        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "model": payload["model"],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    embedding = NebiusEmbedding(api_key="test-key", http_client=http_client)

    result = embedding.get_text_embedding("Everyone loves justice")

    assert result == [0.1, 0.2]
