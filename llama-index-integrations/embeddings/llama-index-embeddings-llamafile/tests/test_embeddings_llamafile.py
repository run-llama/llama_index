from json import dumps as json_dumps

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.llamafile import LlamafileEmbedding
from pytest import MonkeyPatch

BASE_URL = "http://llamafile-host:8080"


def test_embedding_class():
    emb = LlamafileEmbedding()
    assert isinstance(emb, BaseEmbedding)


def test_init():
    emb = LlamafileEmbedding()
    assert emb.request_timeout == 30.0
    emb = LlamafileEmbedding(request_timeout=10.0)
    assert emb.request_timeout == 10.0


def mock_embedding_response() -> httpx.Response:
    content = json_dumps({"embedding": [0.1, 0.2, 0.1, 0.2]})
    return httpx.Response(
        status_code=200,
        content=content,
        request=httpx.Request(method="POST", url=BASE_URL),
    )


def test_get_text_embedding(monkeypatch: MonkeyPatch):
    embedder = LlamafileEmbedding(base_url=BASE_URL)

    def mock_post(self, url, headers, json):  # type: ignore[no-untyped-def]
        assert url == f"{BASE_URL}/embedding"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {
            "content": "Test prompt",
        }
        return mock_embedding_response()

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    out = embedder.get_text_embedding("Test prompt")
    assert out == [0.1, 0.2, 0.1, 0.2]


def mock_embeddings_batched_response() -> httpx.Response:
    content = json_dumps(
        {
            "results": [
                {"embedding": [0.1, 0.2, 0.1, 0.2]},
                {"embedding": [0.3, 0.3, 0.3, 0.3]},
            ]
        }
    )
    return httpx.Response(
        status_code=200,
        content=content,
        request=httpx.Request(method="POST", url=BASE_URL),
    )


def test_get_text_embeddings(monkeypatch: MonkeyPatch):
    embedder = LlamafileEmbedding(base_url=BASE_URL)

    def mock_post(self, url, headers, json):  # type: ignore[no-untyped-def]
        assert url == f"{BASE_URL}/embedding"
        assert headers == {
            "Content-Type": "application/json",
        }
        assert json == {"content": ["Test prompt", "Another test prompt"]}
        return mock_embeddings_batched_response()

    monkeypatch.setattr(httpx.Client, "post", mock_post)

    out = embedder.get_text_embedding_batch(["Test prompt", "Another test prompt"])
    assert len(out) == 2
    assert out[0] == [0.1, 0.2, 0.1, 0.2]
    assert out[1] == [0.3, 0.3, 0.3, 0.3]
