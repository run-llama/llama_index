import asyncio
import json
from typing import Callable

import httpx
import pytest

from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference


@pytest.fixture()
def transport(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    real_client = httpx.Client
    real_async_client = httpx.AsyncClient

    def install(status: int) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/embed"
            assert json.loads(request.content) == {
                "inputs": ["hello"],
                "truncate": True,
            }
            if status == 200:
                return httpx.Response(status, json=[[0.1, 0.2]])
            if status == 503:
                return httpx.Response(
                    status, text="Service unavailable", headers={"Retry-After": "5"}
                )
            return httpx.Response(
                status,
                json={"error": "embedding request rejected"},
                headers={"Retry-After": "5"},
            )

        mocked_transport = httpx.MockTransport(respond)
        monkeypatch.setattr(
            httpx, "Client", lambda: real_client(transport=mocked_transport)
        )
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda: real_async_client(transport=mocked_transport)
        )

    return install


@pytest.mark.parametrize("status", [413, 503])
@pytest.mark.parametrize("kind", ["query", "text", "text_batch"])
def test_http_errors_preserve_status_and_body(
    transport: Callable[[int], None], status: int, kind: str
) -> None:
    transport(status)
    embedder = TextEmbeddingsInference(model_name="some-model")

    with pytest.raises(httpx.HTTPStatusError) as error:
        if kind == "text_batch":
            embedder.get_text_embedding_batch(["hello"])
        else:
            getattr(embedder, f"get_{kind}_embedding")("hello")

    assert error.value.response.status_code == status
    assert (
        error.value.response.text == "Service unavailable"
        if status == 503
        else error.value.response.json() == {"error": "embedding request rejected"}
    )
    assert error.value.response.headers["Retry-After"] == "5"


@pytest.mark.parametrize("status", [413, 503])
@pytest.mark.parametrize("kind", ["query", "text", "text_batch"])
def test_async_http_errors_preserve_status_and_body(
    transport: Callable[[int], None], status: int, kind: str
) -> None:
    transport(status)
    embedder = TextEmbeddingsInference(model_name="some-model")

    with pytest.raises(httpx.HTTPStatusError) as error:
        if kind == "text_batch":
            asyncio.run(embedder.aget_text_embedding_batch(["hello"]))
        else:
            asyncio.run(getattr(embedder, f"aget_{kind}_embedding")("hello"))

    assert error.value.response.status_code == status
    assert (
        error.value.response.text == "Service unavailable"
        if status == 503
        else error.value.response.json() == {"error": "embedding request rejected"}
    )
    assert error.value.response.headers["Retry-After"] == "5"


@pytest.mark.parametrize("kind", ["query", "text", "text_batch"])
def test_successful_response(transport: Callable[[int], None], kind: str) -> None:
    transport(200)
    embedder = TextEmbeddingsInference(model_name="some-model")

    if kind == "text_batch":
        assert embedder.get_text_embedding_batch(["hello"]) == [[0.1, 0.2]]
    else:
        assert getattr(embedder, f"get_{kind}_embedding")("hello") == [0.1, 0.2]


@pytest.mark.parametrize("kind", ["query", "text", "text_batch"])
def test_async_successful_response(transport: Callable[[int], None], kind: str) -> None:
    transport(200)
    embedder = TextEmbeddingsInference(model_name="some-model")

    if kind == "text_batch":
        assert asyncio.run(embedder.aget_text_embedding_batch(["hello"])) == [
            [0.1, 0.2]
        ]
    else:
        assert asyncio.run(getattr(embedder, f"aget_{kind}_embedding")("hello")) == [
            0.1,
            0.2,
        ]
