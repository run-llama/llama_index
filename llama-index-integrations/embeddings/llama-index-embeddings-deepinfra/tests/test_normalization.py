import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "method_name",
    [
        "get_query_embedding",
        "get_text_embedding",
        "get_text_embedding_batch",
        "aget_query_embedding",
        "aget_text_embedding",
        "aget_text_embedding_batch",
    ],
)
def test_normalization_applies_to_public_embedding_methods(
    monkeypatch: pytest.MonkeyPatch, normalize: bool, method_name: str
) -> None:
    is_batch = method_name.endswith("_batch")
    vectors = [[3.0, 4.0], [-3.0, 4.0], [0.0, 0.0]] if is_batch else [[3.0, 4.0]]
    response = MagicMock()
    response.json.return_value = {"embeddings": vectors}
    post = MagicMock(return_value=response)
    monkeypatch.setattr("llama_index.embeddings.deepinfra.base.requests.post", post)

    async_response = MagicMock()
    async_response.json = AsyncMock(return_value={"embeddings": vectors})
    session = MagicMock()
    session.__aenter__.return_value = session
    session.post.return_value.__aenter__.return_value = async_response
    monkeypatch.setattr(
        "llama_index.embeddings.deepinfra.base.aiohttp.ClientSession",
        MagicMock(return_value=session),
    )

    model = DeepInfraEmbeddingModel(api_token="test-token", normalize=normalize)
    inputs = ["first", "second", "zero"] if is_batch else "first"
    result = getattr(model, method_name)(inputs)
    if method_name.startswith("aget_"):
        result = asyncio.run(result)
        session.post.assert_called_once()
        assert session.post.call_args.kwargs["json"]["inputs"] == (
            inputs if is_batch else [inputs]
        )
    else:
        post.assert_called_once()
        assert post.call_args.kwargs["json"]["inputs"] == (
            inputs if is_batch else [inputs]
        )

    expected = (
        [[0.6, 0.8], [-0.6, 0.8], [0.0, 0.0]][: len(vectors)] if normalize else vectors
    )
    actual = result if is_batch else [result]
    assert len(actual) == len(expected)
    for embedding, expected_embedding in zip(actual, expected):
        assert embedding == pytest.approx(expected_embedding)
