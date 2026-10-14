import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel


@pytest.mark.parametrize("status", [200, 429, 503])
@pytest.mark.parametrize(
    "method_name",
    ["aget_query_embedding", "aget_text_embedding", "aget_text_embedding_batch"],
)
def test_async_embedding_preserves_http_errors(
    monkeypatch: pytest.MonkeyPatch, status: int, method_name: str
) -> None:
    model = DeepInfraEmbeddingModel(api_token="test-token")
    request_info = aiohttp.RequestInfo(
        url=URL(model.get_url()),
        method="POST",
        headers=CIMultiDictProxy(CIMultiDict()),
        real_url=URL(model.get_url()),
    )
    response = MagicMock()
    response.json = AsyncMock(
        return_value={"embeddings": [[3.0, 4.0]]}
        if status == 200
        else {"error": "upstream request failed"}
    )
    error = aiohttp.ClientResponseError(
        request_info=request_info,
        history=(),
        status=status,
        message="upstream request failed",
        headers=CIMultiDictProxy(CIMultiDict({"Retry-After": "10"})),
    )
    if status != 200:
        response.raise_for_status.side_effect = error
    session = MagicMock()
    session.__aenter__.return_value = session
    session.post.return_value.__aenter__.return_value = response
    monkeypatch.setattr(
        "llama_index.embeddings.deepinfra.base.aiohttp.ClientSession",
        MagicMock(return_value=session),
    )

    is_batch = method_name.endswith("_batch")
    call = getattr(model, method_name)(["text"] if is_batch else "text")
    if status == 200:
        assert asyncio.run(call) == ([[3.0, 4.0]] if is_batch else [3.0, 4.0])
        response.json.assert_awaited_once()
    else:
        with pytest.raises(aiohttp.ClientResponseError) as caught:
            asyncio.run(call)
        assert caught.value is error
        assert caught.value.status == status
        assert caught.value.headers is not None
        assert caught.value.headers["Retry-After"] == "10"
        response.json.assert_not_awaited()
    session.post.assert_called_once()
