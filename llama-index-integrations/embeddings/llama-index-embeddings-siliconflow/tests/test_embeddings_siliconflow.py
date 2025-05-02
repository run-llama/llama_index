import json
import pytest
import types
from requests import Response
from unittest import mock
from typing import Optional, Type
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding


class MockAsyncResponse:
    def __init__(self, json_data) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        ...

    async def __aenter__(self) -> "MockAsyncResponse":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[types.TracebackType],
    ) -> None:
        pass

    async def json(self) -> dict:
        return self._json_data


def test_embedding_class():
    emb = SiliconFlowEmbedding()
    assert isinstance(emb, BaseEmbedding)


def test_float_format_embedding():
    input_text = "..."
    mock_response = Response()
    mock_response._content = json.dumps(
        {
            "model": "<string>",
            "data": [{"object": "embedding", "embedding": [123], "index": 0}],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 123,
                "total_tokens": 123,
            },
        }
    ).encode("utf-8")
    embedding = SiliconFlowEmbedding(api_key="...")
    with mock.patch("requests.Session.post", return_value=mock_response) as mock_post:
        actual_result = embedding.get_query_embedding(input_text)
        expected_result = [123]

        assert actual_result == expected_result

        mock_post.assert_called_once_with(
            embedding.base_url,
            json={
                "model": embedding.model,
                "input": [input_text],
                "encoding_format": "float",
            },
            headers=embedding._headers,
        )


def test_base64_format_embedding():
    input_text = "..."
    mock_response = Response()
    mock_response._content = json.dumps(
        {
            "model": "<string>",
            "data": [{"object": "embedding", "embedding": "AAD2Qg==", "index": 0}],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 123,
                "total_tokens": 123,
            },
        }
    ).encode("utf-8")
    embedding = SiliconFlowEmbedding(api_key="...", encoding_format="base64")
    with mock.patch("requests.Session.post", return_value=mock_response) as mock_post:
        actual_result = embedding.get_query_embedding(input_text)
        expected_result = [123]

        assert actual_result == expected_result

        mock_post.assert_called_once_with(
            embedding.base_url,
            json={
                "model": embedding.model,
                "input": [input_text],
                "encoding_format": "base64",
            },
            headers=embedding._headers,
        )


@pytest.mark.asyncio()
async def test_float_format_embedding_async():
    input_text = "..."
    mock_response = MockAsyncResponse(
        json_data={
            "model": "<string>",
            "data": [{"object": "embedding", "embedding": [123], "index": 0}],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 123,
                "total_tokens": 123,
            },
        }
    )
    embedding = SiliconFlowEmbedding(api_key="...")
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=mock_response
    ) as mock_post:
        actual_result = await embedding.aget_query_embedding(input_text)
        expected_result = [123]

        assert actual_result == expected_result

        mock_post.assert_called_once_with(
            embedding.base_url,
            json={
                "model": embedding.model,
                "input": [input_text],
                "encoding_format": "float",
            },
            headers=embedding._headers,
        )
