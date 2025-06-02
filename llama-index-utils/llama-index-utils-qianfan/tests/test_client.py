import json
import asyncio
from unittest.mock import patch, MagicMock

import httpx

from llama_index.utils.qianfan.client import Client


@patch("httpx.Client")
def test_post(mock_client: httpx.Client):
    content = {"content": "Hello"}

    mock_response = MagicMock()
    mock_response.json.return_value = {"echo": content}
    mock_client.return_value.__enter__.return_value.send.return_value = mock_response

    client = Client("mock_access_key", "mock_secret_key")
    resp_dict = client.post(
        url="https://127.0.0.1/mock/echo", params={"param": "123"}, json=content
    )
    assert resp_dict == {"echo": content}

    mock_client.return_value.__enter__.return_value.send.assert_called_once()


@patch("httpx.AsyncClient")
def test_apost(mock_client: httpx.AsyncClient):
    content = {"content": "Hello"}

    mock_response = MagicMock()
    mock_response.json.return_value = {"echo": content}
    mock_client.return_value.__aenter__.return_value.send.return_value = mock_response

    async def async_process():
        client = Client("mock_access_key", "mock_secret_key")
        resp_dict = await client.apost(
            url="https://127.0.0.1/mock/echo", params={"param": "123"}, json=content
        )
        assert resp_dict == {"echo": content}

    asyncio.run(async_process())

    mock_client.return_value.__aenter__.return_value.send.assert_called_once()


@patch("httpx.Client")
def test_post_reply_stream(mock_client: httpx.Client):
    content = [{"content": "Hello"}, {"content": "world"}]
    reply_data = ["data: " + json.dumps(item) for item in content]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(reply_data)
    mock_client.return_value.__enter__.return_value.send.return_value = mock_response

    client = Client("mock_access_key", "mock_secret_key")
    resp_dict_iter = client.post_reply_stream(
        url="https://127.0.0.1/mock/echo", params={"param": "123"}, json=content
    )
    assert list(resp_dict_iter) == content

    mock_client.return_value.__enter__.return_value.send.assert_called_once()


@patch("httpx.AsyncClient")
def test_apost_reply_stream(mock_client: httpx.AsyncClient):
    content = [{"content": "Hello"}, {"content": "world"}]
    reply_data = ["data: " + json.dumps(item) for item in content]

    async def mock_async_gen():
        for part in reply_data:
            yield part

    mock_response = MagicMock()
    mock_response.aiter_lines.return_value = mock_async_gen()
    mock_client.return_value.__aenter__.return_value.send.return_value = mock_response

    async def async_process():
        client = Client("mock_access_key", "mock_secret_key")
        resp_dict_iter = client.apost_reply_stream(
            url="https://127.0.0.1/mock/echo", params={"param": "123"}, json=content
        )
        assert [part async for part in resp_dict_iter] == content

    asyncio.run(async_process())

    mock_client.return_value.__aenter__.return_value.send.assert_called_once()
