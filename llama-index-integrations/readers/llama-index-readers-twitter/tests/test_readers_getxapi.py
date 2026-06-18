from typing import Any
from unittest.mock import MagicMock, patch

from llama_index.core.readers.base import BasePydanticReader
from llama_index.readers.twitter import GetXAPISearchReader


def test_class():
    names_of_base_classes = [b.__name__ for b in GetXAPISearchReader.__mro__]
    assert BasePydanticReader.__name__ in names_of_base_classes


def test_defaults():
    reader = GetXAPISearchReader(bearer_token="token-abc")
    assert reader.bearer_token == "token-abc"
    assert reader.base_url == "https://api.getxapi.com"
    assert reader.timeout == 30.0
    assert reader._build_url() == "https://api.getxapi.com/twitter/tweet/advanced_search"
    headers = reader._headers()
    assert headers["Authorization"] == "Bearer token-abc"
    assert headers["Accept"] == "application/json"


def test_custom_base_url():
    reader = GetXAPISearchReader(
        bearer_token="t", base_url="https://staging.getxapi.com/"
    )
    assert reader._build_url() == "https://staging.getxapi.com/twitter/tweet/advanced_search"


class _MockResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


def test_load_data_parses_documents():
    reader = GetXAPISearchReader(bearer_token="t")
    payload = {
        "data": [
            {
                "id": "12345",
                "text": "hello world",
                "author": {"username": "alice", "name": "Alice"},
                "created_at": "2026-01-01T00:00:00Z",
                "like_count": 7,
                "retweet_count": 2,
                "reply_count": 1,
            },
            {
                "id": "67890",
                "text": "second tweet",
                "author": {"username": "bob", "name": "Bob"},
            },
        ]
    }

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.get.return_value = _MockResponse(payload)

    with patch("httpx.Client", return_value=mock_client):
        docs = reader.load_data(query="llamaindex", limit=25)

    assert len(docs) == 2
    assert docs[0].text == "hello world"
    assert docs[0].metadata["author_username"] == "alice"
    assert docs[0].metadata["url"] == "https://x.com/alice/status/12345"
    assert docs[0].metadata["like_count"] == 7
    assert docs[1].text == "second tweet"

    mock_client.get.assert_called_once()
    call = mock_client.get.call_args
    assert call.args[0] == "https://api.getxapi.com/twitter/tweet/advanced_search"
    assert call.kwargs["headers"]["Authorization"] == "Bearer t"
    assert call.kwargs["params"] == {"q": "llamaindex", "limit": 25}


def test_load_data_handles_list_payload():
    reader = GetXAPISearchReader(bearer_token="t")
    payload = [{"id": "1", "text": "x", "author": {"username": "u"}}]

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.get.return_value = _MockResponse(payload)

    with patch("httpx.Client", return_value=mock_client):
        docs = reader.load_data(query="q")

    assert len(docs) == 1
    assert docs[0].text == "x"
