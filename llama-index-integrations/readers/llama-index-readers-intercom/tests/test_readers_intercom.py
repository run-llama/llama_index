import json
from unittest import mock

import pytest
import requests

from llama_index.core.readers.base import BaseReader
from llama_index.readers.intercom import IntercomReader

ARTICLE = {
    "id": "1",
    "title": "Title",
    "url": "https://example.intercom.help/articles/1",
    "updated_at": 1700000000,
    "body": "<p>Body</p>",
}


def _response(status_code, body):
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(body).encode()
    response.url = "https://api.intercom.io/articles"
    return response


def test_class():
    names_of_base_classes = [b.__name__ for b in IntercomReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_data_follows_pages():
    pages = [
        _response(
            200,
            {
                "data": [ARTICLE],
                "pages": {"next": "https://api.intercom.io/articles?page=2"},
            },
        ),
        _response(200, {"data": [ARTICLE], "pages": {}}),
    ]
    with mock.patch("requests.get", side_effect=pages) as get:
        documents = IntercomReader("token").load_data()

    assert [doc.text for doc in documents] == ["Body", "Body"]
    assert get.call_args_list[1].args[0] == "https://api.intercom.io/articles?page=2"


def test_load_data_raises_on_invalid_token():
    error = {"type": "error.list", "errors": [{"code": "unauthorized"}]}
    with mock.patch("requests.get", return_value=_response(401, error)):
        with pytest.raises(requests.HTTPError):
            IntercomReader("bad-token").load_data()


def test_load_data_raises_on_error_after_first_page():
    pages = [
        _response(
            200,
            {
                "data": [ARTICLE],
                "pages": {"next": "https://api.intercom.io/articles?page=2"},
            },
        ),
        _response(429, {"type": "error.list", "errors": [{"code": "rate_limit"}]}),
    ]
    with mock.patch("requests.get", side_effect=pages):
        with pytest.raises(requests.HTTPError):
            IntercomReader("token").load_data()
