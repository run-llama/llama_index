from unittest.mock import patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.requests import RequestsToolSpec, INVALID_URL_PROMPT


def test_class():
    names_of_base_classes = [b.__name__ for b in RequestsToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_replace_path_params():
    url = RequestsToolSpec._replace_path_params(
        "https://example.com/a/{a_id}/b/{b_id}/c", {"a_id": 1, "b_id": 2}
    )
    assert url == "https://example.com/a/1/b/2/c"


def test_invalid_url():
    spec = RequestsToolSpec()
    result = spec.get_request(url_template="/hi/there")
    assert result == INVALID_URL_PROMPT


@patch("requests.get")
def test_get(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {}

    llamaindex_tool_spec = RequestsToolSpec(domain_headers={}, timeout_seconds=2.2)
    result = llamaindex_tool_spec.get_request(
        "https://example.com/a/{a_id}/b/{b_id}/c",
        path_params={"a_id": "1", "b_id": "2"},
        query_params={"page": "1", "tags": "tag1"},
    )

    mock_get.assert_called_once_with(
        "https://example.com/a/1/b/2/c",
        headers={},
        timeout=2.2,
        params={"page": "1", "tags": "tag1"},
    )
