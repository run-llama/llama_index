from unittest.mock import MagicMock, patch

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.brave_search import BraveSearchToolSpec
from llama_index.tools.brave_search.base import DEFAULT_TIMEOUT


def test_class():
    names_of_base_classes = [b.__name__ for b in BraveSearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


@patch("llama_index.tools.brave_search.base.requests.get")
def test_brave_search_passes_default_timeout(mock_get):
    mock_response = MagicMock()
    mock_response.text = "{}"
    mock_get.return_value = mock_response

    tool = BraveSearchToolSpec(api_key="test-key")
    results = tool.brave_search("hello world")

    assert mock_get.call_count == 1
    assert mock_get.call_args.kwargs["timeout"] == DEFAULT_TIMEOUT
    assert all(isinstance(doc, Document) for doc in results)


@patch("llama_index.tools.brave_search.base.requests.get")
def test_brave_search_passes_custom_timeout(mock_get):
    mock_response = MagicMock()
    mock_response.text = "{}"
    mock_get.return_value = mock_response

    tool = BraveSearchToolSpec(api_key="test-key", timeout=2.5)
    tool.brave_search("hello world")

    assert mock_get.call_args.kwargs["timeout"] == 2.5
