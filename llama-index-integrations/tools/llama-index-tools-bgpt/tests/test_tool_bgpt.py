from unittest.mock import MagicMock, patch

import requests
from llama_index.tools.bgpt import BGPTToolSpec


def test_class_initialization() -> None:
    """Test that the BGPTToolSpec initializes correctly."""
    # Without API key
    tool = BGPTToolSpec()
    assert tool.api_key is None
    assert tool.base_url == "https://bgpt.pro/api/mcp-search"

    # With API key
    tool_with_key = BGPTToolSpec(api_key="test_key")
    assert tool_with_key.api_key == "test_key"


def test_tool_functions_list() -> None:
    """Test that the correct functions are exposed to the agent."""
    tool = BGPTToolSpec()
    assert "search_literature" in tool.spec_functions


@patch("requests.post")
def test_search_literature_success(mock_post: MagicMock) -> None:
    """Test successful API call with mocked response."""
    # Setup the fake response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [{"title": "Fake Sleep Study", "paper_limitations_and_biases": "None"}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    # Run the tool
    tool = BGPTToolSpec()
    results = tool.search_literature("sleep and memory", num_results=1)

    # Assertions
    assert len(results) == 1
    assert results[0]["title"] == "Fake Sleep Study"
    mock_post.assert_called_once()


@patch("requests.post")
def test_search_literature_network_error(mock_post: MagicMock) -> None:
    """Test handling of network/HTTP errors."""
    # Setup the mock to throw an error
    mock_post.side_effect = requests.exceptions.HTTPError("404 Client Error")

    tool = BGPTToolSpec()
    results = tool.search_literature("sleep and memory")

    # Assertions
    assert len(results) == 1
    assert "error" in results[0]
    assert "404 Client Error" in results[0]["error"]