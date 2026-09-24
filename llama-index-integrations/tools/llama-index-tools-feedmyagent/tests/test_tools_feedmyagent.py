from unittest.mock import Mock, patch

from feedmyagent import Item
from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.feedmyagent import FeedMyAgentToolSpec


def _item(id_="1", title="Item one", summary="s1", url="https://example.com/1",
          tags=None, score=4.2):
    return Item(id=id_, title=title, summary=summary, url=url, tags=tags or ["security"], score=score)


def test_class():
    names_of_base_classes = [b.__name__ for b in FeedMyAgentToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_spec_functions():
    assert FeedMyAgentToolSpec.spec_functions == [
        "get_latest",
        "search_feed",
        "report_incident",
    ]


@patch("feedmyagent.FeedMyAgent")
def test_init_passes_api_key_and_base_url(mock_fma_cls):
    FeedMyAgentToolSpec(api_key="ask_test", base_url="https://custom.example.com")
    mock_fma_cls.assert_called_once_with(
        api_key="ask_test", base_url="https://custom.example.com", user_agent="feedmyagent-llamaindex/0.1"
    )


@patch("feedmyagent.FeedMyAgent")
def test_get_latest_maps_items_to_documents(mock_fma_cls):
    mock_client = Mock()
    mock_client.latest.return_value = [_item()]
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentToolSpec()
    results = tool.get_latest(tags=["security"], use_case="security", limit=1)

    mock_client.latest.assert_called_once_with(tags=["security"], use_case="security", limit=1)
    assert len(results) == 1
    assert all(isinstance(doc, Document) for doc in results)
    assert results[0].text == "Item one\n\ns1"
    assert results[0].metadata == {
        "id": "1",
        "url": "https://example.com/1",
        "tags": ["security"],
        "score": 4.2,
    }


@patch("feedmyagent.FeedMyAgent")
def test_get_latest_uses_title_only_when_no_summary(mock_fma_cls):
    mock_client = Mock()
    mock_client.latest.return_value = [_item(summary=None)]
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentToolSpec()
    results = tool.get_latest()

    assert results[0].text == "Item one"


@patch("feedmyagent.FeedMyAgent")
def test_search_feed_delegates_to_query(mock_fma_cls):
    mock_client = Mock()
    mock_client.query.return_value = [_item(id_="b", title="MCP auth bug")]
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentToolSpec()
    results = tool.search_feed("mcp auth", tags=["mcp"], limit=5)

    mock_client.query.assert_called_once_with("mcp auth", tags=["mcp"], limit=5)
    assert len(results) == 1
    assert results[0].metadata["id"] == "b"


@patch("feedmyagent.FeedMyAgent")
def test_report_incident_delegates_and_formats_confirmation(mock_fma_cls):
    mock_client = Mock()
    mock_client.report.return_value = _item(
        id_="pending-1", title="New prompt-injection technique",
        summary=None, url="https://example.com/incidents/xyz",
    )
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentToolSpec(api_key="ask_test")
    result = tool.report_incident(
        title="New prompt-injection technique",
        description="Observed a tool description embedding an instruction...",
        url="https://example.com/incidents/xyz",
    )

    mock_client.report.assert_called_once_with(
        "New prompt-injection technique",
        "Observed a tool description embedding an instruction...",
        "https://example.com/incidents/xyz",
    )
    assert result == (
        "Reported item 'pending-1': 'New prompt-injection technique' "
        "(https://example.com/incidents/xyz)"
    )


@patch("feedmyagent.FeedMyAgent")
def test_report_incident_url_defaults_to_none(mock_fma_cls):
    mock_client = Mock()
    mock_client.report.return_value = _item()
    mock_fma_cls.return_value = mock_client

    tool = FeedMyAgentToolSpec(api_key="ask_test")
    tool.report_incident(title="t", description="d")

    mock_client.report.assert_called_once_with("t", "d", None)
