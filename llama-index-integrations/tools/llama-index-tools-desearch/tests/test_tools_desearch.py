from unittest.mock import AsyncMock, MagicMock, patch

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.desearch import DesearchToolSpec


def test_class():
    names_of_base_classes = [b.__name__ for b in DesearchToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_tool_list():
    tools = DesearchToolSpec(api_key="test").to_tool_list()
    assert [t.metadata.name for t in tools] == [
        "ai_search_tool",
        "twitter_search_tool",
        "web_search_tool",
    ]
    for t in tools:
        assert t.async_fn is not None


def _mock_client(**methods):
    client = MagicMock()
    for name, value in methods.items():
        setattr(client, name, AsyncMock(return_value=value))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx, client


def test_web_search_tool_calls_web_search():
    ctx, client = _mock_client(
        web_search={
            "data": [{"title": str(i), "link": "l", "snippet": "s"} for i in range(5)]
        }
    )
    with patch("llama_index.tools.desearch.base.Desearch", return_value=ctx):
        result = DesearchToolSpec(api_key="test").web_search_tool(
            query="q", num=2, start=3
        )
    client.web_search.assert_awaited_once_with(query="q", start=3)
    assert len(result) == 2


def test_twitter_search_tool_calls_x_search():
    ctx, client = _mock_client(x_search=[{"id": "1", "text": "hi"}])
    with patch("llama_index.tools.desearch.base.Desearch", return_value=ctx):
        result = DesearchToolSpec(api_key="test").twitter_search_tool(
            query="q", count=5
        )
    client.x_search.assert_awaited_once_with(query="q", sort="Top", count=5)
    assert result == [{"id": "1", "text": "hi"}]


def test_ai_search_tool_calls_ai_search():
    ctx, client = _mock_client(ai_search={"text": "summary"})
    with patch("llama_index.tools.desearch.base.Desearch", return_value=ctx):
        result = DesearchToolSpec(api_key="test").ai_search_tool(
            prompt="p", tool=["web"]
        )
    client.ai_search.assert_awaited_once_with(
        prompt="p", tools=["web"], date_filter="PAST_24_HOURS", count=None
    )
    assert result == {"text": "summary"}


def test_errors_are_returned_as_string():
    ctx, client = _mock_client()
    client.web_search = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("llama_index.tools.desearch.base.Desearch", return_value=ctx):
        assert DesearchToolSpec(api_key="test").web_search_tool(query="q") == "boom"
