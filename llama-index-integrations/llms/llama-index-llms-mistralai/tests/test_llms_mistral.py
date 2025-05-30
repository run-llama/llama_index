import os
from unittest.mock import patch

from mistralai import ToolCall
import pytest

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.mistralai import MistralAI


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in MistralAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


@pytest.mark.skipif(
    os.environ.get("MISTRAL_API_KEY") is None, reason="MISTRAL_API_KEY not set"
)
def test_tool_required():
    llm = MistralAI()
    result = llm.chat_with_tools(
        tools=[search_tool],
        user_msg="What is the capital of France?",
        tool_required=True,
    )
    additional_kwargs = result.message.additional_kwargs
    assert "tool_calls" in additional_kwargs
    tool_calls = additional_kwargs["tool_calls"]
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.function.name == "search_tool"
    assert "query" in tool_call.function.arguments


@patch("mistralai.Mistral")
def test_prepare_chat_with_tools_tool_required(mock_mistral_client):
    """Test that tool_required is correctly passed to the API request when True."""
    # Mock the API key and client
    with patch("llama_index.llms.mistralai.base.get_from_param_or_env") as mock_get_env:
        mock_get_env.return_value = "fake-api-key"

        llm = MistralAI()

        # Test with tool_required=True
        result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

        assert result["tool_choice"] == "required"
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "search_tool"


@patch("mistralai.Mistral")
def test_prepare_chat_with_tools_tool_not_required(mock_mistral_client):
    """Test that tool_required is correctly passed to the API request when False."""
    # Mock the API key and client
    with patch("llama_index.llms.mistralai.base.get_from_param_or_env") as mock_get_env:
        mock_get_env.return_value = "fake-api-key"

        llm = MistralAI()

        # Test with tool_required=False (default)
        result = llm._prepare_chat_with_tools(
            tools=[search_tool],
        )

        assert result["tool_choice"] == "auto"
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "search_tool"
