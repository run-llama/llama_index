from unittest.mock import MagicMock
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.oci_genai import OCIGenAI


def test_oci_genai_embedding_class():
    names_of_base_classes = [b.__name__ for b in OCIGenAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


# Shared test tool for tool_required tests
def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


def test_prepare_chat_with_tools_tool_required():
    """Test that tool_required is correctly passed to the API request when True."""
    # Mock the client to avoid authentication issues
    mock_client = MagicMock()

    llm = OCIGenAI(
        model="cohere.command-r-16k",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="test_compartment_id",
        client=mock_client,
    )

    # Test with tool_required=True
    result = llm._prepare_chat_with_tools(
        tools=[search_tool], user_msg="Test message", tool_required=True
    )

    assert result["tool_choice"] == "REQUIRED"
    assert len(result["tools"]) == 1
    # CohereTool objects have a `name` attribute directly
    assert result["tools"][0].name == "search_tool"


def test_prepare_chat_with_tools_tool_not_required():
    """Test that tool_required is correctly passed to the API request when False."""
    # Mock the client to avoid authentication issues
    mock_client = MagicMock()

    llm = OCIGenAI(
        model="cohere.command-r-16k",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="test_compartment_id",
        client=mock_client,
    )

    # Test with tool_required=False (default)
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
        user_msg="Test message",
    )

    # When tool_required is False, tool_choice should not be included
    assert "tool_choice" not in result
    assert len(result["tools"]) == 1
    # CohereTool objects have a `name` attribute directly
    assert result["tools"][0].name == "search_tool"
