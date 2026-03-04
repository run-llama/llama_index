from unittest.mock import MagicMock, patch
from llama_index.core.tools import FunctionTool
from llama_index.llms.ibm import WatsonxLLM


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


calculator_tool = FunctionTool.from_defaults(
    fn=lambda a, b: a + b,
    name="calculator",
    description="A tool for calculating the sum of two numbers",
)

search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


@patch("llama_index.llms.ibm.base.ModelInference")
@patch("llama_index.llms.ibm.base.resolve_watsonx_credentials")
def test_prepare_chat_with_tools_tool_required_single_tool(
    mock_resolve_credentials, MockModelInference
):
    """Test that tool_required selects the first tool when there's only one tool."""
    mock_resolve_credentials.return_value = {}
    mock_instance = MockModelInference.return_value

    llm = WatsonxLLM(
        model_id="test_model",
        project_id="test_project",
        apikey="test_apikey",
        url="https://test-url.com",
        api_client=MagicMock(),  # Use mock client to bypass credential checks
    )

    # Test with tool_required=True and a single tool
    result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

    assert "tool_choice" in result
    assert result["tool_choice"]["type"] == "function"
    assert result["tool_choice"]["function"]["name"] == "search_tool"
    assert len(result["tools"]) == 1
    assert result["tools"][0]["function"]["name"] == "search_tool"


@patch("llama_index.llms.ibm.base.ModelInference")
@patch("llama_index.llms.ibm.base.resolve_watsonx_credentials")
def test_prepare_chat_with_tools_tool_required_multiple_tools(
    mock_resolve_credentials, MockModelInference
):
    """Test that tool_required selects the first tool when there are multiple tools."""
    mock_resolve_credentials.return_value = {}
    mock_instance = MockModelInference.return_value

    llm = WatsonxLLM(
        model_id="test_model",
        project_id="test_project",
        apikey="test_apikey",
        url="https://test-url.com",
        api_client=MagicMock(),  # Use mock client to bypass credential checks
    )

    # Test with tool_required=True and multiple tools
    result = llm._prepare_chat_with_tools(
        tools=[search_tool, calculator_tool], tool_required=True
    )

    assert "tool_choice" in result
    assert result["tool_choice"]["type"] == "function"
    # It should select the first tool when tool_required=True
    assert result["tool_choice"]["function"]["name"] == "search_tool"
    assert len(result["tools"]) == 2


@patch("llama_index.llms.ibm.base.ModelInference")
@patch("llama_index.llms.ibm.base.resolve_watsonx_credentials")
def test_prepare_chat_with_tools_tool_not_required(
    mock_resolve_credentials, MockModelInference
):
    """Test that tool_required=False doesn't specify a tool choice."""
    mock_resolve_credentials.return_value = {}
    mock_instance = MockModelInference.return_value

    llm = WatsonxLLM(
        model_id="test_model",
        project_id="test_project",
        apikey="test_apikey",
        url="https://test-url.com",
        api_client=MagicMock(),  # Use mock client to bypass credential checks
    )

    # Test with tool_required=False (default)
    result = llm._prepare_chat_with_tools(
        tools=[search_tool, calculator_tool],
    )

    # When tool_required=False, there should be no tool_choice specified
    assert "tool_choice" not in result
    assert len(result["tools"]) == 2


@patch("llama_index.llms.ibm.base.ModelInference")
@patch("llama_index.llms.ibm.base.resolve_watsonx_credentials")
def test_prepare_chat_with_tools_explicit_tool_choice(
    mock_resolve_credentials, MockModelInference
):
    """Test that an explicit tool_choice overrides tool_required."""
    mock_resolve_credentials.return_value = {}
    mock_instance = MockModelInference.return_value

    llm = WatsonxLLM(
        model_id="test_model",
        project_id="test_project",
        apikey="test_apikey",
        url="https://test-url.com",
        api_client=MagicMock(),  # Use mock client to bypass credential checks
    )

    # Test with explicit tool_choice parameter, which should override tool_required
    result = llm._prepare_chat_with_tools(
        tools=[search_tool, calculator_tool],
        tool_required=True,
        tool_choice="calculator",
    )

    assert "tool_choice" in result
    assert result["tool_choice"]["type"] == "function"
    assert result["tool_choice"]["function"]["name"] == "calculator"
    assert len(result["tools"]) == 2


@patch("llama_index.llms.ibm.base.ModelInference")
@patch("llama_index.llms.ibm.base.resolve_watsonx_credentials")
def test_prepare_chat_with_tools_no_tools(mock_resolve_credentials, MockModelInference):
    """Test that tool_required=True with no tools doesn't add a tool_choice."""
    mock_resolve_credentials.return_value = {}
    mock_instance = MockModelInference.return_value

    llm = WatsonxLLM(
        model_id="test_model",
        project_id="test_project",
        apikey="test_apikey",
        url="https://test-url.com",
        api_client=MagicMock(),  # Use mock client to bypass credential checks
    )

    # Test with tool_required=True but no tools
    result = llm._prepare_chat_with_tools(tools=[], tool_required=True)

    # When there are no tools, tool_choice should not be specified even if tool_required=True
    assert "tool_choice" not in result
    assert result["tools"] is None
