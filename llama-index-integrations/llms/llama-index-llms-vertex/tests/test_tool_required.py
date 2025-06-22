from unittest.mock import Mock, patch

from llama_index.core.tools import FunctionTool
from llama_index.llms.vertex import Vertex
from vertexai.generative_models import ToolConfig


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


def calculate(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)

calculator_tool = FunctionTool.from_defaults(
    fn=calculate,
    name="calculator",
    description="A tool for calculating the sum of two numbers",
)


class TestVertexToolRequired:
    """Test suite for Vertex AI tool_required functionality."""

    @patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
    def test_to_function_calling_config_tool_required_true(self, mock_create_client):
        """Test that _to_function_calling_config correctly sets mode to ANY when tool_required=True."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")
        config = llm._to_function_calling_config(tool_required=True)

        # Check config mode through string representation since direct attribute access is problematic
        config_str = str(config)
        assert isinstance(config, ToolConfig)
        assert "mode: ANY" in config_str

    @patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
    def test_to_function_calling_config_tool_required_false(self, mock_create_client):
        """Test that _to_function_calling_config correctly sets mode to AUTO when tool_required=False."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")
        config = llm._to_function_calling_config(tool_required=False)

        # Check config mode through string representation
        config_str = str(config)
        assert isinstance(config, ToolConfig)
        assert "mode: AUTO" in config_str

    @patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
    def test_prepare_chat_with_tools_tool_required_gemini(self, mock_create_client):
        """Test that tool_required is correctly passed to tool_config for Gemini models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")

        # Test with tool_required=True
        result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

        # Verify tool_config mode using string representation
        tool_config_str = str(result["tool_config"])
        assert "tool_config" in result
        assert isinstance(result["tool_config"], ToolConfig)
        assert "mode: ANY" in tool_config_str
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search_tool"

    @patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
    def test_prepare_chat_with_tools_tool_not_required_gemini(self, mock_create_client):
        """Test that tool_required=False correctly sets mode to AUTO for Gemini models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")

        # Test with tool_required=False
        result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=False)

        # Verify tool_config mode using string representation
        tool_config_str = str(result["tool_config"])
        assert "tool_config" in result
        assert isinstance(result["tool_config"], ToolConfig)
        assert "mode: AUTO" in tool_config_str
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search_tool"

    @patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
    def test_prepare_chat_with_tools_default_behavior_gemini(self, mock_create_client):
        """Test default behavior when tool_required is not specified for Gemini models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")

        # Test without specifying tool_required (should default to False)
        result = llm._prepare_chat_with_tools(tools=[search_tool])

        # Verify tool_config mode using string representation
        tool_config_str = str(result["tool_config"])
        assert "tool_config" in result
        assert isinstance(result["tool_config"], ToolConfig)
        # Should default to AUTO when tool_required=False (default)
        assert "mode: AUTO" in tool_config_str
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search_tool"

    @patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
    def test_prepare_chat_with_tools_multiple_tools_gemini(self, mock_create_client):
        """Test tool_required with multiple tools for Gemini models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")

        # Test with tool_required=True and multiple tools
        result = llm._prepare_chat_with_tools(
            tools=[search_tool, calculator_tool], tool_required=True
        )

        # Verify tool_config mode using string representation
        tool_config_str = str(result["tool_config"])
        assert "tool_config" in result
        assert isinstance(result["tool_config"], ToolConfig)
        assert "mode: ANY" in tool_config_str
        assert len(result["tools"]) == 2
        tool_names = [tool["name"] for tool in result["tools"]]
        assert "search_tool" in tool_names
        assert "calculator" in tool_names

    @patch("vertexai.language_models.TextGenerationModel.from_pretrained")
    @patch("vertexai.language_models.ChatModel.from_pretrained")
    def test_prepare_chat_with_tools_non_gemini_no_tool_config(
        self, mock_chat_from_pretrained, mock_text_from_pretrained
    ):
        """Test that non-Gemini models don't include tool_config regardless of tool_required."""
        mock_chat_client = Mock()
        mock_text_client = Mock()
        mock_chat_from_pretrained.return_value = mock_chat_client
        mock_text_from_pretrained.return_value = mock_text_client

        # Use a non-Gemini model name
        llm = Vertex(model="text-bison", project="test-project")

        # Test with tool_required=True for non-Gemini model
        result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=True)

        # Non-Gemini models should not have tool_config
        assert "tool_config" not in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search_tool"

        # Test with tool_required=False for non-Gemini model
        result = llm._prepare_chat_with_tools(tools=[search_tool], tool_required=False)

        # Non-Gemini models should not have tool_config
        assert "tool_config" not in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search_tool"

    @patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
    def test_prepare_chat_with_tools_no_tools_gemini(self, mock_create_client):
        """Test tool behavior when no tools are provided for Gemini models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")

        # Test with tool_required=True but no tools
        result = llm._prepare_chat_with_tools(tools=[], tool_required=True)

        # Verify tool_config mode using string representation
        tool_config_str = str(result["tool_config"])
        # The current implementation still includes tool_config even with no tools if tool_required=True
        assert "tool_config" in result
        assert isinstance(result["tool_config"], ToolConfig)
        assert "mode: ANY" in tool_config_str
        assert result["tools"] is None

    @patch("llama_index.llms.vertex.gemini_utils.create_gemini_client")
    def test_prepare_chat_with_tools_with_kwargs_gemini(self, mock_create_client):
        """Test that additional kwargs are preserved when using tool_required for Gemini models."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        llm = Vertex(model="gemini-pro", project="test-project")

        # Test with tool_required=True and additional kwargs
        result = llm._prepare_chat_with_tools(
            tools=[search_tool], tool_required=True, temperature=0.7, max_tokens=1000
        )

        # Verify tool_config mode using string representation
        tool_config_str = str(result["tool_config"])
        assert "tool_config" in result
        assert isinstance(result["tool_config"], ToolConfig)
        assert "mode: ANY" in tool_config_str
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search_tool"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000
