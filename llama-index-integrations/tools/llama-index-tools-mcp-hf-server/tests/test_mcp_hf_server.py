"""Tests for the Hugging Face MCP Server tool spec."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.tools.mcp_hf_server import (
    HF_BUILT_IN_TOOLS,
    HF_MCP_URL,
    HF_MCP_URL_AUTH,
    HfMcpToolSpec,
)


def test_class_hierarchy():
    """HfMcpToolSpec should inherit from McpToolSpec and BaseToolSpec."""
    names_of_base_classes = [b.__name__ for b in HfMcpToolSpec.__mro__]
    assert "McpToolSpec" in names_of_base_classes
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_constants():
    """Verify that the module constants are defined correctly."""
    assert HF_MCP_URL == "https://huggingface.co/mcp"
    assert HF_MCP_URL_AUTH == "https://huggingface.co/mcp?login"
    assert isinstance(HF_BUILT_IN_TOOLS, list)
    assert len(HF_BUILT_IN_TOOLS) > 0
    assert "hf_model_search" in HF_BUILT_IN_TOOLS
    assert "hf_dataset_search" in HF_BUILT_IN_TOOLS
    assert "hf_paper_search" in HF_BUILT_IN_TOOLS
    assert "hf_space_search" in HF_BUILT_IN_TOOLS
    assert "hf_doc_search" in HF_BUILT_IN_TOOLS


def test_default_client_creation():
    """HfMcpToolSpec should create a BasicMCPClient when none is provided."""
    tool_spec = HfMcpToolSpec()
    assert isinstance(tool_spec.client, BasicMCPClient)
    assert tool_spec.client.command_or_url == HF_MCP_URL_AUTH


def test_default_client_no_auth():
    """When use_auth_url is False, it should use the non-auth URL."""
    tool_spec = HfMcpToolSpec(use_auth_url=False)
    assert isinstance(tool_spec.client, BasicMCPClient)
    assert tool_spec.client.command_or_url == HF_MCP_URL


def test_custom_client():
    """HfMcpToolSpec should accept a custom client."""
    custom_client = BasicMCPClient("https://custom.endpoint.com/mcp")
    tool_spec = HfMcpToolSpec(client=custom_client)
    assert tool_spec.client is custom_client
    assert tool_spec.client.command_or_url == "https://custom.endpoint.com/mcp"


def test_hf_token_in_headers():
    """When hf_token is provided, it should be set as an Authorization header."""
    tool_spec = HfMcpToolSpec(hf_token="hf_test_token_123")
    assert tool_spec.client.headers is not None
    assert tool_spec.client.headers["Authorization"] == "Bearer hf_test_token_123"


def test_no_token_no_headers():
    """When no hf_token is provided, headers should be None."""
    tool_spec = HfMcpToolSpec()
    assert tool_spec.client.headers is None


def test_custom_timeout():
    """Custom timeout should be passed to the client."""
    tool_spec = HfMcpToolSpec(timeout=120)
    assert tool_spec.client.timeout == 120


def test_default_timeout():
    """Default timeout should be 60 seconds."""
    tool_spec = HfMcpToolSpec()
    assert tool_spec.client.timeout == 60


def test_allowed_tools_passed():
    """allowed_tools should be correctly stored on the tool spec."""
    allowed = ["hf_model_search", "hf_dataset_search"]
    tool_spec = HfMcpToolSpec(allowed_tools=allowed)
    assert tool_spec.allowed_tools == allowed


def test_include_resources():
    """include_resources should be correctly stored on the tool spec."""
    tool_spec = HfMcpToolSpec(include_resources=True)
    assert tool_spec.include_resources is True

    tool_spec = HfMcpToolSpec(include_resources=False)
    assert tool_spec.include_resources is False


@pytest.mark.asyncio
async def test_to_tool_list_async_with_mock():
    """Test that to_tool_list_async returns FunctionTool objects."""
    mock_tool = MagicMock()
    mock_tool.name = "hf_model_search"
    mock_tool.description = "Search for ML models"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]

    mock_client = AsyncMock(spec=BasicMCPClient)
    mock_client.list_tools = AsyncMock(return_value=mock_response)

    tool_spec = HfMcpToolSpec(client=mock_client)
    tools = await tool_spec.to_tool_list_async()

    assert len(tools) == 1
    assert isinstance(tools[0], FunctionTool)
    assert tools[0].metadata.name == "hf_model_search"
    assert tools[0].metadata.description == "Search for ML models"


@pytest.mark.asyncio
async def test_allowed_tools_filtering_with_mock():
    """Test that allowed_tools correctly filters the tool list."""
    mock_tools = []
    for name in ["hf_model_search", "hf_dataset_search", "hf_paper_search"]:
        mock_tool = MagicMock()
        mock_tool.name = name
        mock_tool.description = f"Tool: {name}"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        mock_tools.append(mock_tool)

    mock_response = MagicMock()
    mock_response.tools = mock_tools

    mock_client = AsyncMock(spec=BasicMCPClient)
    mock_client.list_tools = AsyncMock(return_value=mock_response)

    tool_spec = HfMcpToolSpec(
        client=mock_client,
        allowed_tools=["hf_model_search"],
    )
    tools = await tool_spec.to_tool_list_async()

    assert len(tools) == 1
    assert tools[0].metadata.name == "hf_model_search"


@pytest.mark.asyncio
async def test_aget_hf_tools_with_mock():
    """Test the aget_hf_tools convenience function."""
    from llama_index.tools.mcp_hf_server.base import aget_hf_tools

    mock_tool = MagicMock()
    mock_tool.name = "hf_model_search"
    mock_tool.description = "Search for ML models"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]

    with patch.object(
        BasicMCPClient, "list_tools", new_callable=AsyncMock, return_value=mock_response
    ):
        tool_spec = HfMcpToolSpec()
        tools = await tool_spec.to_tool_list_async()
        assert len(tools) == 1
