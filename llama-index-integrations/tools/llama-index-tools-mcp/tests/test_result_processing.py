import pytest

from llama_index.core.base.llms.types import TextBlock, ImageBlock
from llama_index.tools.mcp.base import McpToolSpec
from mcp import types as mcp_types


def test_process_text_content():
    """Test processing a CallToolResult with text content."""
    result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="Hello, World!")],
        isError=False,
    )
    blocks = McpToolSpec._process_call_result(result)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "Hello, World!"


def test_process_multiple_text_content():
    """Test processing a CallToolResult with multiple text content items."""
    result = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="Line 1"),
            mcp_types.TextContent(type="text", text="Line 2"),
        ],
        isError=False,
    )
    blocks = McpToolSpec._process_call_result(result)

    assert len(blocks) == 2
    assert blocks[0].text == "Line 1"
    assert blocks[1].text == "Line 2"


def test_process_image_content():
    """Test processing a CallToolResult with image content."""
    result = mcp_types.CallToolResult(
        content=[
            mcp_types.ImageContent(
                type="image", data="base64data", mimeType="image/png"
            )
        ],
        isError=False,
    )
    blocks = McpToolSpec._process_call_result(result)

    assert len(blocks) == 1
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].image_mimetype == "image/png"


def test_process_mixed_content():
    """Test processing a CallToolResult with mixed content types."""
    result = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type="text", text="Here's the chart:"),
            mcp_types.ImageContent(
                type="image", data="chartdata", mimeType="image/png"
            ),
        ],
        isError=False,
    )
    blocks = McpToolSpec._process_call_result(result)

    assert len(blocks) == 2
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "Here's the chart:"
    assert isinstance(blocks[1], ImageBlock)


def test_process_error_result():
    """Test processing a CallToolResult with isError=True."""
    result = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text="Something went wrong")],
        isError=True,
    )
    blocks = McpToolSpec._process_call_result(result)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert "MCP tool error" in blocks[0].text
    assert "Something went wrong" in blocks[0].text


def test_process_empty_content():
    """Test processing a CallToolResult with empty content."""
    result = mcp_types.CallToolResult(content=[], isError=False)
    blocks = McpToolSpec._process_call_result(result)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == ""
