import os
import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Path to the test server script - adjust as needed
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")


@pytest.fixture(scope="session")
def client() -> BasicMCPClient:
    """Create a basic MCP client connected to the test server."""
    return BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)


def test_class():
    names_of_base_classes = [b.__name__ for b in McpToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_get_tools(client: BasicMCPClient):
    tool_spec = McpToolSpec(client)
    tools = tool_spec.to_tool_list()
    assert len(tools) > 0

    tool_spec = McpToolSpec(client, include_resources=True)
    tools_plus_resources = tool_spec.to_tool_list()
    assert len(tools_plus_resources) > len(tools)


@pytest.mark.asyncio
async def test_get_tools_async(client: BasicMCPClient):
    tool_spec = McpToolSpec(client)
    tools = await tool_spec.to_tool_list_async()
    assert len(tools) > 0

    tool_spec = McpToolSpec(client, include_resources=True)
    tools_plus_resources = await tool_spec.to_tool_list_async()
    assert len(tools_plus_resources) > len(tools)


def test_get_single_tool(client: BasicMCPClient):
    tool_spec = McpToolSpec(client, allowed_tools=["echo"])

    tools = tool_spec.to_tool_list()
    assert len(tools) == 1
    assert tools[0].metadata.name == "echo"


@pytest.mark.asyncio
async def test_get_single_tool_async(client: BasicMCPClient):
    tool_spec = McpToolSpec(client, allowed_tools=["echo"])

    tools = await tool_spec.to_tool_list_async()
    assert len(tools) == 1
    assert tools[0].metadata.name == "echo"


def test_get_zero_tools(client: BasicMCPClient):
    tool_spec = McpToolSpec(client, allowed_tools=[])
    tools = tool_spec.to_tool_list()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_get_zero_tools_async(client: BasicMCPClient):
    tool_spec = McpToolSpec(client, allowed_tools=[])
    tools = await tool_spec.to_tool_list_async()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_pydantic_models_tool(client: BasicMCPClient):
    """Test the test_pydantic tool with Pydantic models that use custom type aliases and Literal types."""
    tool_spec = McpToolSpec(client, allowed_tools=["test_pydantic"])
    tools = await tool_spec.to_tool_list_async()

    tool = tools[0]
    assert tool.metadata.name == "test_pydantic"

    result = await tool.async_fn(
        name={"name": "John Doe"},
        method={"method": "POST"},
        lst={"lst": [1, 2, 3, 4, 5]},
    )

    assert (
        "Name: John Doe, Method: POST, List: [1, 2, 3, 4, 5]" in result.content[0].text
    )
