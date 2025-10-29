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
async def test_resource_tool_uses_uri_not_name(client: BasicMCPClient):
    """
    Tests that a tool from a static resource is executable.
    """
    tool_spec = McpToolSpec(
        client, allowed_tools=["get_app_config"], include_resources=True
    )
    tools = await tool_spec.to_tool_list_async()

    assert len(tools) == 1
    tool = tools[0]
    assert tool.metadata.name == "get_app_config"

    # This call should succeed now that the bug is fixed.
    result = await tool.acall()
    assert "MCP Test Server" in result.raw_output.contents[0].text


@pytest.mark.asyncio
async def test_dynamic_resource_template_tool_is_created(client: BasicMCPClient):
    """
    Tests that a tool is created for a dynamic resource template.
    """
    tool_spec = McpToolSpec(client, include_resources=True)
    tools = await tool_spec.to_tool_list_async()

    # The server.py defines a dynamic resource template named 'get_user_profile'.
    # This should now be found.
    tool_names = {t.metadata.name for t in tools}
    assert "get_user_profile" in tool_names
