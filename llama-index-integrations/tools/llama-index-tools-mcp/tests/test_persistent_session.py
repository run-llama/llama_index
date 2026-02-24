import os
import pytest

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from mcp import types


# Path to the test server script
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")


@pytest.mark.asyncio
async def test_persistent_session_connect_disconnect():
    """Test that connect/disconnect lifecycle works."""
    client = BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)

    assert not client.is_connected

    await client.connect()
    assert client.is_connected

    # Should be able to list tools with persistent session
    tools = await client.list_tools()
    assert len(tools.tools) > 0

    await client.disconnect()
    assert not client.is_connected


@pytest.mark.asyncio
async def test_persistent_session_context_manager():
    """Test using BasicMCPClient as an async context manager."""
    async with BasicMCPClient(
        "python", args=[SERVER_SCRIPT], timeout=5
    ) as client:
        assert client.is_connected

        # Multiple operations on the same session
        tools = await client.list_tools()
        assert len(tools.tools) > 0

        result = await client.call_tool("echo", {"message": "hello"})
        assert result.content[0].text == "Echo: hello"

        resources = await client.list_resources()
        assert len(resources.resources) > 0

    # After exiting context, should be disconnected
    assert not client.is_connected


@pytest.mark.asyncio
async def test_persistent_session_multiple_tool_calls():
    """Test multiple tool calls on a persistent session."""
    async with BasicMCPClient(
        "python", args=[SERVER_SCRIPT], timeout=5
    ) as client:
        # Call multiple tools in sequence on the same connection
        result1 = await client.call_tool("echo", {"message": "first"})
        assert result1.content[0].text == "Echo: first"

        result2 = await client.call_tool("add", {"a": 3, "b": 4})
        assert result2.content[0].text == "7.0"

        result3 = await client.call_tool("echo", {"message": "third"})
        assert result3.content[0].text == "Echo: third"


@pytest.mark.asyncio
async def test_persistent_session_with_tool_spec():
    """Test McpToolSpec with a persistent session client."""
    async with BasicMCPClient(
        "python", args=[SERVER_SCRIPT], timeout=5
    ) as client:
        tool_spec = McpToolSpec(client, allowed_tools=["echo"])
        tools = await tool_spec.to_tool_list_async()

        assert len(tools) == 1
        assert tools[0].metadata.name == "echo"

        result = await tools[0].acall(message="persistent test")
        assert "Echo: persistent test" in result.content


@pytest.mark.asyncio
async def test_connect_idempotent():
    """Test that calling connect multiple times is safe."""
    client = BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)

    await client.connect()
    assert client.is_connected

    # Second connect should be a no-op
    await client.connect()
    assert client.is_connected

    await client.disconnect()
    assert not client.is_connected


@pytest.mark.asyncio
async def test_disconnect_idempotent():
    """Test that calling disconnect when not connected is safe."""
    client = BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)

    # Disconnecting when not connected should not raise
    await client.disconnect()
    assert not client.is_connected
