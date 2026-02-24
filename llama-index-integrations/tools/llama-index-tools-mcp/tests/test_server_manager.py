import os
import pytest

from llama_index.tools.mcp import BasicMCPClient, McpServerManager


# Path to the test server script
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")


@pytest.mark.asyncio
async def test_server_manager_basic():
    """Test basic server manager operations."""
    manager = McpServerManager()

    client = BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)
    manager.add_server("test", client)

    assert "test" in manager.server_names
    assert manager.get_server("test") is client
    assert manager.get_server("nonexistent") is None


@pytest.mark.asyncio
async def test_server_manager_context_manager():
    """Test McpServerManager as an async context manager."""
    manager = McpServerManager()
    manager.add_server(
        "server1", BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)
    )

    async with manager:
        # All servers should be connected
        server = manager.get_server("server1")
        assert server is not None
        assert server.is_connected

        tools = await manager.get_tools()
        assert len(tools) > 0

    # After exiting context, servers should be disconnected
    assert not server.is_connected


@pytest.mark.asyncio
async def test_server_manager_get_tools():
    """Test getting tools from all servers."""
    manager = McpServerManager()
    manager.add_server(
        "server1", BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)
    )

    async with manager:
        tools = await manager.get_tools()
        tool_names = [t.metadata.name for t in tools]
        assert "echo" in tool_names
        assert "add" in tool_names


@pytest.mark.asyncio
async def test_server_manager_get_tools_with_filter():
    """Test getting filtered tools from servers."""
    manager = McpServerManager()
    manager.add_server(
        "server1", BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)
    )

    async with manager:
        tools = await manager.get_tools(
            allowed_tools={"server1": ["echo"]}
        )
        assert len(tools) == 1
        assert tools[0].metadata.name == "echo"


@pytest.mark.asyncio
async def test_server_manager_get_tools_from_server():
    """Test getting tools from a specific server."""
    manager = McpServerManager()
    manager.add_server(
        "server1", BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)
    )

    async with manager:
        tools = await manager.get_tools_from_server(
            "server1", allowed_tools=["echo", "add"]
        )
        tool_names = [t.metadata.name for t in tools]
        assert "echo" in tool_names
        assert "add" in tool_names


@pytest.mark.asyncio
async def test_server_manager_get_tools_from_nonexistent_server():
    """Test that getting tools from nonexistent server raises ValueError."""
    manager = McpServerManager()

    async with manager:
        with pytest.raises(ValueError, match="not found"):
            await manager.get_tools_from_server("nonexistent")


@pytest.mark.asyncio
async def test_server_manager_remove_server():
    """Test removing a server from the manager."""
    manager = McpServerManager()
    manager.add_server(
        "test", BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)
    )

    assert "test" in manager.server_names
    manager.remove_server("test")
    assert "test" not in manager.server_names

    # Removing nonexistent server should not raise
    manager.remove_server("nonexistent")


@pytest.mark.asyncio
async def test_server_manager_init_with_servers():
    """Test initializing McpServerManager with a dict of servers."""
    servers = {
        "server1": BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5),
    }
    manager = McpServerManager(servers=servers)

    assert "server1" in manager.server_names

    async with manager:
        tools = await manager.get_tools()
        assert len(tools) > 0


@pytest.mark.asyncio
async def test_server_manager_multiple_servers():
    """Test manager with multiple servers pointing to the same test server."""
    manager = McpServerManager()
    manager.add_server(
        "server_a", BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)
    )
    manager.add_server(
        "server_b", BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)
    )

    async with manager:
        tools = await manager.get_tools()
        # Should have tools from both servers (duplicated)
        tool_names = [t.metadata.name for t in tools]
        assert tool_names.count("echo") == 2
        assert tool_names.count("add") == 2
