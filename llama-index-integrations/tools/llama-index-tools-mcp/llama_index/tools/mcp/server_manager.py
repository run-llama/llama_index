import logging
from typing import Any, Dict, List, Optional

from llama_index.core.tools.function_tool import FunctionTool
from llama_index.tools.mcp.base import McpToolSpec
from llama_index.tools.mcp.client import BasicMCPClient

logger = logging.getLogger(__name__)


class McpServerManager:
    """
    Manages multiple MCP server connections and aggregates their tools.

    This class simplifies working with multiple MCP servers by providing
    a unified interface to connect, disconnect, and retrieve tools from
    all servers at once.

    Args:
        servers: Optional dict mapping server names to BasicMCPClient instances.

    Example:
        .. code-block:: python

            from llama_index.tools.mcp import BasicMCPClient, McpServerManager

            manager = McpServerManager()
            manager.add_server("github", BasicMCPClient("https://api.github.com/mcp"))
            manager.add_server("db", BasicMCPClient("python", args=["db_server.py"]))

            async with manager:
                tools = await manager.get_tools()
                # Use tools with an agent

    """

    def __init__(
        self, servers: Optional[Dict[str, BasicMCPClient]] = None
    ) -> None:
        self._servers: Dict[str, BasicMCPClient] = dict(servers) if servers else {}

    def add_server(self, name: str, client: BasicMCPClient) -> None:
        """
        Add an MCP server to the manager.

        Args:
            name: A unique name for this server connection.
            client: A BasicMCPClient instance configured for the server.

        """
        if name in self._servers:
            logger.warning(
                f"Server '{name}' already exists and will be replaced."
            )
        self._servers[name] = client

    def remove_server(self, name: str) -> None:
        """
        Remove an MCP server from the manager.

        Args:
            name: The name of the server to remove.

        """
        self._servers.pop(name, None)

    @property
    def server_names(self) -> List[str]:
        """Return the names of all registered servers."""
        return list(self._servers.keys())

    def get_server(self, name: str) -> Optional[BasicMCPClient]:
        """
        Get a specific server client by name.

        Args:
            name: The server name.

        Returns:
            The BasicMCPClient if found, None otherwise.

        """
        return self._servers.get(name)

    async def connect_all(self) -> None:
        """
        Establish persistent connections to all registered servers.

        Servers that fail to connect will log a warning but not prevent
        other servers from connecting.
        """
        for name, client in self._servers.items():
            try:
                await client.connect()
            except Exception as e:
                logger.warning(f"Failed to connect to server '{name}': {e}")

    async def disconnect_all(self) -> None:
        """
        Close persistent connections to all registered servers.

        Errors during disconnection are logged but do not raise.
        """
        for name, client in self._servers.items():
            try:
                await client.disconnect()
            except BaseException as e:
                logger.warning(
                    f"Error disconnecting from server '{name}': {e}"
                )

    async def __aenter__(self) -> "McpServerManager":
        """Enter async context manager, connecting all servers."""
        await self.connect_all()
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Exit async context manager, disconnecting all servers."""
        await self.disconnect_all()

    async def get_tools(
        self,
        allowed_tools: Optional[Dict[str, List[str]]] = None,
        include_resources: bool = False,
    ) -> List[FunctionTool]:
        """
        Get tools from all connected servers.

        Args:
            allowed_tools: Optional dict mapping server name to a list of
                allowed tool names for that server. Servers not in the dict
                will return all their tools.
            include_resources: Whether to include MCP resources as tools.

        Returns:
            A list of FunctionTool objects from all servers.

        """
        all_tools: List[FunctionTool] = []
        for name, client in self._servers.items():
            try:
                server_allowed = (
                    allowed_tools.get(name) if allowed_tools else None
                )
                tool_spec = McpToolSpec(
                    client,
                    allowed_tools=server_allowed,
                    include_resources=include_resources,
                )
                tools = await tool_spec.to_tool_list_async()
                all_tools.extend(tools)
            except Exception as e:
                logger.warning(
                    f"Failed to get tools from server '{name}': {e}"
                )

        return all_tools

    async def get_tools_from_server(
        self,
        name: str,
        allowed_tools: Optional[List[str]] = None,
        include_resources: bool = False,
    ) -> List[FunctionTool]:
        """
        Get tools from a specific server.

        Args:
            name: The server name.
            allowed_tools: Optional list of allowed tool names.
            include_resources: Whether to include MCP resources as tools.

        Returns:
            A list of FunctionTool objects from the specified server.

        Raises:
            ValueError: If the server name is not found.

        """
        client = self._servers.get(name)
        if client is None:
            raise ValueError(
                f"Server '{name}' not found. "
                f"Available servers: {self.server_names}"
            )
        tool_spec = McpToolSpec(
            client,
            allowed_tools=allowed_tools,
            include_resources=include_resources,
        )
        return await tool_spec.to_tool_list_async()
