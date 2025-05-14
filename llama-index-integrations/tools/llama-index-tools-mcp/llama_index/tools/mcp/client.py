from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional, List, Dict
from urllib.parse import urlparse

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters


class BasicMCPClient(ClientSession):
    """
    Basic MCP client that can be used to connect to an MCP server.

    This is useful for verifying that the MCP server which implements `FastMCP` is working.

    Args:
        command_or_url: The command to run or the URL to connect to.
        args: The arguments to pass to StdioServerParameters.
        env: The environment variables to set for StdioServerParameters.
        timeout: The timeout for the command in seconds.

    """

    def __init__(
        self,
        command_or_url: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ):
        self.command_or_url = command_or_url
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout

    @asynccontextmanager
    async def _run_session(self):
        if urlparse(self.command_or_url).scheme in ("http", "https"):
            async with sse_client(self.command_or_url) as streams:
                async with ClientSession(
                    *streams, read_timeout_seconds=timedelta(seconds=self.timeout)
                ) as session:
                    await session.initialize()
                    yield session
        else:
            server_parameters = StdioServerParameters(
                command=self.command_or_url, args=self.args, env=self.env
            )
            async with stdio_client(server_parameters) as streams:
                async with ClientSession(
                    *streams, read_timeout_seconds=timedelta(seconds=self.timeout)
                ) as session:
                    await session.initialize()
                    yield session

    async def call_tool(self, tool_name: str, arguments: dict):
        async with self._run_session() as session:
            return await session.call_tool(tool_name, arguments)

    async def list_tools(self):
        async with self._run_session() as session:
            return await session.list_tools()
