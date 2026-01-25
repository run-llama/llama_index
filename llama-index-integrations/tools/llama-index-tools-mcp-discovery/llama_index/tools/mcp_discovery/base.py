"""MCP Discovery tool spec."""

import aiohttp
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class MCPDiscoveryTool(BaseToolSpec):
    """
    MCP Discovery Tool.

    This tool queries the MCP Discovery API for autonomous tool recommendations.
    It accepts a natural language description of the need and returns a
    human-readable list of recommended MCP servers with name, category, and description.

    Attributes:
        api_url: The URL of the MCP discovery API endpoint.

    """

    spec_functions = ["discover_tools"]

    def __init__(self, api_url: str) -> None:
        """
        Initialize the MCP Discovery Tool.

        Args:
            api_url: The URL of the MCP discovery API endpoint.

        """
        self.api_url = api_url

    async def discover_tools(self, user_request: str, limit: int = 5) -> str:
        """
        Discover tools based on a natural language request.

        This method allows an agent to discover needed tools without human intervention.
        It queries the MCP discovery API with the user's request and returns formatted
        tool recommendations.

        Args:
            user_request: Natural language description of the tool needed.
            limit: Maximum number of tool recommendations to return. Defaults to 5.

        Returns:
            A formatted string containing the discovered tools with their names,
            descriptions, and categories. Returns an error message if the request fails.

        Example:
            >>> tool = MCPDiscoveryTool(api_url="http://localhost:8000/api")
            >>> result = await tool.discover_tools("I need a math calculator", limit=3)
            >>> print(result)
            Found 2 tools:
            1. Name: math-calculator,
               Description: A tool for calculations,
               Category: math

        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, json={"need": user_request, "limit": limit}
                ) as response:
                    data = await response.json()

            tools_json = data.get("recommendations", [])
            num = data.get("total_found", -1)

            if num == -1:
                tools = "Following tools are found:\n"
            else:
                tools = f"Found {num} tools:\n"

            if tools_json:
                for ind, i in enumerate(tools_json, start=1):
                    tools += f"{ind}. Name: {i.get('name')},\n"
                    tools += f"   Description: {i.get('description')},\n"
                    tools += f"   Category: {i.get('category')}\n\n"
                return tools.strip()

            return tools

        except Exception as e:
            return f"Error discovering tools: {e}"
