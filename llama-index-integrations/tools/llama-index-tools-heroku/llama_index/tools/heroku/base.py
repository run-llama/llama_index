"""Heroku Agents API tools for LlamaIndex."""

from typing import Any, List

import httpx
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class HerokuToolSpec(BaseToolSpec):
    """Heroku Agents API tool specification for LlamaIndex.

    This class provides tools that integrate with the Heroku Agents API,
    enabling LlamaIndex agents to execute SQL queries, run code, and
    interact with Heroku apps.

    Args:
        api_key: Your Heroku Inference API key.
        app_name: The Heroku app name to interact with.
        base_url: The Heroku Inference API base URL.
        timeout: Request timeout in seconds.

    Example:
        >>> from llama_index.tools.heroku import HerokuToolSpec
        >>> tool_spec = HerokuToolSpec(api_key="your-key", app_name="my-app")
        >>> tools = tool_spec.to_tool_list()
    """

    spec_functions = ["run_sql", "run_python", "run_javascript", "get_app_info"]

    def __init__(
        self,
        api_key: str,
        app_name: str,
        base_url: str = "https://us.inference.heroku.com",
        timeout: float = 120.0,
    ) -> None:
        """Initialize the Heroku tool specification."""
        super().__init__()
        self.api_key = api_key
        self.app_name = app_name
        self.base_url = base_url
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_agent_tool(self, tool_name: str, tool_input: dict) -> str:
        """Call a Heroku Agents API tool.

        Args:
            tool_name: The name of the tool to call.
            tool_input: The input parameters for the tool.

        Returns:
            The tool execution result as a string.
        """
        response = self._client.post(
            f"{self.base_url}/v1/agents/heroku",
            headers=self._get_headers(),
            json={
                "app_name": self.app_name,
                "tool_choice": {
                    "type": "tool",
                    "name": tool_name,
                },
                "tool_input": tool_input,
            },
        )
        response.raise_for_status()

        result = response.json()
        if "error" in result:
            return f"Error: {result['error']}"
        return result.get("output", str(result))

    async def _acall_agent_tool(self, tool_name: str, tool_input: dict) -> str:
        """Async call a Heroku Agents API tool.

        Args:
            tool_name: The name of the tool to call.
            tool_input: The input parameters for the tool.

        Returns:
            The tool execution result as a string.
        """
        response = await self._async_client.post(
            f"{self.base_url}/v1/agents/heroku",
            headers=self._get_headers(),
            json={
                "app_name": self.app_name,
                "tool_choice": {
                    "type": "tool",
                    "name": tool_name,
                },
                "tool_input": tool_input,
            },
        )
        response.raise_for_status()

        result = response.json()
        if "error" in result:
            return f"Error: {result['error']}"
        return result.get("output", str(result))

    def run_sql(self, query: str) -> str:
        """Execute a SQL query on the Heroku Postgres database.

        This tool executes SQL queries against the Postgres database
        attached to your Heroku app. Use this for data retrieval,
        analysis, and database operations.

        Args:
            query: The SQL query to execute.

        Returns:
            The query results as a formatted string.

        Example:
            >>> result = tool_spec.run_sql("SELECT COUNT(*) FROM users")
            >>> print(result)
        """
        return self._call_agent_tool("pg_psql", {"query": query})

    def run_python(self, code: str) -> str:
        """Execute Python code in a sandboxed environment.

        This tool executes Python code on Heroku's secure code execution
        environment. Use this for data processing, calculations, and
        other Python operations.

        Args:
            code: The Python code to execute.

        Returns:
            The execution output as a string.

        Example:
            >>> result = tool_spec.run_python("print(sum(range(100)))")
            >>> print(result)
        """
        return self._call_agent_tool("code_exec_python", {"code": code})

    def run_javascript(self, code: str) -> str:
        """Execute JavaScript code in a sandboxed environment.

        This tool executes JavaScript/Node.js code on Heroku's secure
        code execution environment. Use this for JavaScript operations.

        Args:
            code: The JavaScript code to execute.

        Returns:
            The execution output as a string.

        Example:
            >>> result = tool_spec.run_javascript("console.log([1,2,3].reduce((a,b) => a+b))")
            >>> print(result)
        """
        return self._call_agent_tool("code_exec_javascript", {"code": code})

    def get_app_info(self) -> str:
        """Get information about the Heroku app.

        This tool retrieves metadata and configuration information
        about the specified Heroku app.

        Returns:
            App information as a formatted string.

        Example:
            >>> info = tool_spec.get_app_info()
            >>> print(info)
        """
        return self._call_agent_tool("app_info", {})


def create_heroku_tools(
    api_key: str,
    app_name: str,
    base_url: str = "https://us.inference.heroku.com",
    timeout: float = 120.0,
) -> List:
    """Create a list of Heroku tools for use with LlamaIndex agents.

    This is a convenience function that creates a HerokuToolSpec and
    returns its tools as a list.

    Args:
        api_key: Your Heroku Inference API key.
        app_name: The Heroku app name to interact with.
        base_url: The Heroku Inference API base URL.
        timeout: Request timeout in seconds.

    Returns:
        A list of FunctionTool objects.

    Example:
        >>> tools = create_heroku_tools(api_key="key", app_name="my-app")
        >>> agent = ReActAgent.from_tools(tools, llm=llm)
    """
    tool_spec = HerokuToolSpec(
        api_key=api_key,
        app_name=app_name,
        base_url=base_url,
        timeout=timeout,
    )
    return tool_spec.to_tool_list()
