"""Hugging Face MCP Server tool spec for LlamaIndex.

This module provides a specialized wrapper around the generic MCP tool spec
that is pre-configured for the Hugging Face MCP server. It enables LlamaIndex
agents to search models, datasets, Spaces, papers, and documentation on the
Hugging Face Hub.
"""

import asyncio
from typing import Any, Callable, List, Optional

from llama_index.core.tools.function_tool import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Default HuggingFace MCP server endpoints
HF_MCP_URL = "https://huggingface.co/mcp"
HF_MCP_URL_AUTH = "https://huggingface.co/mcp?login"

# Known built-in tools provided by the HF MCP server
HF_BUILT_IN_TOOLS = [
    "hf_space_search",
    "hf_paper_search",
    "hf_model_search",
    "hf_dataset_search",
    "hf_doc_search",
    "hf_doc_fetch",
    "hf_repo_details",
    "hf_jobs",
]


class HfMcpToolSpec(McpToolSpec):
    """Tool spec for the Hugging Face MCP server.

    Connects to the Hugging Face MCP server and exposes its tools
    (model search, dataset search, Spaces search, paper search,
    documentation search, etc.) as LlamaIndex FunctionTool objects.

    This is a convenience wrapper around McpToolSpec that is pre-configured
    for the Hugging Face MCP server URL. It also supports filtering to
    specific tool categories.

    Args:
        client: A BasicMCPClient instance. If not provided, one will be
            created automatically pointing to the HF MCP server.
        allowed_tools: If set, only expose tools with these names.
            See HF_BUILT_IN_TOOLS for the list of known tool names.
        include_resources: Whether to include MCP resources as tools.
        hf_token: Optional Hugging Face API token for authenticated access.
            When provided, enables access to private resources and
            higher rate limits.
        use_auth_url: Whether to use the authenticated MCP endpoint
            (https://huggingface.co/mcp?login). Defaults to True.
        timeout: Connection timeout in seconds. Defaults to 60.

    Example:
        >>> from llama_index.tools.mcp_hf_server import HfMcpToolSpec
        >>> tool_spec = HfMcpToolSpec()
        >>> tools = await tool_spec.to_tool_list_async()
        >>> # Use with an agent
        >>> from llama_index.core.agent.workflow import FunctionAgent
        >>> agent = FunctionAgent(tools=tools, llm=llm)

    """

    def __init__(
        self,
        client: Optional[BasicMCPClient] = None,
        allowed_tools: Optional[List[str]] = None,
        include_resources: bool = False,
        hf_token: Optional[str] = None,
        use_auth_url: bool = True,
        timeout: int = 60,
    ) -> None:
        if client is None:
            url = HF_MCP_URL_AUTH if use_auth_url else HF_MCP_URL
            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            client = BasicMCPClient(
                command_or_url=url,
                timeout=timeout,
                headers=headers if headers else None,
            )

        super().__init__(
            client=client,
            allowed_tools=allowed_tools,
            include_resources=include_resources,
        )


def get_hf_tools(
    allowed_tools: Optional[List[str]] = None,
    hf_token: Optional[str] = None,
    use_auth_url: bool = True,
    timeout: int = 60,
) -> List[FunctionTool]:
    """Get LlamaIndex tools from the Hugging Face MCP server (synchronous).

    Convenience function that creates an HfMcpToolSpec and returns
    the tool list in a single call.

    Args:
        allowed_tools: If set, only return tools with these names.
        hf_token: Optional HF API token for authenticated access.
        use_auth_url: Whether to use the authenticated endpoint.
        timeout: Connection timeout in seconds.

    Returns:
        A list of FunctionTool objects.

    Example:
        >>> from llama_index.tools.mcp_hf_server import get_hf_tools
        >>> tools = get_hf_tools(allowed_tools=["hf_model_search", "hf_dataset_search"])

    """
    tool_spec = HfMcpToolSpec(
        allowed_tools=allowed_tools,
        hf_token=hf_token,
        use_auth_url=use_auth_url,
        timeout=timeout,
    )
    return tool_spec.to_tool_list()


async def aget_hf_tools(
    allowed_tools: Optional[List[str]] = None,
    hf_token: Optional[str] = None,
    use_auth_url: bool = True,
    timeout: int = 60,
) -> List[FunctionTool]:
    """Get LlamaIndex tools from the Hugging Face MCP server (asynchronous).

    Async convenience function that creates an HfMcpToolSpec and returns
    the tool list in a single call.

    Args:
        allowed_tools: If set, only return tools with these names.
        hf_token: Optional HF API token for authenticated access.
        use_auth_url: Whether to use the authenticated endpoint.
        timeout: Connection timeout in seconds.

    Returns:
        A list of FunctionTool objects.

    Example:
        >>> from llama_index.tools.mcp_hf_server import aget_hf_tools
        >>> tools = await aget_hf_tools(
        ...     allowed_tools=["hf_model_search", "hf_paper_search"]
        ... )

    """
    tool_spec = HfMcpToolSpec(
        allowed_tools=allowed_tools,
        hf_token=hf_token,
        use_auth_url=use_auth_url,
        timeout=timeout,
    )
    return await tool_spec.to_tool_list_async()
