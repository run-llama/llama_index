"""OpenRegistry tool spec for LlamaIndex.

OpenRegistry is a hosted Streamable HTTP MCP server that proxies 27 national
company registries (UK Companies House, France RNE, Germany Handelsregister,
Italy InfoCamere via EU BRIS, Spain BORME, Korea OpenDART, etc.). Every tool
call is a real-time query against the upstream government API; no data is
cached or aggregated.

This package is a thin convenience wrapper around ``McpToolSpec`` from
``llama-index-tools-mcp`` — it preconfigures the LlamaIndex agent with the
hosted endpoint and forwards any optional OAuth bearer token (for higher
rate-limit tiers).
"""

from typing import List, Optional

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

OPENREGISTRY_MCP_URL = "https://openregistry.sophymarine.com/mcp"


class OpenRegistryToolSpec(McpToolSpec):
    """LlamaIndex ToolSpec preconfigured for the OpenRegistry MCP server.

    Use the free anonymous tier with no arguments. For per-user rate limits or
    cross-border fan-out beyond 3 countries / 60s, complete the OAuth 2.1 flow
    against ``openregistry.sophymarine.com`` and pass the resulting bearer
    token to ``oauth_token``.

    Example:
        >>> from llama_index.tools.openregistry import OpenRegistryToolSpec
        >>> from llama_index.core.agent.workflow import FunctionAgent
        >>> from llama_index.llms.openai import OpenAI
        >>>
        >>> tool_spec = OpenRegistryToolSpec()
        >>> agent = FunctionAgent(
        ...     tools=tool_spec.to_tool_list(),
        ...     llm=OpenAI(model="gpt-4.1"),
        ... )
        >>> # await agent.run("Find Tesco PLC on Companies House")

    Args:
        oauth_token: Optional OAuth 2.1 bearer token for authenticated tiers.
            See https://openregistry.sophymarine.com/docs for the OAuth flow.
        url: Override the MCP endpoint. Defaults to the hosted production URL.
            Useful for staging / self-hosted setups.
        allowed_tools: Optional allowlist of tool names. When set, only the
            named tools are exposed to the agent. By default all ~30 tools are
            exposed.
        include_resources: Forwarded to ``McpToolSpec``; enable to surface
            MCP resources alongside tools.

    """

    def __init__(
        self,
        oauth_token: Optional[str] = None,
        url: str = OPENREGISTRY_MCP_URL,
        allowed_tools: Optional[List[str]] = None,
        include_resources: bool = False,
    ) -> None:
        headers = {"Authorization": f"Bearer {oauth_token}"} if oauth_token else None
        client = BasicMCPClient(url, headers=headers)
        super().__init__(
            client=client,
            allowed_tools=allowed_tools,
            include_resources=include_resources,
        )
