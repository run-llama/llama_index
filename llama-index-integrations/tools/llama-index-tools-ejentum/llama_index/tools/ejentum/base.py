"""Ejentum Reasoning Harness tool spec."""

import os
from typing import List, Optional

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec


DEFAULT_API_URL = "https://api.ejentum.com/mcp"
SUPPORTED_MODES = ("reasoning", "code", "anti_deception", "memory")


class EjentumToolSpec(McpToolSpec):
    """
    Ejentum Reasoning Harness as a LlamaIndex tool spec.

    Wraps the hosted Ejentum MCP server (``https://api.ejentum.com/mcp``) and
    exposes its four cognitive-harness tools as LlamaIndex ``FunctionTool``
    objects an agent can route to:

    - ``harness_reasoning`` for analytical, diagnostic, planning, and
      multi-step decision tasks.
    - ``harness_code`` for code generation, refactoring, and architecture
      decisions.
    - ``harness_anti_deception`` for validation requests, ethical reasoning,
      and adversarial framings.
    - ``harness_memory`` for perception sharpening across multi-turn context.

    Each call returns a structured scaffold (named failure pattern, executable
    procedure, suppression vectors, and falsification test) that the calling
    agent absorbs before generating its next response.

    This class is a thin subclass of
    :class:`llama_index.tools.mcp.McpToolSpec` that pre-configures the hosted
    endpoint with Bearer authentication and exposes a mode-subset shorthand.
    For raw MCP usage against arbitrary servers, use ``McpToolSpec`` directly.

    Args:
        api_key: Ejentum API key. Falls back to the ``EJENTUM_API_KEY``
            environment variable. Free and paid tiers at
            https://ejentum.com/auth/register.
        modes: Optional subset of harness modes to expose. Values are short
            names without the ``harness_`` prefix, e.g.
            ``["reasoning", "code"]``. Defaults to all four modes.
        api_url: Override the hosted MCP endpoint. Defaults to
            ``https://api.ejentum.com/mcp``.
        timeout: HTTP timeout in seconds. Defaults to 30.

    Example:
        >>> from llama_index.tools.ejentum import EjentumToolSpec
        >>> spec = EjentumToolSpec()  # reads EJENTUM_API_KEY from env
        >>> tools = spec.to_tool_list()
        >>> # ...pass ``tools`` to your LlamaIndex agent

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        modes: Optional[List[str]] = None,
        api_url: str = DEFAULT_API_URL,
        timeout: int = 30,
    ) -> None:
        resolved_key = api_key or os.getenv("EJENTUM_API_KEY")
        if not resolved_key:
            raise ValueError(
                "EJENTUM_API_KEY is not set. Pass api_key= explicitly or set "
                "the environment variable. Sign up at "
                "https://ejentum.com/auth/register."
            )

        allowed_tools: Optional[List[str]] = None
        if modes is not None:
            unknown = [m for m in modes if m not in SUPPORTED_MODES]
            if unknown:
                raise ValueError(
                    f"Unknown mode(s): {unknown}. "
                    f"Supported modes: {list(SUPPORTED_MODES)}."
                )
            allowed_tools = [f"harness_{m}" for m in modes]

        client = BasicMCPClient(
            api_url,
            headers={"Authorization": f"Bearer {resolved_key}"},
            timeout=timeout,
        )
        super().__init__(client=client, allowed_tools=allowed_tools)
