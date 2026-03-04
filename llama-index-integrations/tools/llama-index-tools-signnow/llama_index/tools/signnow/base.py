"""SignNow MCP tool spec scaffold."""

import os
import shutil
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union, cast

from mcp.client.session import ClientSession

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.tools.mcp.base import McpToolSpec
from llama_index.tools.mcp.client import BasicMCPClient


def _merge_env(overrides: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """
    Build environment for spawning the MCP server:
    - Start from current process environment
    - Overlay provided overrides (takes precedence)
    """
    env = dict(os.environ)
    if overrides:
        env.update({k: v for k, v in overrides.items() if v is not None})
    return env


EXPECTED_SIGNNOW_KEYS = {
    # Auth
    "SIGNNOW_TOKEN",
    "SIGNNOW_USER_EMAIL",
    "SIGNNOW_PASSWORD",
    "SIGNNOW_API_BASIC_TOKEN",
    # API endpoints (optional; defaults may be used)
    "SIGNNOW_APP_BASE",
    "SIGNNOW_API_BASE",
}


def _validate_auth(env: Mapping[str, str]) -> None:
    """Require either SIGNNOW_TOKEN or (SIGNNOW_USER_EMAIL + SIGNNOW_PASSWORD + SIGNNOW_API_BASIC_TOKEN)."""
    have_token = bool(env.get("SIGNNOW_TOKEN"))
    have_basic = all(
        env.get(k)
        for k in ("SIGNNOW_USER_EMAIL", "SIGNNOW_PASSWORD", "SIGNNOW_API_BASIC_TOKEN")
    )
    if not (have_token or have_basic):
        raise ValueError(
            "Provide SIGNNOW_TOKEN OR SIGNNOW_USER_EMAIL + SIGNNOW_PASSWORD + SIGNNOW_API_BASIC_TOKEN."
        )


def _resolve_sn_mcp_bin(explicit: Optional[str], require_in_path: bool) -> str:
    """Resolve path to sn-mcp binary from explicit arg, SIGNNOW_MCP_BIN, or PATH."""
    candidate = explicit or os.environ.get("SIGNNOW_MCP_BIN") or "sn-mcp"
    path = shutil.which(candidate)
    if path:
        return path
    if require_in_path:
        raise FileNotFoundError(
            "Cannot find 'sn-mcp' in PATH. Set SIGNNOW_MCP_BIN or install SignNow MCP server."
        )
    return candidate


class SignNowMCPToolSpec(BaseToolSpec):
    """
    Thin wrapper over McpToolSpec:
    - creates BasicMCPClient for STDIO spawn,
    - dynamically pulls tools from SignNow MCP server,
    - sugar factories: from_env.

    See McpToolSpec.to_tool_list() / .to_tool_list_async() for getting FunctionTool.
    """

    # Follow BaseToolSpec typing contract
    spec_functions: List[Union[str, Tuple[str, str]]] = []

    def __init__(
        self,
        client: ClientSession,
        allowed_tools: Optional[List[str]] = None,
        include_resources: bool = False,
    ) -> None:
        self._mcp_spec = McpToolSpec(
            client=client,
            allowed_tools=allowed_tools,
            include_resources=include_resources,
        )

    @classmethod
    def from_env(
        cls,
        *,
        allowed_tools: Optional[Iterable[str]] = None,
        include_resources: bool = False,
        env_overrides: Optional[Mapping[str, str]] = None,
        bin: Optional[str] = None,
        cmd: str = "serve",
        args: Optional[Sequence[str]] = None,
        require_in_path: bool = True,
    ) -> "SignNowMCPToolSpec":
        """
        Spawn STDIO: 'sn-mcp serve' with provided environment overrides merged
        on top of the current process environment.

        Supported variables (see server README):
          SIGNNOW_TOKEN (token-based auth)
          OR
          SIGNNOW_USER_EMAIL, SIGNNOW_PASSWORD, SIGNNOW_API_BASIC_TOKEN (credential-based auth)
          SIGNNOW_APP_BASE, SIGNNOW_API_BASE (optional, defaults can be used)

        Parameters
        ----------
          - bin: binary/command to spawn (default None → uses SIGNNOW_MCP_BIN or 'sn-mcp')
          - cmd: subcommand (default 'serve')
          - args: additional arguments for the server
          - require_in_path: validate presence of binary in PATH if not absolute

        """
        # Build env and filter to expected keys
        env_all = _merge_env(env_overrides)
        filtered = {k: v for k, v in env_all.items() if k in EXPECTED_SIGNNOW_KEYS}

        _validate_auth(filtered)

        # Resolve binary to absolute if possible
        resolved_bin = _resolve_sn_mcp_bin(bin, require_in_path=require_in_path)

        cmd_args: List[str] = [cmd]
        if args:
            cmd_args.extend(args)

        client = BasicMCPClient(resolved_bin, args=cmd_args, env=filtered)
        return cls(
            client=client,
            allowed_tools=list(allowed_tools) if allowed_tools else None,
            include_resources=include_resources,
        )

    async def to_tool_list_async(self) -> List[FunctionTool]:
        """Delegate to underlying `McpToolSpec` with error handling."""
        result = await self._mcp_spec.to_tool_list_async()
        return cast(List[FunctionTool], result)

    def to_tool_list(
        self,
        spec_functions: Optional[List[Union[str, Tuple[str, str]]]] = None,
        func_to_metadata_mapping: Optional[Dict[str, ToolMetadata]] = None,
    ) -> List[FunctionTool]:
        """Delegate to underlying `McpToolSpec` (sync) with error handling."""
        # We discover tools dynamically via MCP; provided parameters are ignored.
        result = self._mcp_spec.to_tool_list()
        return cast(List[FunctionTool], result)
