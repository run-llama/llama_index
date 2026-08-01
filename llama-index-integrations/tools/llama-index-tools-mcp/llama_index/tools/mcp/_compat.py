"""
Compatibility shims for MCP Python SDK 1.x and 2.x.

The MCP Python SDK 2.0 release (2026-07-28) made several breaking changes that
touch this integration. Everything version-specific is isolated here so the
rest of the package can be written once and run under either SDK major:

* ``ProgressFnT`` moved from ``mcp.shared.session`` to ``mcp.client.session``.
* the HTTP stack moved from ``httpx`` to ``httpx2``.
* the high-level server class moved from ``mcp.server.fastmcp.FastMCP`` to
  ``mcp.server.mcpserver.MCPServer`` (the decorator API is otherwise the same).
* ``ClientSession(read_timeout_seconds=...)`` takes a ``float`` of seconds
  instead of a ``datetime.timedelta``.
* ``streamable_http_client`` yields a 2-tuple instead of a 3-tuple.
* ``Context.log(...)`` renamed its second parameter ``message`` -> ``data``
  (call it positionally to stay compatible).
* several protocol-model fields were renamed camelCase -> snake_case
  (``Tool.inputSchema`` -> ``input_schema``, ``*.mimeType`` -> ``mime_type``,
  ``ListResourceTemplatesResult.resourceTemplates`` -> ``resource_templates``).
"""

from datetime import timedelta
from importlib.metadata import version as _dist_version
from typing import Any, Union

# ``mcp`` is always installed; the major version drives every branch below.
MCP_MAJOR = int(_dist_version("mcp").split(".", 1)[0])
IS_MCP_V2 = MCP_MAJOR >= 2

# --- ProgressFnT relocation ------------------------------------------------
try:  # mcp 1.x
    from mcp.shared.session import ProgressFnT
except ModuleNotFoundError:  # mcp >= 2.0
    from mcp.client.session import ProgressFnT

# --- httpx (1.x) vs httpx2 (2.x) -------------------------------------------
if IS_MCP_V2:
    from httpx2 import AsyncClient, Timeout
else:
    from httpx import AsyncClient, Timeout

# --- high-level server class + prompt helpers ------------------------------
if IS_MCP_V2:
    from mcp.server.mcpserver import MCPServer as MCPServerApp, Context, Image
    from mcp.server.mcpserver import prompts as _server_prompts
else:
    from mcp.server.fastmcp import FastMCP as MCPServerApp, Context, Image
    from mcp.server.fastmcp import prompts as _server_prompts

prompt_base = _server_prompts.base

__all__ = [
    "IS_MCP_V2",
    "MCP_MAJOR",
    "ProgressFnT",
    "AsyncClient",
    "Timeout",
    "MCPServerApp",
    "Context",
    "Image",
    "prompt_base",
    "read_timeout_seconds",
    "compat_getattr",
]

_MISSING = object()


def read_timeout_seconds(seconds: float) -> Union[float, timedelta]:
    """
    Build the value ``ClientSession(read_timeout_seconds=...)`` expects.

    mcp 2.0 takes a plain float of seconds; mcp 1.x wants a ``timedelta``.
    """
    return seconds if IS_MCP_V2 else timedelta(seconds=seconds)


def compat_getattr(obj: Any, snake: str, camel: str, default: Any = _MISSING) -> Any:
    """
    Read a protocol-model field that was renamed camelCase -> snake_case in
    mcp 2.0, trying the 2.x name first and falling back to the 1.x name.

    If neither attribute exists and a ``default`` is supplied it is returned,
    otherwise ``AttributeError`` is raised.
    """
    value = getattr(obj, snake, _MISSING)
    if value is _MISSING:
        value = getattr(obj, camel, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise AttributeError(
                f"{type(obj).__name__} has neither {snake!r} nor {camel!r}"
            )
        return default
    return value
