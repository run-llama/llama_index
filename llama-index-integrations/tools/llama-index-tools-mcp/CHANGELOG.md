# CHANGELOG

## [0.5.0] - 2026-08-01

- Support the MCP Python SDK 2.x line alongside 1.x (widened the pin to `mcp>=1.24.0,<3`).
- Added an internal `_compat` layer isolating the 1.x/2.x differences: relocated
  `ProgressFnT`, the `httpx` -> `httpx2` HTTP stack, the `FastMCP` -> `MCPServer`
  high-level server, `read_timeout_seconds` (timedelta vs float), the
  `streamable_http_client` 3- vs 2-tuple yield, `Context.log`'s renamed argument,
  and the camelCase -> snake_case protocol-model field renames.
- Fixed a silent regression where resource templates were dropped under mcp 2.x
  (`resourceTemplates` -> `resource_templates`).

## [0.4.0] - 2025-08-06

- Reworked function `create_model_from_json_schema` in `llama_index.tools.mcp.base`.
- Added handler for pydantic models.
- Added handler for some custom types (List and Union).
- Added handler for enumerations (like Literal).

## [0.1.0] - 2025-02-11

- Add a basic MCP ToolSpec to connect to MCP Server and get tools.
