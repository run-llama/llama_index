# CHANGELOG

## [0.5.0] - 2026-08-02

### Added
- Exposed `JsonSchemaToPydantic` as a standalone public class (importable from `llama_index.tools.mcp`). Converts an MCP tool's JSON Schema into a Pydantic model without requiring an MCP `ClientSession`.

### Changed
- `McpToolSpec` now delegates schema conversion to an internal `JsonSchemaToPydantic` instance. Public API (`create_model_from_json_schema`, `remove_model_fields`, `properties_cache`, `to_tool_list_async`) is unchanged and produces identical models.

## [0.4.0] - 2025-08-06

- Reworked function `create_model_from_json_schema` in `llama_index.tools.mcp.base`.
- Added handler for pydantic models.
- Added handler for some custom types (List and Union).
- Added handler for enumerations (like Literal).

## [0.1.0] - 2025-02-11

- Add a basic MCP ToolSpec to connect to MCP Server and get tools.
