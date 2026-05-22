# CHANGELOG

## [0.1.0]

- Initial release.
- `EjentumToolSpec` subclasses `McpToolSpec` and pre-configures the hosted Ejentum MCP server at `https://api.ejentum.com/mcp` with Bearer authentication.
- Four harness tools exposed: `harness_reasoning`, `harness_code`, `harness_anti_deception`, `harness_memory`.
- Mode-subset shorthand via the `modes=` constructor argument.
