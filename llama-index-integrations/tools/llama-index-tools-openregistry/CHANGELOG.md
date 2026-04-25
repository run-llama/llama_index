# CHANGELOG

## [0.1.0]

- Initial release: `OpenRegistryToolSpec`, a thin convenience wrapper around `McpToolSpec` from `llama-index-tools-mcp` preconfigured for the hosted [OpenRegistry](https://openregistry.sophymarine.com) MCP server.
- Anonymous tier supported with no API key. Optional OAuth bearer token for authenticated higher-rate tiers.
- Optional `allowed_tools` allowlist and `url` override for staging / self-hosted endpoints.
