from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.mcp import McpToolSpec

from llama_index.tools.openregistry import (
    OPENREGISTRY_MCP_URL,
    OpenRegistryToolSpec,
)


def test_default_url() -> None:
    assert OPENREGISTRY_MCP_URL == "https://openregistry.sophymarine.com/mcp"


def test_subclass() -> None:
    """OpenRegistryToolSpec must be drop-in compatible with McpToolSpec."""
    assert issubclass(OpenRegistryToolSpec, McpToolSpec)
    assert issubclass(OpenRegistryToolSpec, BaseToolSpec)


def test_anonymous_init_sets_no_auth_header() -> None:
    spec = OpenRegistryToolSpec()
    assert spec.client.command_or_url == OPENREGISTRY_MCP_URL
    assert spec.client.headers is None
    assert spec.allowed_tools is None
    assert spec.include_resources is False


def test_token_init_sets_bearer_header() -> None:
    spec = OpenRegistryToolSpec(oauth_token="test-token-123")
    assert spec.client.headers == {"Authorization": "Bearer test-token-123"}


def test_url_override() -> None:
    spec = OpenRegistryToolSpec(url="https://staging.openregistry.sophymarine.com/mcp")
    assert spec.client.command_or_url == "https://staging.openregistry.sophymarine.com/mcp"


def test_allowed_tools_passthrough() -> None:
    allow = ["search_companies", "get_company_profile"]
    spec = OpenRegistryToolSpec(allowed_tools=allow)
    assert spec.allowed_tools == allow
