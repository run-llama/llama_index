import sys
import asyncio
from unittest.mock import MagicMock

# --- MOCKING DEPENDENCIES START ---
# We mock external dependencies to ensure tests run even in minimal environments.

# 1. Mock 'aiohttp'
mock_aiohttp = MagicMock()
sys.modules["aiohttp"] = mock_aiohttp

# 2. Mock 'llama_index.core' and 'BaseToolSpec'
# We need strictly what the base.py imports:
# "from llama_index.core.tools.tool_spec.base import BaseToolSpec"
mock_core = MagicMock()
sys.modules["llama_index.core"] = mock_core
sys.modules["llama_index.core.tools"] = mock_core
sys.modules["llama_index.core.tools.tool_spec"] = mock_core
sys.modules["llama_index.core.tools.tool_spec.base"] = mock_core


# Define a real class for BaseToolSpec so inheritance works
class MockBaseToolSpec:
    spec_functions = []


mock_core.BaseToolSpec = MockBaseToolSpec

# --- MOCKING DEPENDENCIES END ---

# Inherit import paths are resolved relative to the package root usually
# Try correct import based on file structure
try:
    from llama_index.tools.mcp_discovery.base import MCPDiscoveryTool
except ImportError:
    # Fallback if running from a different root
    sys.path.append("llama_index/tools/mcp_discovery")
    from base import MCPDiscoveryTool


# Helper for async context managers (async with ...)
class AsyncContextManager:
    def __init__(self, return_value=None, error=None):
        self.return_value = return_value
        self.error = error

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self.return_value

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass


def test_discover_tools_success():
    """Test successful discovery of tools."""
    tool = MCPDiscoveryTool(api_url="http://test-api.com")

    # Mock Data
    mock_data = {
        "recommendations": [
            {"name": "test-tool", "description": "A test tool", "category": "Testing"},
            {"name": "fancy-tool", "description": "A fancy tool", "category": "Fancy"},
        ],
        "total_found": 2,
    }

    # Setup Mocks for aiohttp
    mock_response = MagicMock()

    # Make json() awaitable using an async function
    async def get_json():
        return mock_data

    mock_response.json.side_effect = get_json

    mock_session = MagicMock()
    mock_session.post.return_value = AsyncContextManager(return_value=mock_response)

    sys.modules["aiohttp"].ClientSession.return_value = AsyncContextManager(
        return_value=mock_session
    )

    # Run Code
    result = asyncio.run(tool.discover_tools("help me", limit=2))

    # Assertions
    assert "Found 2 tools" in result
    assert "1. Name: test-tool" in result
    assert "2. Name: fancy-tool" in result
    assert "A test tool" in result


def test_discover_tools_empty():
    """Test behavior when no tools are found."""
    tool = MCPDiscoveryTool(api_url="http://test-api.com")

    mock_data = {"recommendations": [], "total_found": 0}

    mock_response = MagicMock()

    async def get_json():
        return mock_data

    mock_response.json.side_effect = get_json

    mock_session = MagicMock()
    mock_session.post.return_value = AsyncContextManager(return_value=mock_response)
    sys.modules["aiohttp"].ClientSession.return_value = AsyncContextManager(
        return_value=mock_session
    )

    result = asyncio.run(tool.discover_tools("unlikely query", limit=5))

    assert "Found 0 tools" in result or "Following tools are found" in result


def test_discover_tools_error():
    """Test proper error handling on network failure."""
    tool = MCPDiscoveryTool(api_url="http://test-api.com")

    # Force an error in ClientSession init
    sys.modules["aiohttp"].ClientSession.return_value = AsyncContextManager(
        error=Exception("Connection refused")
    )

    result = asyncio.run(tool.discover_tools("crash"))

    assert "Error discovering tools: Connection refused" in result
