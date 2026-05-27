import base64
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient
import pytest

from llama_index.core.llms import AudioBlock, ImageBlock, TextBlock
from llama_index.tools.mcp import BasicMCPClient
from llama_index.tools.mcp.client import enable_sse
from mcp import types
from pydantic import AnyUrl


# Path to the test server script - adjust as needed
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")


@pytest.fixture(scope="session")
def client() -> BasicMCPClient:
    """Create a basic MCP client connected to the test server."""
    return BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)


@pytest.mark.asyncio
async def test_list_tools(client: BasicMCPClient):
    """Test listing tools from the server."""
    tools = await client.list_tools()

    # Check that we got a list of tools
    assert isinstance(tools, types.ListToolsResult)
    assert len(tools.tools) > 0

    # Verify some expected tools are present
    tool_names = [tool.name for tool in tools.tools]
    assert "echo" in tool_names
    assert "add" in tool_names
    assert "get_time" in tool_names


@pytest.mark.asyncio
async def test_call_tools(client: BasicMCPClient):
    """Test calling various tools."""
    # Test echo tool
    result = await client.call_tool("echo", {"message": "Hello, World!"})
    assert result.content[0].text == "Echo: Hello, World!"

    # Test add tool
    result = await client.call_tool("add", {"a": 5, "b": 7})
    assert result.content[0].text == "12.0"

    # Test get_time tool (just verify it returns something)
    result = await client.call_tool("get_time", {})
    assert isinstance(result.content[0].text, str)
    assert len(result.content[0].text) > 0


@pytest.mark.asyncio
async def test_list_resources(client: BasicMCPClient):
    """Test listing resources from the server."""
    resources = await client.list_resources()

    # Check that we got a list of resources
    assert isinstance(resources, types.ListResourcesResult)
    assert len(resources.resources) > 0

    # Verify some expected resources are present
    resource_names = [str(resource.uri) for resource in resources.resources]
    assert "config://app" in resource_names
    assert "help://usage" in resource_names
    assert "counter://value" in resource_names
    assert "weather://current" in resource_names


@pytest.mark.asyncio
async def test_read_resources(client: BasicMCPClient):
    """Test reading various resources."""
    # Test static resource
    resource = await client.read_resource("config://app")
    assert isinstance(resource, types.ReadResourceResult)
    assert resource.contents[0].mimeType == "text/plain"
    config_text = resource.contents[0].text
    assert "app_name" in config_text
    assert "MCP Test Server" in config_text

    # Test parametrized resource
    resource = await client.read_resource("users://123/profile")
    profile_text = resource.contents[0].text
    assert "Test User" in profile_text
    assert "test@example.com" in profile_text


@pytest.mark.asyncio
async def test_list_prompts(client: BasicMCPClient):
    """Test listing prompts from the server."""
    prompts = await client.list_prompts()

    # Check that we got a list of prompts
    assert isinstance(prompts, types.ListPromptsResult)
    assert len(prompts.prompts) > 0

    # Verify some expected prompts are present
    prompt_names = [prompt.name for prompt in prompts.prompts]
    assert "simple_greeting" in prompt_names
    assert "personalized_greeting" in prompt_names
    assert "analyze_data" in prompt_names


@pytest.mark.asyncio
async def test_get_prompts(client: BasicMCPClient):
    """Test getting various prompts."""
    # Test simple prompt
    result = await client.get_prompt("simple_greeting")
    assert len(result) > 0

    # Test prompt with arguments
    result = await client.get_prompt("personalized_greeting", {"name": "Tester"})
    assert len(result) > 0
    assert "Tester" in result[0].content

    # Test multi-message prompt
    result = await client.get_prompt("analyze_data", {"data": "1,2,3,4,5"})
    assert len(result) > 1
    assert any("1,2,3,4,5" in msg.content for msg in result)


@pytest.mark.asyncio
async def test_resource_updates_via_tool(client: BasicMCPClient):
    """Test updating a resource via a tool and reading the changes."""
    # First read the initial user profile
    resource = await client.read_resource("users://123/profile")
    profile1 = resource.contents[0].text
    assert "Test User" in profile1

    # Update the user via the tool
    result = await client.call_tool(
        "update_user", {"user_id": "123", "name": "Updated User"}
    )

    profile2 = result.content[0].text
    assert "Updated User" in profile2
    assert "Test User" not in profile2


@pytest.mark.asyncio
async def test_default_in_memory_storage():
    """Test the default in-memory token storage."""
    # Create client with OAuth using default storage
    client = BasicMCPClient.with_oauth(
        "python",
        args=[SERVER_SCRIPT],
        client_name="Test Client",
        redirect_uris=["http://localhost:3000/callback"],
        redirect_handler=lambda url: None,  # Do nothing in test
        callback_handler=lambda: ("fake_code", None),  # Return fake code
    )

    # Just verify initialization works
    assert client.auth is not None


@pytest.mark.asyncio
async def test_use_provided_http_client():
    """Test the use of a provided HTTP client."""
    # Create client
    provided_http_client = AsyncClient()
    client = BasicMCPClient(
        "python", args=[SERVER_SCRIPT], timeout=5, http_client=provided_http_client
    )
    # Just verify initialization works
    assert client.client_provided is True
    assert client.http_client is provided_http_client


@pytest.mark.asyncio
async def test_use_provided_http_client_with_oauth():
    """Test the use of a provided HTTP client."""
    # Create client with OAuth
    provided_http_client = AsyncClient()
    client = BasicMCPClient.with_oauth(
        "python",
        args=[SERVER_SCRIPT],
        client_name="Test Client",
        redirect_uris=["http://localhost:3000/callback"],
        redirect_handler=lambda url: None,  # Do nothing in test
        callback_handler=lambda: ("fake_code", None),  # Return fake code
        http_client=provided_http_client,
    )
    # Just verify initialization works
    assert client.client_provided is True
    assert client.http_client is provided_http_client
    assert provided_http_client.auth is not None


@pytest.mark.asyncio
async def test_long_running_task(client: BasicMCPClient):
    """Test a long-running task with progress updates."""
    # This will run a task that takes a few seconds and reports progress
    current_progress = 0
    current_message = ""
    expected_total = None

    async def progress_callback(progress: float, total: float, message: str):
        nonlocal current_progress
        nonlocal current_message
        nonlocal expected_total

        current_progress = progress
        current_message = message
        expected_total = total

    result = await client.call_tool(
        "long_task", {"steps": 3}, progress_callback=progress_callback
    )
    assert "Completed 3 steps" in result.content[0].text
    assert current_progress == 3.0
    assert current_progress == expected_total
    assert current_message == "Processing step 3/3"


@pytest.mark.asyncio
async def test_image_return_value(client: BasicMCPClient):
    """Test tools that return images."""
    result = await client.call_tool(
        "generate_image", {"width": 50, "height": 50, "color": "blue"}
    )

    # Check that we got image data back
    assert isinstance(result, types.CallToolResult)
    assert len(result.content[0].data) > 0


def test_enable_sse():
    """Test the enable_sse function with various URL formats."""
    # Test query parameter detection (composio style)
    assert enable_sse("https://example.com/api?transport=sse") is True
    assert enable_sse("http://localhost:8080?transport=sse&other=param") is True
    assert enable_sse("https://example.com/api?other=param&transport=sse") is True

    # Test path suffix detection
    assert enable_sse("https://example.com/sse") is True
    assert enable_sse("http://localhost:8080/api/sse") is True
    assert enable_sse("https://example.com/sse/") is True

    # Test path containing /sse/
    assert enable_sse("https://example.com/api/sse/v1") is True
    assert enable_sse("http://localhost:8080/sse/events") is True
    assert enable_sse("https://example.com/v1/sse/stream") is True

    # Test non-SSE URLs
    assert enable_sse("https://example.com/api") is False
    assert enable_sse("http://localhost:8080") is False
    assert enable_sse("https://example.com/api?transport=http") is False
    assert enable_sse("https://example.com/assets") is False

    # Test edge cases
    assert enable_sse("https://example.com/sse-like") is False
    assert enable_sse("https://example.com/my-sse") is False
    assert enable_sse("https://example.com/api?sse=true") is False

    # Test with multiple transport values (should use first one)
    assert enable_sse("https://example.com?transport=sse&transport=http") is True

    # Test command-style inputs (non-URL)
    assert enable_sse("python") is False
    assert enable_sse("/usr/bin/python") is False


def _make_mock_client() -> BasicMCPClient:
    return BasicMCPClient("python", args=["server.py"], timeout=5)


def _make_prompt_result(role: str, content) -> types.GetPromptResult:
    return types.GetPromptResult(
        messages=[types.PromptMessage(role=role, content=content)]
    )


@asynccontextmanager
async def _mock_session(prompt_result: types.GetPromptResult):
    session = AsyncMock()
    session.get_prompt = AsyncMock(return_value=prompt_result)
    yield session


@pytest.mark.asyncio
async def test_get_prompt_embedded_text_resource():
    """EmbeddedResource with TextResourceContents yields a TextBlock."""
    resource = types.TextResourceContents(
        uri=AnyUrl("file:///hello.txt"), text="hello from resource"
    )
    embedded = types.EmbeddedResource(type="resource", resource=resource)
    prompt_result = _make_prompt_result("user", embedded)
    client = _make_mock_client()

    with patch.object(
        client, "_run_session", return_value=_mock_session(prompt_result)
    ):
        messages = await client.get_prompt("test_prompt")

    assert len(messages) == 1
    assert isinstance(messages[0].blocks[0], TextBlock)
    assert messages[0].blocks[0].text == "hello from resource"


@pytest.mark.asyncio
async def test_get_prompt_embedded_image_blob_resource():
    """EmbeddedResource with BlobResourceContents (image) yields an ImageBlock."""
    fake_png = base64.b64encode(b"\x89PNG\r\n").decode()
    resource = types.BlobResourceContents(
        uri=AnyUrl("file:///photo.png"),
        blob=fake_png,
        mimeType="image/png",
    )
    embedded = types.EmbeddedResource(type="resource", resource=resource)
    prompt_result = _make_prompt_result("user", embedded)
    client = _make_mock_client()

    with patch.object(
        client, "_run_session", return_value=_mock_session(prompt_result)
    ):
        messages = await client.get_prompt("test_prompt")

    assert len(messages) == 1
    assert isinstance(messages[0].blocks[0], ImageBlock)
    assert messages[0].blocks[0].image_mimetype == "image/png"


@pytest.mark.asyncio
async def test_get_prompt_embedded_audio_blob_resource():
    """EmbeddedResource with BlobResourceContents (audio) yields an AudioBlock."""
    fake_mp3 = base64.b64encode(b"ID3\x03").decode()
    resource = types.BlobResourceContents(
        uri=AnyUrl("file:///clip.mp3"),
        blob=fake_mp3,
        mimeType="audio/mpeg",
    )
    embedded = types.EmbeddedResource(type="resource", resource=resource)
    prompt_result = _make_prompt_result("user", embedded)
    client = _make_mock_client()

    with patch.object(
        client, "_run_session", return_value=_mock_session(prompt_result)
    ):
        messages = await client.get_prompt("test_prompt")

    assert len(messages) == 1
    assert isinstance(messages[0].blocks[0], AudioBlock)


@pytest.mark.asyncio
async def test_get_prompt_embedded_unknown_blob_resource():
    """EmbeddedResource with BlobResourceContents (unknown type) yields a TextBlock."""
    fake_data = base64.b64encode(b"\x00\x01\x02").decode()
    resource = types.BlobResourceContents(
        uri=AnyUrl("file:///data.bin"),
        blob=fake_data,
        mimeType="application/octet-stream",
    )
    embedded = types.EmbeddedResource(type="resource", resource=resource)
    prompt_result = _make_prompt_result("user", embedded)
    client = _make_mock_client()

    with patch.object(
        client, "_run_session", return_value=_mock_session(prompt_result)
    ):
        messages = await client.get_prompt("test_prompt")

    assert len(messages) == 1
    assert isinstance(messages[0].blocks[0], TextBlock)
    assert "application/octet-stream" in messages[0].blocks[0].text


@pytest.mark.asyncio
async def test_get_prompt_audio_content():
    """AudioContent yields an AudioBlock."""
    fake_wav = base64.b64encode(b"RIFF").decode()
    audio = types.AudioContent(type="audio", data=fake_wav, mimeType="audio/wav")
    prompt_result = _make_prompt_result("user", audio)
    client = _make_mock_client()

    with patch.object(
        client, "_run_session", return_value=_mock_session(prompt_result)
    ):
        messages = await client.get_prompt("test_prompt")

    assert len(messages) == 1
    assert isinstance(messages[0].blocks[0], AudioBlock)
    assert messages[0].blocks[0].format == "wav"


@pytest.mark.asyncio
async def test_get_prompt_resource_link():
    """ResourceLink yields a TextBlock containing the URI."""
    link = types.ResourceLink(
        type="resource_link",
        uri=AnyUrl("file:///docs/readme.md"),
        name="README",
    )
    prompt_result = _make_prompt_result("user", link)
    client = _make_mock_client()

    with patch.object(
        client, "_run_session", return_value=_mock_session(prompt_result)
    ):
        messages = await client.get_prompt("test_prompt")

    assert len(messages) == 1
    assert isinstance(messages[0].blocks[0], TextBlock)
    assert "readme.md" in messages[0].blocks[0].text
