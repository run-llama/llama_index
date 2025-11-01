import os
import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Path to the test server script - adjust as needed
SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "server.py")


@pytest.fixture(scope="session")
def client() -> BasicMCPClient:
    """Create a basic MCP client connected to the test server."""
    return BasicMCPClient("python", args=[SERVER_SCRIPT], timeout=5)


def test_class():
    names_of_base_classes = [b.__name__ for b in McpToolSpec.__mro__]
    assert BaseToolSpec.__name__ in names_of_base_classes


def test_get_tools(client: BasicMCPClient):
    tool_spec = McpToolSpec(client)
    tools = tool_spec.to_tool_list()
    assert len(tools) > 0

    tool_spec = McpToolSpec(client, include_resources=True)
    tools_plus_resources = tool_spec.to_tool_list()
    assert len(tools_plus_resources) > len(tools)


@pytest.mark.asyncio
async def test_get_tools_async(client: BasicMCPClient):
    tool_spec = McpToolSpec(client)
    tools = await tool_spec.to_tool_list_async()
    assert len(tools) > 0

    tool_spec = McpToolSpec(client, include_resources=True)
    tools_plus_resources = await tool_spec.to_tool_list_async()
    assert len(tools_plus_resources) > len(tools)


def test_get_single_tool(client: BasicMCPClient):
    tool_spec = McpToolSpec(client, allowed_tools=["echo"])

    tools = tool_spec.to_tool_list()
    assert len(tools) == 1
    assert tools[0].metadata.name == "echo"


@pytest.mark.asyncio
async def test_get_single_tool_async(client: BasicMCPClient):
    tool_spec = McpToolSpec(client, allowed_tools=["echo"])

    tools = await tool_spec.to_tool_list_async()
    assert len(tools) == 1
    assert tools[0].metadata.name == "echo"


def test_get_zero_tools(client: BasicMCPClient):
    tool_spec = McpToolSpec(client, allowed_tools=[])
    tools = tool_spec.to_tool_list()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_get_zero_tools_async(client: BasicMCPClient):
    tool_spec = McpToolSpec(client, allowed_tools=[])
    tools = await tool_spec.to_tool_list_async()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_pydantic_models_tool(client: BasicMCPClient):
    """Test the test_pydantic tool with Pydantic models that use custom type aliases and Literal types."""
    tool_spec = McpToolSpec(client, allowed_tools=["test_pydantic"])
    tools = await tool_spec.to_tool_list_async()

    tool = tools[0]
    assert tool.metadata.name == "test_pydantic"

    result = await tool.async_fn(
        name={"name": "John Doe"},
        method={"method": "POST"},
        lst={"lst": [1, 2, 3, 4, 5]},
    )

    assert (
        "Name: John Doe, Method: POST, List: [1, 2, 3, 4, 5]" in result.content[0].text
    )


def test_pydantic_models_schema_structure(client: BasicMCPClient):
    """Test that the test_pydantic tool generates correct schema structure."""
    tool = McpToolSpec(client, allowed_tools=["test_pydantic"]).to_tool_list()[0]

    assert tool.metadata.name == "test_pydantic"
    assert tool.metadata.fn_schema is not None

    json_schema = tool.metadata.fn_schema.model_json_schema()
    assert all(key in json_schema["properties"] for key in ["name", "method", "lst"])
    assert all(
        key in json_schema["$defs"] for key in ["TestName", "TestMethod", "TestList"]
    )

    # Check property types
    assert json_schema["properties"]["name"]["$ref"] == "#/$defs/TestName"
    assert json_schema["properties"]["method"]["$ref"] == "#/$defs/TestMethod"
    assert json_schema["properties"]["lst"]["$ref"] == "#/$defs/TestList"

    # Check model types
    assert json_schema["$defs"]["TestName"]["properties"]["name"]["type"] == "string"
    assert json_schema["$defs"]["TestList"]["properties"]["lst"]["type"] == "array"
    assert (
        json_schema["$defs"]["TestList"]["properties"]["lst"]["items"]["type"]
        == "integer"
    )


def test_schema_structure_exact_match(client: BasicMCPClient):
    """Test that the generated schema structure exactly matches the expected format."""
    json_schema = (
        McpToolSpec(client, allowed_tools=["test_pydantic"])
        .to_tool_list()[0]
        .metadata.fn_schema.model_json_schema()
    )

    assert json_schema["type"] == "object"
    assert set(json_schema["required"]) == {"name", "method", "lst"}


def test_additional_properties_false_parsing(client: BasicMCPClient):
    """Test that schemas with additionalProperties: false are parsed correctly."""
    from typing import Dict, Any

    tool_spec = McpToolSpec(client)

    # Test case 1: additionalProperties is False
    schema_false = {"type": "object", "additionalProperties": False}
    assert not tool_spec._is_simple_object(schema_false)
    result_type = tool_spec._create_dict_type(schema_false, {})
    assert result_type == Dict[str, Any]

    # Test case 2: additionalProperties is None
    schema_none = {"type": "object", "additionalProperties": None}
    result_type = tool_spec._create_dict_type(schema_none, {})
    assert result_type == Dict[str, Any]

    # Test case 3: additionalProperties is a dict (should be treated as simple object)
    schema_dict = {"type": "object", "additionalProperties": {"type": "string"}}
    assert tool_spec._is_simple_object(schema_dict)
    result_type = tool_spec._create_dict_type(schema_dict, {})
    assert result_type == Dict[str, str]


@pytest.mark.asyncio
async def test_resource_tool_uses_uri_not_name(client: BasicMCPClient):
    """
    Tests that a tool from a static resource is executable.

    This test is designed to FAIL with the current bug, because the tool's
    internal function is created with the resource's name ('get_app_config')
    instead of its URI ('config://app'), causing the client call to fail.
    """
    tool_spec = McpToolSpec(
        client, allowed_tools=["get_app_config"], include_resources=True
    )
    tools = await tool_spec.to_tool_list_async()

    assert len(tools) == 1
    tool = tools[0]
    assert tool.metadata.name == "get_app_config"

    # This call will fail due to the bug.
    result = await tool.acall()
    assert "MCP Test Server" in result.content


@pytest.mark.asyncio
async def test_dynamic_resource_template_tool_is_created(client: BasicMCPClient):
    """
    Tests that a tool is created for a dynamic resource template.
    """
    tool_spec = McpToolSpec(client, include_resources=True)
    tools = await tool_spec.to_tool_list_async()

    # The server.py defines a dynamic resource template named 'get_user_profile'.
    # This should now be found.
    tool_names = {t.metadata.name for t in tools}
    assert "get_user_profile" in tool_names
