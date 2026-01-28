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


# --- Tests for partial_params ---


def test_partial_params_removes_fields_from_schema(client: BasicMCPClient):
    """Test that partial_params removes specified fields from the tool schema."""
    # Create tool spec without partial_params
    tool_spec_full = McpToolSpec(client, allowed_tools=["add"])
    tools_full = tool_spec_full.to_tool_list()
    assert len(tools_full) == 1

    full_schema = tools_full[0].metadata.fn_schema.model_json_schema()
    assert "a" in full_schema["properties"]
    assert "b" in full_schema["properties"]
    assert set(full_schema["required"]) == {"a", "b"}

    # Create tool spec with partial_params for parameter 'a'
    tool_spec_partial = McpToolSpec(
        client, allowed_tools=["add"], partial_params={"a": 5.0}
    )
    tools_partial = tool_spec_partial.to_tool_list()
    assert len(tools_partial) == 1

    partial_schema = tools_partial[0].metadata.fn_schema.model_json_schema()
    assert "a" not in partial_schema["properties"]
    assert "b" in partial_schema["properties"]
    assert partial_schema["required"] == ["b"]


@pytest.mark.asyncio
async def test_partial_params_removes_fields_from_schema_async(
    client: BasicMCPClient,
):
    """Test that partial_params removes specified fields from the tool schema (async version)."""
    # Create tool spec without partial_params
    tool_spec_full = McpToolSpec(client, allowed_tools=["add"])
    tools_full = await tool_spec_full.to_tool_list_async()
    assert len(tools_full) == 1

    full_schema = tools_full[0].metadata.fn_schema.model_json_schema()
    assert "a" in full_schema["properties"]
    assert "b" in full_schema["properties"]

    # Create tool spec with partial_params for parameter 'a'
    tool_spec_partial = McpToolSpec(
        client, allowed_tools=["add"], partial_params={"a": 5.0}
    )
    tools_partial = await tool_spec_partial.to_tool_list_async()
    assert len(tools_partial) == 1

    partial_schema = tools_partial[0].metadata.fn_schema.model_json_schema()
    assert "a" not in partial_schema["properties"]
    assert "b" in partial_schema["properties"]


def test_partial_params_multiple_fields(client: BasicMCPClient):
    """Test that partial_params can remove multiple fields from the tool schema."""
    tool_spec = McpToolSpec(
        client, allowed_tools=["update_user"], partial_params={"user_id": "123"}
    )
    tools = tool_spec.to_tool_list()
    assert len(tools) == 1

    schema = tools[0].metadata.fn_schema.model_json_schema()
    # user_id should be removed
    assert "user_id" not in schema["properties"]
    # name and email should still be present
    assert "name" in schema["properties"]
    assert "email" in schema["properties"]
    # required list should not contain user_id
    assert "user_id" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_partial_params_with_pydantic_models(client: BasicMCPClient):
    """Test that partial_params works correctly with tools using Pydantic models."""
    tool_spec_full = McpToolSpec(client, allowed_tools=["test_pydantic"])
    tools_full = await tool_spec_full.to_tool_list_async()

    full_schema = tools_full[0].metadata.fn_schema.model_json_schema()
    assert all(key in full_schema["properties"] for key in ["name", "method", "lst"])

    # Remove one of the Pydantic model fields
    tool_spec_partial = McpToolSpec(
        client,
        allowed_tools=["test_pydantic"],
        partial_params={"name": {"name": "Default Name"}},
    )
    tools_partial = await tool_spec_partial.to_tool_list_async()

    partial_schema = tools_partial[0].metadata.fn_schema.model_json_schema()
    assert "name" not in partial_schema["properties"]
    assert "method" in partial_schema["properties"]
    assert "lst" in partial_schema["properties"]
    assert set(partial_schema["required"]) == {"method", "lst"}


def test_partial_params_empty_dict(client: BasicMCPClient):
    """Test that an empty partial_params dict doesn't affect the schema."""
    tool_spec_empty = McpToolSpec(client, allowed_tools=["add"], partial_params={})
    tools_empty = tool_spec_empty.to_tool_list()

    tool_spec_none = McpToolSpec(client, allowed_tools=["add"], partial_params=None)
    tools_none = tool_spec_none.to_tool_list()

    schema_empty = tools_empty[0].metadata.fn_schema.model_json_schema()
    schema_none = tools_none[0].metadata.fn_schema.model_json_schema()

    # Both should have the same properties
    assert schema_empty["properties"].keys() == schema_none["properties"].keys()
    assert schema_empty["required"] == schema_none["required"]


def test_partial_params_nonexistent_field(client: BasicMCPClient):
    """Test that partial_params with non-existent fields doesn't break the tool."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add"],
        partial_params={"nonexistent_param": "value", "a": 5.0},
    )
    tools = tool_spec.to_tool_list()
    assert len(tools) == 1

    schema = tools[0].metadata.fn_schema.model_json_schema()
    # 'a' should be removed
    assert "a" not in schema["properties"]
    # 'b' should still be present
    assert "b" in schema["properties"]
    # nonexistent_param was never there, so no effect


@pytest.mark.asyncio
async def test_partial_params_all_fields_removed(client: BasicMCPClient):
    """Test that partial_params can remove all parameters from a tool."""
    tool_spec = McpToolSpec(
        client, allowed_tools=["add"], partial_params={"a": 1.0, "b": 2.0}
    )
    tools = await tool_spec.to_tool_list_async()
    assert len(tools) == 1

    schema = tools[0].metadata.fn_schema.model_json_schema()
    # Both parameters should be removed
    assert "a" not in schema["properties"]
    assert "b" not in schema["properties"]
    # No required fields
    assert len(schema.get("required", [])) == 0
    # Properties should be empty
    assert len(schema["properties"]) == 0


def test_partial_params_with_optional_fields(client: BasicMCPClient):
    """Test that partial_params works with tools that have optional fields."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["update_user"],
        partial_params={"user_id": "123", "name": "Partial Name"},
    )
    tools = tool_spec.to_tool_list()
    assert len(tools) == 1

    schema = tools[0].metadata.fn_schema.model_json_schema()
    # Both user_id and name should be removed
    assert "user_id" not in schema["properties"]
    assert "name" not in schema["properties"]
    # email should still be present
    assert "email" in schema["properties"]


@pytest.mark.asyncio
async def test_partial_params_multiple_tools(client: BasicMCPClient):
    """Test that partial_params applies to all tools when multiple tools are loaded."""
    tool_spec = McpToolSpec(
        client, allowed_tools=["add", "update_user"], partial_params={"user_id": "123"}
    )
    tools = await tool_spec.to_tool_list_async()
    assert len(tools) == 2

    # Find each tool by name
    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # add tool should not be affected (it doesn't have user_id parameter)
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" in add_schema["properties"]
    assert "b" in add_schema["properties"]

    # update_user tool should have user_id removed
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert "user_id" not in update_schema["properties"]
    assert "name" in update_schema["properties"]
    assert "email" in update_schema["properties"]
