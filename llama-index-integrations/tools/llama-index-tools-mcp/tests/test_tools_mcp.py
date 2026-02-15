import os
import pytest

from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.tools.mcp import (
    BasicMCPClient,
    McpToolSpec,
    get_tools_from_mcp_url,
    aget_tools_from_mcp_url,
)

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


# --- Tests for global_partial_params ---


def test_global_partial_params_removes_fields_from_schema(client: BasicMCPClient):
    """Test that global_partial_params removes specified fields from the tool schema."""
    # Create tool spec without global_partial_params
    tool_spec_full = McpToolSpec(client, allowed_tools=["add"])
    tools_full = tool_spec_full.to_tool_list()
    assert len(tools_full) == 1

    full_schema = tools_full[0].metadata.fn_schema.model_json_schema()
    assert "a" in full_schema["properties"]
    assert "b" in full_schema["properties"]
    assert set(full_schema["required"]) == {"a", "b"}

    # Create tool spec with global_partial_params for parameter 'a'
    tool_spec_partial = McpToolSpec(
        client, allowed_tools=["add"], global_partial_params={"a": 5.0}
    )
    tools_partial = tool_spec_partial.to_tool_list()
    assert len(tools_partial) == 1

    partial_schema = tools_partial[0].metadata.fn_schema.model_json_schema()
    assert "a" not in partial_schema["properties"]
    assert "b" in partial_schema["properties"]
    assert partial_schema["required"] == ["b"]


@pytest.mark.asyncio
async def test_global_partial_params_removes_fields_from_schema_async(
    client: BasicMCPClient,
):
    """Test that global_partial_params removes specified fields from the tool schema (async version)."""
    # Create tool spec without global_partial_params
    tool_spec_full = McpToolSpec(client, allowed_tools=["add"])
    tools_full = await tool_spec_full.to_tool_list_async()
    assert len(tools_full) == 1

    full_schema = tools_full[0].metadata.fn_schema.model_json_schema()
    assert "a" in full_schema["properties"]
    assert "b" in full_schema["properties"]

    # Create tool spec with global_partial_params for parameter 'a'
    tool_spec_partial = McpToolSpec(
        client, allowed_tools=["add"], global_partial_params={"a": 5.0}
    )
    tools_partial = await tool_spec_partial.to_tool_list_async()
    assert len(tools_partial) == 1

    partial_schema = tools_partial[0].metadata.fn_schema.model_json_schema()
    assert "a" not in partial_schema["properties"]
    assert "b" in partial_schema["properties"]


def test_global_partial_params_applies_to_all_tools(client: BasicMCPClient):
    """Test that global_partial_params applies to all tools."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add", "update_user"],
        global_partial_params={"a": 1.0, "user_id": "global"},
    )
    tools = tool_spec.to_tool_list()
    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # add tool should have 'a' removed (has 'a' param)
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" not in add_schema["properties"]
    assert "b" in add_schema["properties"]

    # update_user tool should have 'user_id' removed (has 'user_id' param)
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert "user_id" not in update_schema["properties"]
    assert "name" in update_schema["properties"]
    assert "email" in update_schema["properties"]


def test_global_partial_params_empty_dict(client: BasicMCPClient):
    """Test that an empty global_partial_params dict doesn't affect the schema."""
    tool_spec_empty = McpToolSpec(
        client, allowed_tools=["add"], global_partial_params={}
    )
    tools_empty = tool_spec_empty.to_tool_list()

    tool_spec_none = McpToolSpec(
        client, allowed_tools=["add"], global_partial_params=None
    )
    tools_none = tool_spec_none.to_tool_list()

    schema_empty = tools_empty[0].metadata.fn_schema.model_json_schema()
    schema_none = tools_none[0].metadata.fn_schema.model_json_schema()

    # Both should have the same properties
    assert schema_empty["properties"].keys() == schema_none["properties"].keys()
    assert schema_empty["required"] == schema_none["required"]


def test_global_partial_params_nonexistent_field(client: BasicMCPClient):
    """Test that global_partial_params with non-existent fields doesn't break the tool."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add"],
        global_partial_params={"nonexistent_param": "value", "a": 5.0},
    )
    tools = tool_spec.to_tool_list()
    assert len(tools) == 1

    schema = tools[0].metadata.fn_schema.model_json_schema()
    # 'a' should be removed
    assert "a" not in schema["properties"]
    # 'b' should still be present
    assert "b" in schema["properties"]


# --- Tests for partial_params_by_tool ---


def test_partial_params_by_tool_removes_fields_for_specific_tool(
    client: BasicMCPClient,
):
    """Test that partial_params_by_tool removes fields only for specified tools."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add", "update_user"],
        partial_params_by_tool={"add": {"a": 5.0}},
    )
    tools = tool_spec.to_tool_list()
    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # add tool should have 'a' removed
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" not in add_schema["properties"]
    assert "b" in add_schema["properties"]

    # update_user tool should be unaffected (not in partial_params_by_tool)
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert "user_id" in update_schema["properties"]
    assert "name" in update_schema["properties"]
    assert "email" in update_schema["properties"]


@pytest.mark.asyncio
async def test_partial_params_by_tool_multiple_tools_async(client: BasicMCPClient):
    """Test that partial_params_by_tool can specify different params for different tools."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add", "update_user"],
        partial_params_by_tool={
            "add": {"a": 5.0},
            "update_user": {"user_id": "123"},
        },
    )
    tools = await tool_spec.to_tool_list_async()
    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # add tool should have 'a' removed
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" not in add_schema["properties"]
    assert "b" in add_schema["properties"]

    # update_user tool should have user_id removed
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert "user_id" not in update_schema["properties"]
    assert "name" in update_schema["properties"]
    assert "email" in update_schema["properties"]


# --- Tests for combined global_partial_params and partial_params_by_tool ---


def test_global_and_by_tool_merge_behavior(client: BasicMCPClient):
    """Test that global params are always applied and by_tool params merge with them."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add", "update_user"],
        global_partial_params={"a": 1.0, "user_id": "global"},
        partial_params_by_tool={"add": {"b": 2.0}},  # Add 'b' to add tool
    )
    tools = tool_spec.to_tool_list()
    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # add tool: global 'a' AND tool-specific 'b' should both be removed
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" not in add_schema["properties"]
    assert "b" not in add_schema["properties"]

    # update_user: only global params apply (user_id removed)
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert "user_id" not in update_schema["properties"]
    assert "name" in update_schema["properties"]
    assert "email" in update_schema["properties"]


def test_by_tool_overrides_global_value(client: BasicMCPClient):
    """Test that by_tool can override a global param value."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add"],
        global_partial_params={"a": 1.0},
        partial_params_by_tool={"add": {"a": 999.0}},  # Override 'a' value
    )
    tools = tool_spec.to_tool_list()

    # The param 'a' should still be removed (with new value 999.0)
    add_schema = tools[0].metadata.fn_schema.model_json_schema()
    assert "a" not in add_schema["properties"]
    assert "b" in add_schema["properties"]

    # Verify the FunctionTool has the overridden value
    assert tools[0].partial_params["a"] == 999.0


def test_by_tool_removes_global_param_with_none(client: BasicMCPClient):
    """Test that setting a param to None in by_tool removes it for that tool."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add", "update_user"],
        global_partial_params={"a": 1.0, "user_id": "global"},
        partial_params_by_tool={"add": {"a": None}},  # Remove 'a' for add tool
    )
    tools = tool_spec.to_tool_list()
    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # add tool: 'a' should NOT be removed (None removes it from partial_params)
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" in add_schema["properties"]
    assert "b" in add_schema["properties"]

    # update_user: global params still apply (user_id removed)
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert "user_id" not in update_schema["properties"]


@pytest.mark.asyncio
async def test_comprehensive_merge_behavior_async(client: BasicMCPClient):
    """
    Comprehensive test for merge behavior with all use cases:
    1. Global params apply to all tools
    2. By_tool adds new params (merged with global)
    3. By_tool overrides global param values
    4. By_tool removes global params with None
    """
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add", "update_user", "echo"],
        global_partial_params={"a": 10.0, "user_id": "global_user"},
        partial_params_by_tool={
            "add": {"b": 20.0},  # Merge: add 'b' to global 'a'
            "update_user": {
                "user_id": None,
                "name": "Override Name",
            },  # Remove global user_id, add name
            # echo: no override, uses only global params
        },
    )
    tools = await tool_spec.to_tool_list_async()
    assert len(tools) == 3

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")
    echo_tool = next(t for t in tools if t.metadata.name == "echo")

    # Test 1: add tool - global 'a' AND tool-specific 'b' both removed
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" not in add_schema["properties"]
    assert "b" not in add_schema["properties"]
    print(add_tool.partial_params)
    assert add_tool.partial_params == {"a": 10.0, "user_id": "global_user", "b": 20.0}

    # Test 2: update_user - global user_id removed (None), name added, global 'a' still there
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert (
        "user_id" in update_schema["properties"]
    )  # user_id NOT removed (None cancelled it)
    assert "name" not in update_schema["properties"]  # name IS removed
    assert "email" in update_schema["properties"]
    assert update_user_tool.partial_params == {"a": 10.0, "name": "Override Name"}

    # Test 3: echo - only global params apply (but echo has 'message', not 'a' or 'user_id')
    echo_schema = echo_tool.metadata.fn_schema.model_json_schema()
    assert "message" in echo_schema["properties"]
    assert echo_tool.partial_params == {"a": 10.0, "user_id": "global_user"}


def test_by_tool_empty_dict_keeps_global(client: BasicMCPClient):
    """Test that empty dict in by_tool still applies global params."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add"],
        global_partial_params={"a": 1.0},
        partial_params_by_tool={"add": {}},  # Empty override
    )
    tools = tool_spec.to_tool_list()

    # Global params should still apply
    schema = tools[0].metadata.fn_schema.model_json_schema()
    assert "a" not in schema["properties"]
    assert "b" in schema["properties"]


def test_by_tool_all_global_params_removed_with_none(client: BasicMCPClient):
    """Test that all global params can be removed for a tool using None values."""
    tool_spec = McpToolSpec(
        client,
        allowed_tools=["add"],
        global_partial_params={"a": 1.0, "b": 2.0},
        partial_params_by_tool={"add": {"a": None, "b": None}},  # Remove all
    )
    tools = tool_spec.to_tool_list()

    # No params should be removed
    schema = tools[0].metadata.fn_schema.model_json_schema()
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]
    # partial_params should be None (empty dict is converted to None)
    assert tools[0].partial_params is None or tools[0].partial_params == {}


# --- Tests for get_tools_from_mcp_url and aget_tools_from_mcp_url utility functions ---
# These tests verify that allowed_tools and partial params propagate correctly through the utility functions


def test_get_tools_from_mcp_url_propagates_allowed_tools(client: BasicMCPClient):
    """Test that allowed_tools propagates correctly from utility function to McpToolSpec."""
    tools = get_tools_from_mcp_url(
        "unused", client=client, allowed_tools=["echo", "add"]
    )

    assert len(tools) == 2
    tool_names = {tool.metadata.name for tool in tools}
    assert tool_names == {"echo", "add"}


def test_get_tools_from_mcp_url_propagates_global_partial_params(
    client: BasicMCPClient,
):
    """Test that global_partial_params propagates correctly to remove fields from schema."""
    tools = get_tools_from_mcp_url(
        "unused",
        client=client,
        allowed_tools=["add"],
        global_partial_params={"a": 10.0},
    )

    assert len(tools) == 1
    tool = tools[0]

    # Verify schema is modified
    schema = tool.metadata.fn_schema.model_json_schema()
    assert "a" not in schema["properties"]
    assert "b" in schema["properties"]
    assert schema["required"] == ["b"]

    # Verify partial_params is set
    assert tool.partial_params == {"a": 10.0}


def test_get_tools_from_mcp_url_propagates_partial_params_by_tool(
    client: BasicMCPClient,
):
    """Test that partial_params_by_tool propagates correctly for specific tools."""
    tools = get_tools_from_mcp_url(
        "unused",
        client=client,
        allowed_tools=["add", "update_user"],
        partial_params_by_tool={
            "add": {"a": 5.0},
            "update_user": {"user_id": "123"},
        },
    )

    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # Verify add tool
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" not in add_schema["properties"]
    assert add_tool.partial_params == {"a": 5.0}

    # Verify update_user tool
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert "user_id" not in update_schema["properties"]
    assert update_user_tool.partial_params == {"user_id": "123"}


def test_get_tools_from_mcp_url_propagates_combined_params(client: BasicMCPClient):
    """Test that both global and tool-specific params propagate and merge correctly."""
    tools = get_tools_from_mcp_url(
        "unused",
        client=client,
        allowed_tools=["add", "update_user"],
        global_partial_params={"a": 1.0, "user_id": "global"},
        partial_params_by_tool={"add": {"b": 2.0}},
    )

    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # add tool: merges global and tool-specific params
    add_schema = add_tool.metadata.fn_schema.model_json_schema()
    assert "a" not in add_schema["properties"]
    assert "b" not in add_schema["properties"]
    assert add_tool.partial_params == {"a": 1.0, "user_id": "global", "b": 2.0}

    # update_user: only global params
    update_schema = update_user_tool.metadata.fn_schema.model_json_schema()
    assert "user_id" not in update_schema["properties"]
    assert update_user_tool.partial_params == {"a": 1.0, "user_id": "global"}


@pytest.mark.asyncio
async def test_aget_tools_from_mcp_url_propagates_allowed_tools(client: BasicMCPClient):
    """Test that allowed_tools propagates correctly through async utility function."""
    tools = await aget_tools_from_mcp_url(
        "unused", client=client, allowed_tools=["echo", "add"]
    )

    assert len(tools) == 2
    tool_names = {tool.metadata.name for tool in tools}
    assert tool_names == {"echo", "add"}


@pytest.mark.asyncio
async def test_aget_tools_from_mcp_url_propagates_global_partial_params(
    client: BasicMCPClient,
):
    """Test that global_partial_params propagates correctly through async utility function."""
    tools = await aget_tools_from_mcp_url(
        "unused",
        client=client,
        allowed_tools=["add"],
        global_partial_params={"a": 10.0},
    )

    assert len(tools) == 1
    tool = tools[0]

    # Verify schema is modified
    schema = tool.metadata.fn_schema.model_json_schema()
    assert "a" not in schema["properties"]
    assert "b" in schema["properties"]

    # Verify partial_params is set
    assert tool.partial_params == {"a": 10.0}


@pytest.mark.asyncio
async def test_aget_tools_from_mcp_url_propagates_partial_params_by_tool(
    client: BasicMCPClient,
):
    """Test that partial_params_by_tool propagates correctly through async utility function."""
    tools = await aget_tools_from_mcp_url(
        "unused",
        client=client,
        allowed_tools=["add", "update_user"],
        partial_params_by_tool={
            "add": {"a": 5.0},
            "update_user": {"user_id": "123"},
        },
    )

    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # Verify both tools have correct partial_params
    assert add_tool.partial_params == {"a": 5.0}
    assert update_user_tool.partial_params == {"user_id": "123"}


@pytest.mark.asyncio
async def test_aget_tools_from_mcp_url_propagates_combined_params(
    client: BasicMCPClient,
):
    """Test that combined params propagate and merge correctly through async utility function."""
    tools = await aget_tools_from_mcp_url(
        "unused",
        client=client,
        allowed_tools=["add", "update_user"],
        global_partial_params={"a": 1.0, "user_id": "global"},
        partial_params_by_tool={"add": {"b": 2.0}},
    )

    assert len(tools) == 2

    add_tool = next(t for t in tools if t.metadata.name == "add")
    update_user_tool = next(t for t in tools if t.metadata.name == "update_user")

    # Verify merged params
    assert add_tool.partial_params == {"a": 1.0, "user_id": "global", "b": 2.0}
    assert update_user_tool.partial_params == {"a": 1.0, "user_id": "global"}
