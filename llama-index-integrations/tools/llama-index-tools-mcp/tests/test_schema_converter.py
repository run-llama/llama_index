"""
Tests for JsonSchemaToPydantic — the schema converter used without an MCP client.

These deliberately never construct a ``ClientSession`` (or mock one), which is the
whole point of https://github.com/run-llama/llama_index/issues/22510.
"""

from typing import get_args

from pydantic import BaseModel

from llama_index.tools.mcp import JsonSchemaToPydantic, McpToolSpec

# A schema shaped like a real MCP tool inputSchema: a $ref to a nested model,
# an array, an enum, and an optional (anyOf + null) field.
TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"$ref": "#/$defs/PersonName"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "color": {"enum": ["red", "green", "blue"]},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["name", "color"],
    "$defs": {
        "PersonName": {
            "type": "object",
            "properties": {
                "first": {"type": "string"},
                "last": {"type": "string"},
            },
            "required": ["first"],
        }
    },
}


def test_convert_without_a_client():
    """The converter builds a model from a tool schema with no client at all."""
    converter = JsonSchemaToPydantic()
    model = converter.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")

    assert issubclass(model, BaseModel)
    assert set(model.model_fields) == {"name", "tags", "color", "note"}

    # required vs optional
    assert model.model_fields["name"].is_required()
    assert model.model_fields["color"].is_required()
    assert not model.model_fields["note"].is_required()


def test_nested_ref_resolves_to_a_submodel():
    """A $ref to a $defs entry becomes a nested Pydantic model."""
    converter = JsonSchemaToPydantic()
    model = converter.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")

    name_type = model.model_fields["name"].annotation
    assert isinstance(name_type, type) and issubclass(name_type, BaseModel)
    assert set(name_type.model_fields) == {"first", "last"}
    # The submodel is also cached under its $defs name.
    assert "PersonName" in converter.properties_cache


def test_enum_resolves_to_literal():
    converter = JsonSchemaToPydantic()
    resolved = converter._resolve_field_type({"enum": ["red", "green", "blue"]}, {})
    from typing import Literal

    assert resolved == Literal["red", "green", "blue"]


def test_array_items_are_typed():
    converter = JsonSchemaToPydantic()
    model = converter.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")
    # tags isn't required, so its annotation is Optional[List[str]]; unwrap the
    # Optional and check the list item type is str.
    tags_type = model.model_fields["tags"].annotation
    list_type = next(a for a in get_args(tags_type) if a is not type(None))
    assert get_args(list_type) == (str,)


def test_remove_model_fields_standalone():
    converter = JsonSchemaToPydantic()
    model = converter.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")
    trimmed = converter.remove_model_fields(model, {"note", "tags"}, "Tool_Schema")
    assert set(trimmed.model_fields) == {"name", "color"}


def test_mcptoolspec_delegates_and_matches():
    """
    McpToolSpec produces the same model as the standalone converter, and never
    touches the client during conversion (a dummy object is enough here).
    """
    dummy_client = object()  # conversion must not read the client
    spec = McpToolSpec(dummy_client)

    from_spec = spec.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")
    from_converter = JsonSchemaToPydantic().create_model_from_json_schema(
        TOOL_SCHEMA, "Tool_Schema"
    )

    assert from_spec.model_json_schema() == from_converter.model_json_schema()
    # The spec's properties_cache is the converter's cache.
    assert spec.properties_cache is spec._schema_converter.properties_cache
