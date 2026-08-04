"""Tests for JsonSchemaToPydantic — JSON Schema -> Pydantic model conversion."""

from typing import Dict, Any, Literal, get_args

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.tools import JsonSchemaToPydantic

# Shaped like a real tool inputSchema: a $ref to a nested model, an array,
# an enum, and an optional (anyOf + null) field.
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


def test_basic_model_construction() -> None:
    converter = JsonSchemaToPydantic()
    model = converter.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")

    assert issubclass(model, BaseModel)
    assert set(model.model_fields) == {"name", "tags", "color", "note"}

    # required vs optional
    assert model.model_fields["name"].is_required()
    assert model.model_fields["color"].is_required()
    assert not model.model_fields["note"].is_required()


def test_nested_ref_resolves_to_a_submodel() -> None:
    converter = JsonSchemaToPydantic()
    model = converter.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")

    name_type = model.model_fields["name"].annotation
    assert isinstance(name_type, type) and issubclass(name_type, BaseModel)
    assert set(name_type.model_fields) == {"first", "last"}
    # The submodel is also cached under its $defs name.
    assert "PersonName" in converter.properties_cache


def test_array_items_are_typed() -> None:
    converter = JsonSchemaToPydantic()
    model = converter.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")
    # tags isn't required, so its annotation is Optional[List[str]]; unwrap
    # the Optional and check the list item type.
    tags_type = model.model_fields["tags"].annotation
    list_type = next(a for a in get_args(tags_type) if a is not type(None))
    assert get_args(list_type) == (str,)


def test_remove_model_fields() -> None:
    converter = JsonSchemaToPydantic()
    model = converter.create_model_from_json_schema(TOOL_SCHEMA, "Tool_Schema")
    trimmed = converter.remove_model_fields(model, {"note", "tags"}, "Tool_Schema")
    assert set(trimmed.model_fields) == {"name", "color"}


def test_enum_resolves_to_literal() -> None:
    """
    Regression coverage carried over from
    https://github.com/run-llama/llama_index/issues/20109 — enum entries in
    anyOf schemas must resolve to Literal types, not fall through to str.
    """
    converter = JsonSchemaToPydantic()

    assert (
        converter._resolve_union_option({"enum": ["red", "green", "blue"]}, {})
        == Literal["red", "green", "blue"]
    )

    resolved = converter._resolve_union_type(
        {"anyOf": [{"enum": ["a", "b"]}, {"type": "null"}]}, {}
    )
    args = get_args(resolved)
    assert type(None) in args
    assert Literal["a", "b"] in args


def test_additional_properties_variants() -> None:
    converter = JsonSchemaToPydantic()

    schema_false = {"type": "object", "additionalProperties": False}
    assert not converter._is_simple_object(schema_false)
    assert converter._create_dict_type(schema_false, {}) == Dict[str, Any]

    schema_typed = {"type": "object", "additionalProperties": {"type": "string"}}
    assert converter._is_simple_object(schema_typed)
    assert converter._create_dict_type(schema_typed, {}) == Dict[str, str]


def test_json_schema_round_trip_defaults() -> None:
    """Optional field with a default keeps the default."""
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "default": 10},
        },
    }
    model = JsonSchemaToPydantic().create_model_from_json_schema(schema, "M")
    assert model().limit == 10
