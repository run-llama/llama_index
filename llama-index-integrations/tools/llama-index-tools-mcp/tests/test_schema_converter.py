from typing import Dict, List, Literal, get_args

from llama_index.tools.mcp import JsonSchemaToPydantic


def test_basic_object_schema():
    converter = JsonSchemaToPydantic()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "A name"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    model = converter.create_model_from_json_schema(schema, "Person")

    assert model.__name__ == "Person"
    fields = model.model_fields
    assert "name" in fields
    assert "age" in fields
    assert fields["name"].is_required()
    assert not fields["age"].is_required()

    instance = model(name="Alice")
    assert instance.name == "Alice"
    assert instance.age is None


def test_ref_and_nested_models():
    converter = JsonSchemaToPydantic()
    schema = {
        "type": "object",
        "properties": {
            "address": {"$ref": "#/$defs/Address"},
        },
        "required": ["address"],
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["street", "city"],
            },
        },
    }
    model = converter.create_model_from_json_schema(schema, "Person")

    assert "Address" in converter.properties_cache
    address_model = converter.properties_cache["Address"]
    assert "street" in address_model.model_fields
    assert "city" in address_model.model_fields

    json_schema = model.model_json_schema()
    assert json_schema["properties"]["address"]["$ref"] == "#/$defs/Address"
    assert "Address" in json_schema["$defs"]


def test_anyof_union():
    converter = JsonSchemaToPydantic()
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
            },
        },
        "required": ["value"],
    }
    model = converter.create_model_from_json_schema(schema, "UnionModel")

    value_field = model.model_fields["value"]
    args = get_args(value_field.annotation)
    assert str in args
    assert int in args


def test_enum_to_literal():
    converter = JsonSchemaToPydantic()
    schema = {
        "type": "object",
        "properties": {
            "color": {"enum": ["red", "green", "blue"]},
        },
        "required": ["color"],
    }
    model = converter.create_model_from_json_schema(schema, "ColorModel")

    color_field = model.model_fields["color"]
    assert color_field.annotation == Literal["red", "green", "blue"]


def test_array_of_items():
    converter = JsonSchemaToPydantic()
    schema = {
        "type": "object",
        "properties": {
            "numbers": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": ["numbers"],
    }
    model = converter.create_model_from_json_schema(schema, "ArrayModel")

    numbers_field = model.model_fields["numbers"]
    assert numbers_field.annotation == List[int]


def test_remove_model_fields():
    converter = JsonSchemaToPydantic()
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"},
            "c": {"type": "boolean"},
        },
        "required": ["a", "b", "c"],
    }
    model = converter.create_model_from_json_schema(schema, "Original")
    reduced = converter.remove_model_fields(model, {"b"}, "Reduced")

    assert "a" in reduced.model_fields
    assert "b" not in reduced.model_fields
    assert "c" in reduced.model_fields
    assert reduced.__name__ == "Reduced"


def test_optional_field_with_default():
    converter = JsonSchemaToPydantic()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer", "default": 42},
        },
        "required": ["name"],
    }
    model = converter.create_model_from_json_schema(schema, "Defaults")

    assert model.model_fields["name"].is_required()
    assert not model.model_fields["count"].is_required()
    assert model.model_fields["count"].default == 42

    instance = model(name="test")
    assert instance.count == 42


def test_additional_properties_dict():
    converter = JsonSchemaToPydantic()
    schema = {
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["metadata"],
    }
    model = converter.create_model_from_json_schema(schema, "DictModel")

    metadata_field = model.model_fields["metadata"]
    assert metadata_field.annotation == Dict[str, str]


def test_matches_mcp_tool_spec_output(client):
    """Verify that standalone converter produces the same schema as McpToolSpec."""
    from llama_index.tools.mcp import McpToolSpec

    schema = {
        "type": "object",
        "properties": {
            "name": {"$ref": "#/$defs/Name"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "tags"],
        "$defs": {
            "Name": {
                "type": "object",
                "properties": {"first": {"type": "string"}, "last": {"type": "string"}},
                "required": ["first", "last"],
            },
        },
    }

    converter = JsonSchemaToPydantic()
    standalone_model = converter.create_model_from_json_schema(schema, "TestSchema")

    tool_spec = McpToolSpec(client)
    spec_model = tool_spec.create_model_from_json_schema(schema, "TestSchema")

    assert standalone_model.model_json_schema() == spec_model.model_json_schema()
