"""
Tests for property graph schema utility functions.

Validates that auto-generated Entity/Relation Pydantic models produce
JSON schemas compatible with OpenAI structured outputs and Google Gemini.
"""

import json
from typing import Literal

from llama_index.core.indices.property_graph.transformations.utils import (
    _clean_additional_properties,
    get_entity_class,
    get_relation_class,
)


def _schema_contains(schema: dict, key: str, value: object) -> bool:
    """Recursively check whether *schema* contains *key* mapped to *value*."""
    if isinstance(schema, dict):
        if schema.get(key) is value:
            return True
        return any(_schema_contains(v, key, value) for v in schema.values())
    if isinstance(schema, list):
        return any(_schema_contains(item, key, value) for item in schema)
    return False


# -- _clean_additional_properties ------------------------------------------


def test_clean_additional_properties_sets_true_to_false():
    schema = {"additionalProperties": True, "properties": {"x": {"type": "string"}}}
    _clean_additional_properties(schema)
    assert schema["additionalProperties"] is False


def test_clean_additional_properties_nested():
    schema = {
        "properties": {
            "inner": {
                "additionalProperties": True,
                "type": "object",
            }
        }
    }
    _clean_additional_properties(schema)
    assert schema["properties"]["inner"]["additionalProperties"] is False


def test_clean_additional_properties_ignores_false():
    schema = {"additionalProperties": False}
    _clean_additional_properties(schema)
    assert schema["additionalProperties"] is False


def test_clean_additional_properties_ignores_absent():
    schema = {"properties": {"x": {"type": "string"}}}
    _clean_additional_properties(schema)
    assert "additionalProperties" not in schema


def test_clean_additional_properties_handles_list():
    schema = {"anyOf": [{"additionalProperties": True}, {"type": "null"}]}
    _clean_additional_properties(schema)
    assert schema["anyOf"][0]["additionalProperties"] is False


# -- get_entity_class (no props → no additionalProperties issue) -----------


def test_entity_class_without_props_has_no_additional_properties_true():
    entities = Literal["PERSON", "LOCATION"]
    cls = get_entity_class(entities, None, strict=True)
    schema = cls.model_json_schema()
    assert not _schema_contains(schema, "additionalProperties", True)


# -- get_entity_class (with props → fix applied) --------------------------


def test_entity_class_with_props_has_no_additional_properties_true():
    entities = Literal["PERSON", "LOCATION"]
    cls = get_entity_class(entities, ["age", "occupation"], strict=True)
    schema = cls.model_json_schema()
    # The fix should have cleaned additionalProperties: true → false
    assert not _schema_contains(schema, "additionalProperties", True), (
        f"Schema still contains additionalProperties: true:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def test_entity_class_with_props_non_strict():
    cls = get_entity_class(str, ["age"], strict=False)
    schema = cls.model_json_schema()
    assert not _schema_contains(schema, "additionalProperties", True)


# -- get_relation_class (no props → no issue) ------------------------------


def test_relation_class_without_props_has_no_additional_properties_true():
    relations = Literal["USED_BY", "PART_OF"]
    cls = get_relation_class(relations, None, strict=True)
    schema = cls.model_json_schema()
    assert not _schema_contains(schema, "additionalProperties", True)


# -- get_relation_class (with props → fix applied) ------------------------


def test_relation_class_with_props_has_no_additional_properties_true():
    relations = Literal["USED_BY", "PART_OF"]
    cls = get_relation_class(relations, ["weight", "source"], strict=True)
    schema = cls.model_json_schema()
    assert not _schema_contains(schema, "additionalProperties", True), (
        f"Schema still contains additionalProperties: true:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def test_relation_class_with_props_non_strict():
    cls = get_relation_class(str, ["weight"], strict=False)
    schema = cls.model_json_schema()
    assert not _schema_contains(schema, "additionalProperties", True)


# -- Models still validate correctly after the fix -------------------------


def test_entity_model_with_props_roundtrips():
    entities = Literal["PERSON", "LOCATION"]
    cls = get_entity_class(entities, ["age", "occupation"], strict=True)
    instance = cls(type="PERSON", name="Alice", properties={"age": 30})
    assert instance.type == "PERSON"
    assert instance.name == "Alice"
    assert instance.properties == {"age": 30}


def test_relation_model_with_props_roundtrips():
    relations = Literal["USED_BY", "PART_OF"]
    cls = get_relation_class(relations, ["weight"], strict=True)
    instance = cls(type="USED_BY", properties={"weight": 0.9})
    assert instance.type == "USED_BY"
    assert instance.properties == {"weight": 0.9}
