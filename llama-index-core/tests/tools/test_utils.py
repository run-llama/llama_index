"""Test utils."""

from typing import List, Annotated
import datetime

from llama_index.core.bridge.pydantic import Field
from llama_index.core.tools.utils import create_schema_from_function


def test_create_schema_from_function() -> None:
    """Test create schema from function."""

    def test_fn(x: int, y: int, z: List[str]) -> None:
        """Test function."""

    SchemaCls = create_schema_from_function("test_schema", test_fn)
    schema = SchemaCls.model_json_schema()
    assert schema["properties"]["x"]["type"] == "integer"
    assert schema["properties"]["y"]["type"] == "integer"
    assert schema["properties"]["z"]["type"] == "array"
    assert schema["required"] == ["x", "y", "z"]

    SchemaCls = create_schema_from_function("test_schema", test_fn, [("a", bool, 1)])
    schema = SchemaCls.model_json_schema()
    assert schema["properties"]["a"]["type"] == "boolean"

    def test_fn2(x: int = 1) -> None:
        """Optional input."""

    SchemaCls = create_schema_from_function("test_schema", test_fn2)
    schema = SchemaCls.model_json_schema()
    assert "required" not in schema


def test_create_schema_from_function_with_field() -> None:
    """Test create_schema_from_function with pydantic.Field."""

    def tmp_function(x: int = Field(3, description="An integer")) -> str:
        return str(x)

    schema = create_schema_from_function("TestSchema", tmp_function)
    actual_schema = schema.model_json_schema()

    assert "x" in actual_schema["properties"]
    assert actual_schema["properties"]["x"]["type"] == "integer"
    assert actual_schema["properties"]["x"]["default"] == 3
    assert actual_schema["properties"]["x"]["description"] == "An integer"

    # Test the created schema
    instance = schema()
    assert instance.x == 3  # type: ignore

    instance = schema(x=5)
    assert instance.x == 5  # type: ignore


def test_create_schema_from_function_with_typing_annotated() -> None:
    """Test create_schema_from_function with pydantic.Field."""

    def tmp_function(x: Annotated[int, "An integer"] = 3) -> str:
        return str(x)

    schema = create_schema_from_function("TestSchema", tmp_function)
    actual_schema = schema.model_json_schema()

    assert "x" in actual_schema["properties"]
    assert actual_schema["properties"]["x"]["type"] == "integer"
    assert actual_schema["properties"]["x"]["default"] == 3
    assert actual_schema["properties"]["x"]["description"] == "An integer"

    # Test the created schema
    instance = schema()
    assert instance.x == 3  # type: ignore

    instance = schema(x=5)
    assert instance.x == 5  # type: ignore


def test_create_schema_from_function_with_field_annotated() -> None:
    """Test create_schema_from_function with Annotated[pydantic.Field]."""

    def tmp_function(x: Annotated[int, Field(description="An integer")] = 3) -> str:
        return str(x)

    schema = create_schema_from_function("TestSchema", tmp_function)
    actual_schema = schema.model_json_schema()

    assert "x" in actual_schema["properties"]
    assert actual_schema["properties"]["x"]["type"] == "integer"
    assert actual_schema["properties"]["x"]["default"] == 3
    assert actual_schema["properties"]["x"]["description"] == "An integer"

    # Test the created schema
    instance = schema()
    assert instance.x == 3  # type: ignore

    instance = schema(x=5)
    assert instance.x == 5  # type: ignore


def test_create_schema_with_date_and_metadata():
    def sample_func(
        birth_date: Annotated[
            datetime.date,
            Field(
                description="The birth date",
                json_schema_extra={"example": "2000-01-01"},
            ),
        ],
        timestamp: Annotated[
            datetime.datetime,
            Field(
                description="Timestamp",
                json_schema_extra={"example": "2023-05-12T08:00:00"},
            ),
        ],
    ):
        pass

    schema = create_schema_from_function("TestSchema", sample_func)

    properties = schema.model_json_schema()["properties"]

    assert properties["birth_date"]["format"] == "date"
    assert properties["birth_date"]["description"] == "The birth date"
    assert properties["birth_date"]["example"] == "2000-01-01"

    assert properties["timestamp"]["format"] == "date-time"
    assert properties["timestamp"]["example"] == "2023-05-12T08:00:00"
