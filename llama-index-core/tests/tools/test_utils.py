"""Test utils."""

from typing import List, Annotated
import datetime

import pytest

from llama_index.core.bridge.pydantic import Field, ValidationError
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


def test_create_schema_from_function_keeps_annotated_field_constraints() -> None:
    """
    `Annotated[T, Field(...)]` constraints must reach the schema and validation.

    Regression test: the `FieldInfo` built for the parameter carried only the
    description and `json_schema_extra`, so `ge`/`le` never reached the tool schema and
    out-of-range arguments were accepted silently.
    """

    def tmp_function(
        seats: Annotated[int, Field(description="number of seats", ge=1, le=8)],
    ) -> str:
        return str(seats)

    schema = create_schema_from_function("TestSchema", tmp_function)
    properties = schema.model_json_schema()["properties"]

    assert properties["seats"]["description"] == "number of seats"
    assert properties["seats"]["minimum"] == 1
    assert properties["seats"]["maximum"] == 8
    assert schema.model_json_schema()["required"] == ["seats"]

    assert schema(seats=8).seats == 8  # type: ignore
    with pytest.raises(ValidationError):
        schema(seats=9)


def test_create_schema_from_function_keeps_annotated_field_constraints_with_default() -> (
    None
):
    """The same holds when the parameter also has a Python default."""

    def tmp_function(
        name: Annotated[str, Field(description="guest name", min_length=2)] = "ok",
    ) -> str:
        return name

    schema = create_schema_from_function("TestSchema", tmp_function)
    properties = schema.model_json_schema()["properties"]

    assert properties["name"]["minLength"] == 2
    assert properties["name"]["description"] == "guest name"
    assert properties["name"]["default"] == "ok"

    with pytest.raises(ValidationError):
        schema(name="a")


def test_annotated_field_can_be_reused_across_schemas() -> None:
    """
    Merging constraints must not mutate the `FieldInfo` the caller shared.

    If it did, the second schema would inherit whatever the first one recorded.
    """
    shared_field = Field(description="An integer", ge=1)

    def first(x: Annotated[int, shared_field]) -> str:
        return str(x)

    def second(y: Annotated[int, shared_field], z: str = "z") -> str:
        return str(y)

    first_schema = create_schema_from_function("FirstSchema", first)
    second_schema = create_schema_from_function("SecondSchema", second)

    assert first_schema.model_json_schema()["properties"]["x"]["minimum"] == 1
    assert second_schema.model_json_schema()["properties"]["y"]["minimum"] == 1
    assert shared_field.description == "An integer"


def test_create_schema_skips_variadic_args_kwargs() -> None:
    def fn(q: str, *args: int, **kwargs: int) -> None:
        pass

    schema = create_schema_from_function("TestSchema", fn).model_json_schema()

    assert "args" not in schema["properties"]
    assert "kwargs" not in schema["properties"]
    assert schema["required"] == ["q"]


def test_create_schema_keeps_real_param_named_kwargs() -> None:
    def fn(kwargs: dict) -> None:
        pass

    schema = create_schema_from_function("TestSchema", fn).model_json_schema()

    assert "kwargs" in schema["properties"]
    assert schema["required"] == ["kwargs"]


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


def test_create_schema_from_function_with_param_descriptions() -> None:
    """Test that param_descriptions land in the generated JSON schema."""

    def tmp_function(x: int, y: str = "a") -> str:
        return str(x)

    schema = create_schema_from_function(
        "TestSchema",
        tmp_function,
        param_descriptions={"x": "An integer", "y": "A string"},
    )
    actual_schema = schema.model_json_schema()

    assert actual_schema["properties"]["x"]["description"] == "An integer"
    assert actual_schema["properties"]["y"]["description"] == "A string"
    assert actual_schema["properties"]["y"]["default"] == "a"
    assert actual_schema["required"] == ["x"]


def test_param_descriptions_do_not_override_an_explicit_description() -> None:
    """A description on the parameter itself wins over the fallback."""

    def tmp_function(
        x: Annotated[int, "From the annotation"],
        y: str = Field("a", description="From the field"),
    ) -> str:
        return str(x)

    schema = create_schema_from_function(
        "TestSchema",
        tmp_function,
        param_descriptions={"x": "Fallback", "y": "Fallback"},
    )
    actual_schema = schema.model_json_schema()

    assert actual_schema["properties"]["x"]["description"] == "From the annotation"
    assert actual_schema["properties"]["y"]["description"] == "From the field"


def test_param_descriptions_do_not_mutate_the_callers_field() -> None:
    """Filling in a description must not write back to the shared FieldInfo."""
    shared_field = Field("a")

    def tmp_function(x: str = shared_field) -> str:
        return x

    schema = create_schema_from_function(
        "TestSchema", tmp_function, param_descriptions={"x": "A string"}
    )

    assert schema.model_json_schema()["properties"]["x"]["description"] == "A string"
    assert shared_field.description is None
