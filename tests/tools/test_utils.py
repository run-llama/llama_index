"""Test utils."""
from llama_index.tools.utils import create_schema_from_function
from typing import List


def test_create_schema_from_function() -> None:
    """Test create schema from function."""

    def test_fn(x: int, y: int, z: List[str]) -> None:
        """Test function."""
        pass

    SchemaCls = create_schema_from_function("test_schema", test_fn)
    schema = SchemaCls.schema()
    assert schema["properties"]["x"]["type"] == "integer"
    assert schema["properties"]["y"]["type"] == "integer"
    assert schema["properties"]["z"]["type"] == "array"

    SchemaCls = create_schema_from_function("test_schema", test_fn, [("a", bool, 1)])
    schema = SchemaCls.schema()
    assert schema["properties"]["a"]["type"] == "boolean"
