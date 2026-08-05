"""Test for the standalone schema converter."""

import json
from llama_index.tools.mcp.schema_converter import SchemaConverter

def test_schema_converter():
    """Test that the schema converter works standalone."""
    
    # Simple schema test
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"]
    }
    
    converter = SchemaConverter()
    model = converter.create_model_from_json_schema(schema, "TestModel")
    
    # Test that we can create an instance
    instance = model(name="test", age=25)
    assert instance.name == "test"
    assert instance.age == 25
    
    print("Schema converter test passed!")

if __name__ == "__main__":
    test_schema_converter()
