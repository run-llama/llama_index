import pytest
from typing import List, Optional
from pydantic import BaseModel

from llama_index.tools.artifact_editor.base import (
    ArtifactEditorToolSpec,
    JsonPatch,
    PatchOperation,
)


# Test models for testing purposes
class Address(BaseModel):
    """Address model for testing nested objects."""

    street: str
    city: str
    zipcode: str
    country: Optional[str] = None


class Person(BaseModel):
    """Person model for testing the artifact editor."""

    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = []
    address: Optional[Address] = None


class SimpleModel(BaseModel):
    """Simple model for basic testing."""

    value: str
    number: Optional[int] = None
    optional_number: Optional[int] = None


@pytest.fixture
def editor():
    return ArtifactEditorToolSpec(Person)


@pytest.fixture
def simple_editor():
    return ArtifactEditorToolSpec(SimpleModel)


def test_create_artifact(editor: ArtifactEditorToolSpec):
    """Test creating an initial artifact."""
    result = editor.create_artifact(
        name="John Doe", age=30, email="john@example.com", tags=["developer", "python"]
    )

    expected = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "tags": ["developer", "python"],
        "address": None,
    }

    assert result == expected
    assert editor.get_current_artifact() == expected


def test_create_artifact_with_nested_object(editor: ArtifactEditorToolSpec):
    """Test creating artifact with nested objects."""
    address_data = {
        "street": "123 Main St",
        "city": "Springfield",
        "zipcode": "12345",
        "country": None,
    }

    result = editor.create_artifact(name="Jane Doe", age=25, address=address_data)

    assert result["address"] == address_data
    assert isinstance(editor.current_artifact.address, Address)


def test_get_current_artifact(editor: ArtifactEditorToolSpec):
    """Test getting the current artifact."""
    # Test when no artifact exists
    assert editor.get_current_artifact() is None

    # Create an artifact and test retrieval
    editor.create_artifact(name="Test User", age=20)
    result = editor.get_current_artifact()

    expected = {
        "name": "Test User",
        "age": 20,
        "email": None,
        "tags": [],
        "address": None,
    }

    assert result == expected
    assert editor.get_current_artifact() == expected


def test_apply_patch_replace_operation(editor: ArtifactEditorToolSpec):
    """Test applying replace operations."""
    editor.create_artifact(name="John", age=30)

    patch = JsonPatch(
        operations=[
            PatchOperation(op="replace", path="/name", value="Jane"),
            PatchOperation(op="replace", path="/age", value=25),
        ]
    )

    result = editor.apply_patch(patch)

    assert result["name"] == "Jane"
    assert result["age"] == 25
    assert editor.get_current_artifact() == result


def test_apply_patch_add_operation(editor: ArtifactEditorToolSpec):
    """Test applying add operations."""
    editor.create_artifact(name="John", age=30, tags=["python"])

    patch = JsonPatch(
        operations=[
            PatchOperation(op="add", path="/email", value="john@example.com"),
            PatchOperation(op="add", path="/tags/1", value="developer"),
            PatchOperation(op="add", path="/tags/-", value="expert"),  # Append to array
        ]
    )

    result = editor.apply_patch(patch)

    assert result["email"] == "john@example.com"
    assert result["tags"] == ["python", "developer", "expert"]
    assert editor.get_current_artifact() == result


def test_apply_patch_remove_operation(editor: ArtifactEditorToolSpec):
    """Test applying remove operations."""
    editor.create_artifact(
        name="John",
        age=30,
        email="john@example.com",
        tags=["python", "developer", "expert"],
    )

    patch = JsonPatch(
        operations=[
            PatchOperation(op="remove", path="/email"),
            PatchOperation(op="remove", path="/tags/1"),  # Remove "developer"
        ]
    )

    result = editor.apply_patch(patch)

    assert result["email"] is None
    assert result["tags"] == ["python", "expert"]


def test_apply_patch_move_operation(simple_editor: ArtifactEditorToolSpec):
    """Test applying move operations."""
    simple_editor.create_artifact(value="test", number=42)

    patch = JsonPatch(
        operations=[
            PatchOperation(op="move", path="/optional_number", from_path="/number")
        ]
    )

    result = simple_editor.apply_patch(patch)

    # Note: This test assumes we're moving the value, not the key
    # The actual behavior depends on the implementation details
    assert result["number"] is None
    assert result["optional_number"] == 42


def test_apply_patch_copy_operation(editor: ArtifactEditorToolSpec):
    """Test applying copy operations."""
    editor.create_artifact(name="John", age=30)

    patch = JsonPatch(
        operations=[PatchOperation(op="copy", path="/email", from_path="/name")]
    )

    result = editor.apply_patch(patch)

    assert result["email"] == "John"
    assert result["name"] == "John"  # Original should still exist


def test_apply_patch_nested_paths(editor: ArtifactEditorToolSpec):
    """Test operations on nested object paths."""
    address_data = {"street": "123 Main St", "city": "Springfield", "zipcode": "12345"}
    editor.create_artifact(name="John", age=30, address=address_data)

    patch = JsonPatch(
        operations=[
            PatchOperation(op="replace", path="/address/city", value="New York"),
            PatchOperation(op="add", path="/address/country", value="USA"),
        ]
    )

    result = editor.apply_patch(patch)

    assert result["address"]["city"] == "New York"
    assert result["address"]["country"] == "USA"


def test_apply_patch_array_operations(editor: ArtifactEditorToolSpec):
    """Test various array operations."""
    editor.create_artifact(name="John", age=30, tags=["python", "java", "go"])

    patch = JsonPatch(
        operations=[
            PatchOperation(op="replace", path="/tags/1", value="javascript"),
            PatchOperation(op="add", path="/tags/0", value="rust"),
            PatchOperation(
                op="remove", path="/tags/3"
            ),  # Remove "java" (now at index 3)
        ]
    )

    result = editor.apply_patch(patch)

    # Expected: ["rust", "python", "javascript", "go"]
    assert "rust" in result["tags"]
    assert "javascript" in result["tags"]


def test_path_parsing():
    """Test path parsing functionality."""
    editor = ArtifactEditorToolSpec(Person)

    # Test basic path parsing
    assert editor._parse_path("/") == []
    assert editor._parse_path("/name") == ["name"]
    assert editor._parse_path("/tags/0") == ["tags", 0]
    assert editor._parse_path("/address/street") == ["address", "street"]

    # Test escaped characters
    assert editor._parse_path("/field~0name") == ["field~name"]
    assert editor._parse_path("/field~1name") == ["field/name"]


def test_invalid_path_format(editor: ArtifactEditorToolSpec):
    """Test error handling for invalid path formats."""
    editor.create_artifact(name="John", age=30)

    patch = JsonPatch(
        operations=[PatchOperation(op="replace", path="invalid_path", value="test")]
    )

    with pytest.raises(ValueError, match="Path must start with"):
        editor.apply_patch(patch)


def test_nonexistent_path(editor: ArtifactEditorToolSpec):
    """Test error handling for nonexistent paths."""
    editor.create_artifact(name="John", age=30)

    patch = JsonPatch(
        operations=[PatchOperation(op="replace", path="/nonexistent", value="test")]
    )

    with pytest.raises(ValueError):
        editor.apply_patch(patch)


def test_array_index_out_of_range(editor: ArtifactEditorToolSpec):
    """Test error handling for array index out of range."""
    editor.create_artifact(name="John", age=30, tags=["python"])

    patch = JsonPatch(
        operations=[PatchOperation(op="replace", path="/tags/5", value="test")]
    )

    with pytest.raises(ValueError, match="Failed to apply operation"):
        editor.apply_patch(patch)


def test_invalid_operation_type(editor: ArtifactEditorToolSpec):
    """Test error handling for invalid operation types."""
    editor.create_artifact(name="John", age=30)

    patch = JsonPatch(
        operations=[PatchOperation(op="invalid_op", path="/name", value="test")]
    )

    with pytest.raises(ValueError, match="Unknown operation"):
        editor.apply_patch(patch)


def test_move_without_from_path(editor: ArtifactEditorToolSpec):
    """Test error handling for move operation without from_path."""
    editor.create_artifact(name="John", age=30)

    patch = JsonPatch(
        operations=[
            PatchOperation(op="move", path="/name", value="test")  # Missing from_path
        ]
    )

    with pytest.raises(ValueError, match="'move' operation requires 'from_path'"):
        editor.apply_patch(patch)


def test_copy_without_from_path(editor: ArtifactEditorToolSpec):
    """Test error handling for copy operation without from_path."""
    editor.create_artifact(name="John", age=30)

    patch = JsonPatch(
        operations=[
            PatchOperation(op="copy", path="/email", value="test")  # Missing from_path
        ]
    )

    with pytest.raises(ValueError, match="'copy' operation requires 'from_path'"):
        editor.apply_patch(patch)


def test_patch_validation_error(editor: ArtifactEditorToolSpec):
    """Test error handling when patch results in invalid model."""
    editor.create_artifact(name="John", age=30)

    # Try to set age to a string, which should violate the model
    patch = JsonPatch(
        operations=[
            PatchOperation(op="replace", path="/name", value=None)  # Required field
        ]
    )

    with pytest.raises(ValueError, match="Patch resulted in invalid"):
        editor.apply_patch(patch)


def test_patch_from_dict(editor: ArtifactEditorToolSpec):
    """Test applying patch from dictionary format."""
    editor.create_artifact(name="John", age=30)

    patch_dict = {"operations": [{"op": "replace", "path": "/name", "value": "Jane"}]}

    result = editor.apply_patch(patch_dict)
    assert result["name"] == "Jane"


def test_patch_from_json_string(editor: ArtifactEditorToolSpec):
    """Test applying patch from JSON string format."""
    editor.create_artifact(name="John", age=30)

    patch_json = '{"operations": [{"op": "replace", "path": "/name", "value": "Jane"}]}'

    result = editor.apply_patch(patch_json)
    assert result["name"] == "Jane"


def test_to_tool_list(editor: ArtifactEditorToolSpec):
    """Test converting to tool list includes all expected tools."""
    tools = editor.to_tool_list()

    # Should have 3 tools: apply_patch, get_current_artifact, create_artifact
    assert len(tools) == 3

    tool_names = [tool.metadata.name for tool in tools]
    assert "apply_patch" in tool_names
    assert "get_current_artifact" in tool_names
    assert "create_artifact" in tool_names


def test_no_current_artifact_apply_patch(editor: ArtifactEditorToolSpec):
    """Test error when trying to apply patch without current artifact."""
    patch = JsonPatch(
        operations=[PatchOperation(op="replace", path="/name", value="Jane")]
    )

    with pytest.raises(AttributeError):
        editor.apply_patch(patch)


def test_complex_nested_operations(editor: ArtifactEditorToolSpec):
    """Test complex operations on deeply nested structures."""
    complex_data = {
        "name": "John",
        "age": 30,
        "address": {"street": "123 Main St", "city": "Springfield", "zipcode": "12345"},
        "tags": ["python", "developer"],
    }

    editor.create_artifact(**complex_data)

    patch = JsonPatch(
        operations=[
            PatchOperation(op="replace", path="/address/street", value="456 Oak Ave"),
            PatchOperation(op="add", path="/tags/-", value="senior"),
            PatchOperation(op="remove", path="/tags/0"),  # Remove "python"
        ]
    )

    result = editor.apply_patch(patch)

    assert result["address"]["street"] == "456 Oak Ave"
    assert "senior" in result["tags"]
    assert "python" not in result["tags"]
    assert result["age"] == 30


def test_set_invalid_field_path(editor: ArtifactEditorToolSpec):
    """Test setting a field that doesn't exist in the Pydantic model schema."""
    editor.create_artifact(name="John", age=30)

    # Try to add a field that doesn't exist in the Person model
    patch = JsonPatch(
        operations=[PatchOperation(op="add", path="/invalid_field", value="test")]
    )

    # This should raise an error since invalid_field is not in the Person model
    with pytest.raises(ValueError, match="Invalid field 'invalid_field'"):
        editor.apply_patch(patch)


def test_set_invalid_nested_field_path(editor: ArtifactEditorToolSpec):
    """Test setting a nested field that doesn't exist in the Pydantic model schema."""
    address_data = {"street": "123 Main St", "city": "Springfield", "zipcode": "12345"}
    editor.create_artifact(name="John", age=30, address=address_data)

    # Try to add a field that doesn't exist in the Address model
    patch = JsonPatch(
        operations=[
            PatchOperation(op="add", path="/address/invalid_nested_field", value="test")
        ]
    )

    # This should raise an error since invalid_nested_field is not in the Address model
    with pytest.raises(ValueError, match="Invalid field 'invalid_nested_field'"):
        editor.apply_patch(patch)


def test_valid_nested_field_addition(editor: ArtifactEditorToolSpec):
    """Test adding a valid nested field that exists in the model schema."""
    address_data = {"street": "123 Main St", "city": "Springfield", "zipcode": "12345"}
    editor.create_artifact(name="John", age=30, address=address_data)

    # Add the country field which exists in the Address model
    patch = JsonPatch(
        operations=[PatchOperation(op="add", path="/address/country", value="USA")]
    )

    result = editor.apply_patch(patch)
    assert result["address"]["country"] == "USA"


def test_validation_with_array_access(editor: ArtifactEditorToolSpec):
    """Test validation works correctly with array access patterns."""
    editor.create_artifact(name="John", age=30, tags=["python", "developer"])

    # Valid array operations should work
    patch = JsonPatch(
        operations=[
            PatchOperation(op="replace", path="/tags/0", value="rust"),
            PatchOperation(op="add", path="/tags/-", value="expert"),
        ]
    )

    result = editor.apply_patch(patch)
    assert result["tags"] == ["rust", "developer", "expert"]


def test_validation_does_not_affect_existing_operations(editor: ArtifactEditorToolSpec):
    """Test that validation doesn't break existing valid operations."""
    editor.create_artifact(name="John", age=30, email="john@example.com")

    # All these operations should still work
    patch = JsonPatch(
        operations=[
            PatchOperation(op="replace", path="/name", value="Jane"),
            PatchOperation(op="replace", path="/age", value=25),
            PatchOperation(op="remove", path="/email"),
        ]
    )

    result = editor.apply_patch(patch)
    assert result["name"] == "Jane"
    assert result["age"] == 25
    assert result["email"] is None


def test_move_operation_validates_target_path(editor: ArtifactEditorToolSpec):
    """Test that move operations validate the target path."""
    editor.create_artifact(name="John", age=30, email="john@example.com")

    # Try to move to an invalid field
    patch = JsonPatch(
        operations=[
            PatchOperation(op="move", path="/invalid_field", from_path="/email")
        ]
    )

    with pytest.raises(ValueError, match="Invalid field 'invalid_field'"):
        editor.apply_patch(patch)


def test_copy_operation_validates_target_path(editor: ArtifactEditorToolSpec):
    """Test that copy operations validate the target path."""
    editor.create_artifact(name="John", age=30, email="john@example.com")

    # Try to copy to an invalid field
    patch = JsonPatch(
        operations=[
            PatchOperation(op="copy", path="/invalid_field", from_path="/email")
        ]
    )

    with pytest.raises(ValueError, match="Invalid field 'invalid_field'"):
        editor.apply_patch(patch)
