import pytest
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from typing import Optional, List
from llama_index.llms.google_genai.conversion.tools import ToolSchemaConverter
import google.genai.types as types
from tests.conftest import Schema
from tests.conftest import BlogPost


def dummy_function(x: int, y: int) -> int:
    """
    Adds two numbers.

    Args:
        x: The first number.
        y: The second number.

    """
    return x + y


@pytest.fixture
def converter(mock_genai_client):
    return ToolSchemaConverter(client=mock_genai_client)


def test_to_function_declaration(converter, mock_genai_client):
    tool = FunctionTool.from_defaults(fn=dummy_function)
    decl = converter.to_function_declaration(tool)

    assert decl.name == "dummy_function"
    assert decl.description == "Adds two numbers."
    assert decl.parameters is not None


def test_optional_value_gemini(converter):
    class OptionalContent(BaseModel):
        content: Optional[str] = Field(default=None)
        content2: str | None

    tool = FunctionTool.from_defaults(
        fn=lambda x: x, fn_schema=OptionalContent, name="OptionalContent"
    )

    decl = converter.to_function_declaration(tool)

    # Verify nullable fields
    props = decl.parameters.properties
    assert props["content"].nullable is True
    assert props["content"].default is None
    assert props["content2"].nullable is True
    assert props["content2"].default is None


def test_nested_list_schema(converter):
    class Item(BaseModel):
        name: str

    class Container(BaseModel):
        items: List[Item]

    tool = FunctionTool.from_defaults(
        fn=lambda x: x, fn_schema=Container, name="Container"
    )

    decl = converter.to_function_declaration(tool)

    assert decl.name == "Container"
    props = decl.parameters.properties
    assert props["items"].type == types.Type.ARRAY
    assert "name" in props["items"].items.properties


def test_nested_pydantic_schema_has_no_defs_and_required(converter):
    """
    Validate nested schema conversion produces required fields.

    We mainly assert that the generated Gemini schema has the expected top-level
    properties and required fields for a nested model.
    """
    # Old tests asserted `$defs` exists in Pydantic schema output.
    assert "$defs" in Schema.model_json_schema()

    tool = FunctionTool.from_defaults(fn=lambda x: x, fn_schema=Schema, name="Schema")

    decl = converter.to_function_declaration(tool)

    assert decl.name == "Schema"
    assert decl.parameters is not None
    assert list(decl.parameters.properties) == ["schema_name", "tables"]
    assert decl.parameters.required == ["schema_name", "tables"]


def test_tool_schema_converter_preserves_name_description_and_params(converter) -> None:
    """
    This asserts our wrapper behavior, not SDK internals.
    """

    class Poem(BaseModel):
        content: str = Field(description="Poem content")

    def write_poem(content: str) -> str:
        """A simple poem."""
        return content

    tool = FunctionTool.from_defaults(fn=write_poem, fn_schema=Poem, name="Poem")
    decl = converter.to_function_declaration(tool)

    assert decl.name == "Poem"
    assert decl.description == "A simple poem."
    assert decl.parameters is not None
    assert "content" in decl.parameters.properties
    assert decl.parameters.properties["content"].description == "Poem content"
    assert decl.parameters.required == ["content"]


def test_optional_lists_nested_gemini_schema_and_live_generation(converter) -> None:
    """
    Validate deep nested schema conversion for optional lists.

    This test focuses on schema conversion depth. Live structured generation is
    covered in `tests_new/integration/test_google_genai.py`.
    """
    # Old test explicitly asserted `$defs` exists in pydantic schema.
    assert "$defs" in BlogPost.model_json_schema()

    tool = FunctionTool.from_defaults(
        fn=lambda x: x, fn_schema=BlogPost, name="BlogPost"
    )
    decl = converter.to_function_declaration(tool)

    assert decl.name == "BlogPost"
    assert decl.parameters is not None

    contents_property = decl.parameters.properties["contents"]
    assert contents_property.type == types.Type.ARRAY

    content_items = contents_property.items
    assert "text" in content_items.properties
    assert "image" in content_items.properties
    assert "video" in content_items.properties
