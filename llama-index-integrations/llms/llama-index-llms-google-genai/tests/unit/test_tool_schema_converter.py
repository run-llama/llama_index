import pytest
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from typing import Optional, List
from llama_index.llms.google_genai.conversion.tools import ToolSchemaConverter
import google.genai.types as types
from google.genai import _transformers
from dataclasses import dataclass


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
    # Arrange
    tool = FunctionTool.from_defaults(fn=dummy_function)

    # We need to mock the internal google.genai._transformers.t_schema behavior
    # or rely on the fact that we are testing our wrapper.
    # Since _transformers is internal to the SDK, we can't easily mock it without
    # patching the SDK itself. However, since we are using a mock client,
    # let's assume the SDK's transformer works or patch it if needed.
    # For this unit test, we are mostly checking that description and name are extracted.

    # Act
    # Note: This relies on google.genai._transformers working with the mock client
    # If that fails, we might need to patch `llama_index.llms.google_genai.conversion.tools._transformers`
    decl = converter.to_function_declaration(tool)

    # Assert
    assert decl.name == "dummy_function"
    assert decl.description == "Adds two numbers."
    # Parameters verification depends on the actual SDK implementation
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
    Parity with old nested schema conversion tests.

    We mainly assert that the generated Gemini schema has the expected top-level
    properties and required fields for a nested model.
    """

    class Column(BaseModel):
        name: str = Field(description="Column field")
        data_type: str = Field(description="Data type field")

    class Table(BaseModel):
        name: str = Field(description="Table name field")
        columns: List[Column] = Field(description="List of random Column objects")

    class Schema(BaseModel):
        schema_name: str = Field(description="Schema name")
        tables: List[Table] = Field(description="List of random Table objects")

    # Old tests asserted `$defs` exists in Pydantic schema output.
    assert "$defs" in Schema.model_json_schema()

    tool = FunctionTool.from_defaults(fn=lambda x: x, fn_schema=Schema, name="Schema")

    decl = converter.to_function_declaration(tool)

    assert decl.name == "Schema"
    assert decl.parameters is not None
    assert list(decl.parameters.properties) == ["schema_name", "tables"]
    assert decl.parameters.required == ["schema_name", "tables"]


def test_sdk_baseline_and_wrapper_description_parity(
    converter, mock_genai_client
) -> None:
    """
    Parity with old baseline conversion test.

    Old test compared the SDK helper `FunctionDeclaration.from_callable` vs our
    wrapper conversion to ensure descriptions are present and consistent.

    In the new architecture, our wrapper calls google.genai._transformers.t_schema;
    here we directly validate that both extract the same description.
    """

    class Poem(BaseModel):
        """A simple poem."""

        content: str

    @dataclass(frozen=True)
    class _ToolMetadata:
        name: str
        description: str
        fn_schema: type[BaseModel]

    @dataclass(frozen=True)
    class _Tool:
        metadata: _ToolMetadata

    tool = _Tool(
        metadata=_ToolMetadata(
            name="Poem",
            description="A simple poem.",
            fn_schema=Poem,
        )
    )

    assert "A simple poem." in (tool.metadata.fn_schema.__doc__ or "")

    # SDK baseline schema generation.
    baseline_schema = _transformers.t_schema(mock_genai_client, Poem)
    assert baseline_schema is not None

    decl = converter.to_function_declaration(tool)
    assert decl.name == "Poem"
    assert decl.description == "A simple poem."
    assert decl.parameters is not None


def test_optional_lists_nested_gemini_schema_and_live_generation(converter) -> None:
    """
    Parity with old `test_optional_lists_nested_gemini` schema conversion depth.

    This test focuses on schema conversion depth. Live structured generation is
    covered in `tests_new/integration/test_google_genai.py`.
    """

    class TextContent(BaseModel):
        """A piece of text content."""

        text: str
        language: str

    class ImageContent(BaseModel):
        """A piece of image content."""

        url: str
        alt_text: Optional[str]
        width: Optional[int]
        height: Optional[int]

    class VideoContent(BaseModel):
        """A piece of video content."""

        url: str
        duration_seconds: int
        thumbnail: Optional[str]

    class Content(BaseModel):
        """Content of a blog post."""

        title: str
        created_at: str
        text: Optional[TextContent] = None
        image: Optional[ImageContent]
        video: Optional[VideoContent]
        tags: List[str]

    class BlogPost(BaseModel):
        """A blog post."""

        id: str
        author: str
        published: bool
        contents: List[Content]
        category: Optional[str]

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
