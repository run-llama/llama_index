import os
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types
from llama_index.core.base.llms.types import (
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
)
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from google.genai.types import GenerateContentConfig, ThinkingConfig
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.google_genai.utils import (
    convert_schema_to_function_declaration,
    prepare_chat_params,
)


SKIP_GEMINI = (
    os.environ.get("GOOGLE_API_KEY") is None
    or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false") == "true"
)


class Poem(BaseModel):
    """A simple poem."""

    content: str


class Column(BaseModel):
    """A model of a column in a table."""

    name: str = Field(description="Column field")
    data_type: str = Field(description="Data type field")


class Table(BaseModel):
    """A model of a table in a database."""

    name: str = Field(description="Table name field")
    columns: List[Column] = Field(description="List of random Column objects")


class Schema(BaseModel):
    """A model of a schema in a database."""

    schema_name: str = Field(description="Schema name")
    tables: List[Table] = Field(description="List of random Table objects")


# Define the models to test against
GEMINI_MODELS_TO_TEST = (
    [
        {"model": "models/gemini-2.0-flash-001", "config": {}},
        {
            "model": "models/gemini-2.5-flash-preview-04-17",
            "config": {
                "generation_config": GenerateContentConfig(
                    thinking_config=ThinkingConfig(thinking_budget=0)
                )
            },
        },
    ]
    if not SKIP_GEMINI
    else []
)


@pytest.fixture(params=GEMINI_MODELS_TO_TEST)
def llm(request) -> GoogleGenAI:
    """Fixture to create a GoogleGenAI instance for each model."""
    return GoogleGenAI(
        model=request.param["model"],
        api_key=os.environ["GOOGLE_API_KEY"],
        **request.param.get("config", {}),
    )


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_complete(llm: GoogleGenAI) -> None:
    """Test both sync and async complete methods."""
    prompt = "Write a poem about a magic backpack"

    # Test synchronous complete
    sync_response = llm.complete(prompt)
    assert sync_response is not None
    assert len(sync_response.text) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_acomplete(llm: GoogleGenAI) -> None:
    """Test both sync and async complete methods."""
    prompt = "Write a poem about a magic backpack"

    # Test async complete
    async_response = await llm.acomplete(prompt)
    assert async_response is not None
    assert len(async_response.text) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_chat(llm: GoogleGenAI) -> None:
    """Test both sync and async chat methods."""
    message = ChatMessage(content="Write a poem about a magic backpack")

    # Test synchronous chat
    sync_response = llm.chat(messages=[message])
    assert sync_response is not None
    assert sync_response.message.content and len(sync_response.message.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_achat(llm: GoogleGenAI) -> None:
    """Test both sync and async chat methods."""
    message = ChatMessage(content="Write a poem about a magic backpack")

    # Test async chat
    async_response = await llm.achat(messages=[message])
    assert async_response is not None
    assert async_response.message.content and len(async_response.message.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_stream_chat(llm: GoogleGenAI) -> None:
    """Test both sync and async stream chat methods."""
    message = ChatMessage(content="Write a poem about a magic backpack")

    # Test synchronous stream chat
    sync_chunks = list(llm.stream_chat(messages=[message]))
    assert len(sync_chunks) > 0
    assert all(isinstance(chunk.message.content, str) for chunk in sync_chunks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_astream_chat(llm: GoogleGenAI) -> None:
    """Test both sync and async stream chat methods."""
    message = ChatMessage(content="Write a poem about a magic backpack")

    # Test async stream chat
    response_gen = await llm.astream_chat(messages=[message])
    chunks = []
    async for chunk in response_gen:
        chunks.append(chunk)
    assert len(chunks) > 0
    assert all(isinstance(chunk.message.content, str) for chunk in chunks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_stream_complete(llm: GoogleGenAI) -> None:
    """Test both sync and async stream complete methods."""
    prompt = "Write a poem about a magic backpack"

    # Test synchronous stream complete
    sync_chunks = list(llm.stream_complete(prompt))
    assert len(sync_chunks) > 0
    assert all(isinstance(chunk.text, str) for chunk in sync_chunks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_astream_complete(llm: GoogleGenAI) -> None:
    """Test both sync and async stream complete methods."""
    prompt = "Write a poem about a magic backpack"

    # Test async stream complete
    response_gen = await llm.astream_complete(prompt)
    chunks = []
    async for chunk in response_gen:
        chunks.append(chunk)
    assert len(chunks) > 0
    assert all(isinstance(chunk.text, str) for chunk in chunks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_astructured_predict(llm: GoogleGenAI) -> None:
    """Test async structured prediction with a simple schema."""
    response = await llm.astructured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    assert response is not None
    assert isinstance(response, Poem)
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_simple_stream_structured_predict(llm: GoogleGenAI) -> None:
    """Test stream structured prediction with a simple schema."""
    response = llm.stream_structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    result = None
    for partial_response in response:
        assert hasattr(partial_response, "content")
        result = partial_response

    assert result is not None
    assert isinstance(result, Poem)
    assert len(result.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_simple_astream_structured_predict(llm: GoogleGenAI) -> None:
    """Test async stream structured prediction with a simple schema."""
    response = await llm.astream_structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    result = None
    async for partial_response in response:
        result = partial_response
        assert hasattr(result, "content")

    assert result is not None
    assert isinstance(result, Poem)
    assert isinstance(result.content, str)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_simple_structured_predict(llm: GoogleGenAI) -> None:
    """Test structured prediction with a simple schema."""
    response = llm.structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    assert response is not None
    assert isinstance(response, Poem)
    assert isinstance(response.content, str)
    assert len(response.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_complex_structured_predict(llm: GoogleGenAI) -> None:
    """Test structured prediction with a complex nested schema."""
    prompt = PromptTemplate("Generate a simple database structure")
    response = llm.structured_predict(output_cls=Schema, prompt=prompt)

    assert response is not None
    assert isinstance(response, Schema)
    assert isinstance(response.schema_name, str)
    assert len(response.schema_name) > 0
    assert len(response.tables) > 0
    assert all(isinstance(table, Table) for table in response.tables)
    assert all(len(table.columns) > 0 for table in response.tables)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_anyof_optional_structured_predict(llm: GoogleGenAI) -> None:
    class Person(BaseModel):
        last_name: str = Field(description="Last name")
        first_name: Optional[str] = Field(None, description="Optional first name")

    prompt = PromptTemplate("Create a fake person ")
    response = llm.structured_predict(output_cls=Person, prompt=prompt)

    assert response is not None
    assert isinstance(response, Person)
    assert isinstance(response.last_name, str)
    assert isinstance(response.first_name, None | str)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_as_structured_llm_native_genai(llm: GoogleGenAI) -> None:
    schema_response = llm._client.models.generate_content(
        model=llm.model,
        contents="Generate a simple database structure with at least one table called 'experiments'",
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Schema,
        ),
    ).parsed
    assert isinstance(schema_response, Schema)
    assert len(schema_response.schema_name) > 0
    assert len(schema_response.tables) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_as_structured_llm(llm: GoogleGenAI) -> None:
    prompt = PromptTemplate("Generate content")

    # Test with simple schema
    poem_response = llm.as_structured_llm(output_cls=Poem, prompt=prompt).complete(
        "Write a poem about a magic backpack"
    )
    assert isinstance(poem_response.raw, Poem)
    assert len(poem_response.raw.content) > 0

    # Test with complex schema
    schema_response = llm.as_structured_llm(output_cls=Schema, prompt=prompt).complete(
        "Generate a simple database structure with at least one table called 'experiments'"
    )

    assert isinstance(schema_response.raw, Schema)
    assert len(schema_response.raw.schema_name) > 0
    assert len(schema_response.raw.tables) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_as_structured_llm_async(llm: GoogleGenAI) -> None:
    prompt = PromptTemplate("Generate content")

    # Test with simple schema
    poem_response = await llm.as_structured_llm(
        output_cls=Poem, prompt=prompt
    ).acomplete("Write a poem about a magic backpack")
    assert isinstance(poem_response.raw, Poem)
    assert len(poem_response.raw.content) > 0

    # Test with complex schema
    schema_response = await llm.as_structured_llm(
        output_cls=Schema, prompt=prompt
    ).acomplete("Generate a simple database structure")
    assert isinstance(schema_response.raw, Schema)
    assert len(schema_response.raw.schema_name) > 0
    assert len(schema_response.raw.tables) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_as_structure_llm_with_config(llm: GoogleGenAI) -> None:
    response = (
        llm.as_structured_llm(output_cls=Poem)
        .complete(
            prompt="Write a poem about a magic backpack",
            # here we want to change the temperature, but it must not override the whole config
            config={"temperature": 0.1},
        )
        .raw
    )

    assert isinstance(response, Poem)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_as_structured_llm_async_with_config(llm: GoogleGenAI) -> None:
    response = await llm.as_structured_llm(output_cls=Poem).acomplete(
        prompt="Write a poem about a magic backpack",
        # here we want to change the temperature, but it must not override the whole config
        config={"temperature": 0.1},
    )

    assert isinstance(response.raw, Poem)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_structured_predict_multiple_block(llm: GoogleGenAI) -> None:
    chat_messages = [
        ChatMessage(
            content=[
                TextBlock(text="which logo is this?"),
                ImageBlock(
                    url="https://upload.wikimedia.org/wikipedia/commons/7/7a/Nohat-wiki-logo.png"
                ),
            ],
            role=MessageRole.USER,
        ),
    ]

    class Response(BaseModel):
        answer: str

    support = llm.structured_predict(
        output_cls=Response, prompt=ChatPromptTemplate(message_templates=chat_messages)
    )
    assert isinstance(support, Response)
    assert "wiki" in support.answer.lower()


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_get_tool_calls_from_response(llm: GoogleGenAI) -> None:
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer."""
        return a + b

    add_tool = FunctionTool.from_defaults(fn=add)
    msg = ChatMessage("What is the result of adding 2 and 3?")
    response = llm.chat_with_tools(
        user_msg=msg,
        tools=[add_tool],
    )

    tool_calls: List[ToolSelection] = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"
    assert tool_calls[0].tool_kwargs == {"a": 2, "b": 3}


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


@pytest.fixture
def mock_google_genai() -> GoogleGenAI:
    """Fixture to create a mocked GoogleGenAI instance for unit testing."""
    with patch("google.genai.Client") as mock_client_class:
        # Mock the client and its methods
        mock_client = MagicMock()
        mock_client.models.get.return_value = MagicMock(
            input_token_limit=200000, output_token_limit=8192
        )
        mock_client_class.return_value = mock_client

        return GoogleGenAI(model="models/gemini-2.0-flash-001", api_key="test-key")


def test_prepare_chat_with_tools_tool_required(mock_google_genai: GoogleGenAI) -> None:
    """Test that tool_required is correctly passed to the API request when True."""
    # Test with tool_required=True
    result = mock_google_genai._prepare_chat_with_tools(
        tools=[search_tool], tool_required=True
    )

    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.ANY
    )
    assert len(result["tools"]) == 1
    assert result["tools"][0].function_declarations[0].name == "search_tool"


def test_prepare_chat_with_tools_tool_not_required(
    mock_google_genai: GoogleGenAI,
) -> None:
    """Test that tool_required is correctly passed to the API request when False."""
    # Test with tool_required=False (default)
    result = mock_google_genai._prepare_chat_with_tools(
        tools=[search_tool], tool_required=False
    )

    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.AUTO
    )
    assert len(result["tools"]) == 1
    assert result["tools"][0].function_declarations[0].name == "search_tool"


def test_prepare_chat_with_tools_default_behavior(
    mock_google_genai: GoogleGenAI,
) -> None:
    """Test that tool_required defaults to False."""
    # Test default behavior (should be equivalent to tool_required=False)
    result = mock_google_genai._prepare_chat_with_tools(tools=[search_tool])

    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.AUTO
    )
    assert len(result["tools"]) == 1
    assert result["tools"][0].function_declarations[0].name == "search_tool"


def test_prepare_chat_with_tools_explicit_tool_choice_overrides_tool_required(
    mock_google_genai: GoogleGenAI,
) -> None:
    """Test that explicit tool_choice overrides tool_required parameter."""
    # Test with tool_required=True but explicit tool_choice="auto"
    result = mock_google_genai._prepare_chat_with_tools(
        tools=[search_tool], tool_required=True, tool_choice="auto"
    )

    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.AUTO
    )
    assert len(result["tools"]) == 1
    assert result["tools"][0].function_declarations[0].name == "search_tool"

    # Test with tool_required=False but explicit tool_choice="any"
    result = mock_google_genai._prepare_chat_with_tools(
        tools=[search_tool], tool_required=False, tool_choice="any"
    )

    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.ANY
    )
    assert len(result["tools"]) == 1
    assert result["tools"][0].function_declarations[0].name == "search_tool"


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_tool_required_integration(llm: GoogleGenAI) -> None:
    """Test tool_required parameter in actual chat_with_tools calls."""
    # Test with tool_required=True
    response = llm.chat_with_tools(
        user_msg="What is the weather in Paris?",
        tools=[search_tool],
        tool_required=True,
    )
    assert response.message.additional_kwargs.get("tool_calls") is not None
    assert len(response.message.additional_kwargs["tool_calls"]) > 0

    # Test with tool_required=False
    response = llm.chat_with_tools(
        user_msg="Say hello!",
        tools=[search_tool],
        tool_required=False,
    )
    # Should not use tools for a simple greeting when tool_required=False
    # Note: This might still use tools depending on the model's behavior,
    # but the important thing is that the tool_config is set correctly
    assert response is not None


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_convert_llama_index_schema_to_gemini_function_declaration(
    llm: GoogleGenAI,
) -> None:
    """Test conversion of a llama_index schema to a gemini function declaration."""
    function_tool = get_function_tool(Poem)
    # this is our baseline, which is not working because:
    # 1. the descriptions are missing
    google_openai_function = types.FunctionDeclaration.from_callable(
        client=llm._client,
        callable=function_tool.metadata.fn_schema,  # type: ignore
    )

    assert google_openai_function.description == "A simple poem."

    # this is our custom conversion that can take a llama index: fn_schema and convert it to a gemini compatible
    # function declaration (subset of OpenAPI v3)
    converted = convert_schema_to_function_declaration(llm._client, function_tool)

    assert converted.name == "Poem"
    assert converted.description == "A simple poem."
    assert converted.parameters.required is not None

    assert converted.parameters


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_convert_llama_index_schema_to_gemini_function_declaration_nested_case(
    llm: GoogleGenAI,
) -> None:
    """Test conversion of a llama_index fn_schema to a gemini function declaration."""
    function_tool = get_function_tool(Schema)

    llama_index_model_json_schema = function_tool.metadata.fn_schema.model_json_schema()
    # check that the model_json_schema contains a $defs key, which is not supported by Gemini
    assert "$defs" in llama_index_model_json_schema

    converted = convert_schema_to_function_declaration(llm._client, function_tool)

    assert converted.name == "Schema"
    assert converted.description is not None
    assert converted.parameters.required is not None

    assert converted.parameters
    assert list(converted.parameters.properties) == [
        "schema_name",
        "tables",
    ]

    assert converted.parameters.required == ["schema_name", "tables"]


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_optional_value_gemini(llm: GoogleGenAI) -> None:
    class OptionalContent(BaseModel):
        content: Optional[str] = Field(default=None)
        content2: str | None

    function_tool = get_function_tool(OptionalContent)
    decl = convert_schema_to_function_declaration(llm._client, function_tool)

    assert decl.parameters.properties["content"].nullable
    assert decl.parameters.properties["content"].default is None

    assert decl.parameters.properties["content2"].nullable
    assert decl.parameters.properties["content2"].default is None


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_optional_lists_nested_gemini(llm: GoogleGenAI) -> None:
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

    function_tool = get_function_tool(BlogPost)

    llama_index_model_json_schema = function_tool.metadata.fn_schema.model_json_schema()
    assert "$defs" in llama_index_model_json_schema

    converted = convert_schema_to_function_declaration(llm._client, function_tool)

    assert converted.name == "BlogPost"

    contents_property = converted.parameters.properties["contents"]
    assert contents_property.type == types.Type.ARRAY

    content_items = contents_property.items
    assert "text" in content_items.properties
    assert "image" in content_items.properties
    assert "video" in content_items.properties

    blogpost = (
        llm.as_structured_llm(output_cls=BlogPost)
        .complete(prompt="Write a blog post with at least 3 contents")
        .raw
    )
    assert isinstance(blogpost, BlogPost)
    assert len(blogpost.contents) >= 3


def test_prepare_chat_params_more_than_2_tool_calls():
    expected_generation_config = types.GenerateContentConfig()
    expected_model_name = "models/gemini-foo"
    test_messages = [
        ChatMessage(content="Find me a puppy.", role=MessageRole.USER),
        ChatMessage(
            content="Let me search for puppies.",
            role=MessageRole.ASSISTANT,
            additional_kwargs={
                "tool_calls": [
                    {"name": "tool_1"},
                    {"name": "tool_2"},
                    {"name": "tool_3"},
                ]
            },
        ),
        ChatMessage(
            content="Tool 1 Response",
            role=MessageRole.TOOL,
            additional_kwargs={"tool_call_id": "tool_1"},
        ),
        ChatMessage(
            content="Tool 2 Response",
            role=MessageRole.TOOL,
            additional_kwargs={"tool_call_id": "tool_2"},
        ),
        ChatMessage(
            content="Tool 3 Response",
            role=MessageRole.TOOL,
            additional_kwargs={"tool_call_id": "tool_3"},
        ),
        ChatMessage(content="Here is a list of puppies.", role=MessageRole.ASSISTANT),
    ]

    next_msg, chat_kwargs = prepare_chat_params(expected_model_name, test_messages)

    assert chat_kwargs["model"] == expected_model_name
    assert chat_kwargs["config"] == expected_generation_config
    assert next_msg == types.Content(
        parts=[types.Part(text="Here is a list of puppies.")], role=MessageRole.MODEL
    )
    assert chat_kwargs["history"] == [
        types.Content(
            parts=[types.Part(text="Find me a puppy.")], role=MessageRole.USER
        ),
        types.Content(
            parts=[
                types.Part(text="Let me search for puppies."),
                types.Part.from_function_call(name="tool_1", args=None),
                types.Part.from_function_call(name="tool_2", args=None),
                types.Part.from_function_call(name="tool_3", args=None),
            ],
            role=MessageRole.MODEL,
        ),
        types.Content(
            parts=[
                types.Part.from_function_response(
                    name="tool_1", response={"result": "Tool 1 Response"}
                ),
                types.Part.from_function_response(
                    name="tool_2", response={"result": "Tool 2 Response"}
                ),
                types.Part.from_function_response(
                    name="tool_3", response={"result": "Tool 3 Response"}
                ),
            ],
            role=MessageRole.USER,
        ),
    ]
