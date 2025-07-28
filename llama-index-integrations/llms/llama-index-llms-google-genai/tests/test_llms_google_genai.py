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
    chat_from_gemini_response,
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
        {"model": "models/gemini-2.5-flash-lite", "config": {}},
        {
            "model": "models/gemini-2.5-flash",
            "config": {
                "generation_config": GenerateContentConfig(
                    thinking_config=ThinkingConfig(thinking_budget=512)
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


def test_prepare_chat_params_with_system_message():
    # Setup a conversation starting with a SYSTEM message
    model_name = "models/gemini-test"
    system_prompt = "You are a test system."
    user_message_1 = "Hello from user 1."
    assistant_message_1 = "Hello from assistant 1."
    user_message_2 = "Hello from user 2."
    messages = [
        ChatMessage(content=system_prompt, role=MessageRole.SYSTEM),
        ChatMessage(content=user_message_1, role=MessageRole.USER),
        ChatMessage(content=assistant_message_1, role=MessageRole.ASSISTANT),
        ChatMessage(content=user_message_2, role=MessageRole.USER),
    ]

    # Execute prepare_chat_params
    next_msg, chat_kwargs = prepare_chat_params(model_name, messages)

    # Verify system_prompt is forwarded to system_instruction
    cfg = chat_kwargs["config"]
    assert isinstance(cfg, GenerateContentConfig)
    assert cfg.system_instruction == system_prompt

    # Verify history only contains the user messages and the assistant message
    assert chat_kwargs["history"] == [
        types.Content(
            parts=[types.Part(text=user_message_1)],
            role=MessageRole.USER,
        ),
        types.Content(
            parts=[types.Part(text=assistant_message_1)],
            role=MessageRole.MODEL,
        ),
    ]

    # Verify next_msg is the user message
    assert next_msg == types.Content(
        parts=[types.Part(text=user_message_2)],
        role=MessageRole.USER,
    )


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_cached_content_initialization() -> None:
    """Test GoogleGenAI initialization with cached_content parameter."""
    cached_content_value = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-123"

    llm = GoogleGenAI(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
        cached_content=cached_content_value,
    )

    # Verify cached_content is stored in the instance
    assert llm.cached_content == cached_content_value

    # Verify cached_content is stored in generation config
    assert llm._generation_config["cached_content"] == cached_content_value


def test_cached_content_in_response() -> None:
    """Test that cached_content is extracted from Gemini responses."""
    # Mock response with cached_content
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = types.FinishReason.STOP
    mock_response.candidates[0].content.role = "model"
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Test response"
    mock_response.candidates[0].content.parts[0].inline_data = None
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    mock_response.function_calls = None
    mock_response.cached_content = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-123"

    # Convert response
    chat_response = chat_from_gemini_response(mock_response)

    # Verify cached_content is in raw response
    assert "cached_content" in chat_response.raw
    assert (
        chat_response.raw["cached_content"]
        == "projects/test-project/locations/us-central1/cachedContents/cached-content-id-123"
    )


def test_cached_content_without_cached_content() -> None:
    """Test response processing when cached_content is not present."""
    # Mock response without cached_content
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = types.FinishReason.STOP
    mock_response.candidates[0].content.role = "model"
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "Test response"
    mock_response.candidates[0].content.parts[0].inline_data = None
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    mock_response.function_calls = None
    # No cached_content attribute
    del mock_response.cached_content

    # Convert response
    chat_response = chat_from_gemini_response(mock_response)

    # Verify no cached_content key in raw response
    assert "cached_content" not in chat_response.raw


def test_thoughts_in_response() -> None:
    """Test response processing when thought summaries are present."""
    # Mock response without cached_content
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = types.FinishReason.STOP
    mock_response.candidates[0].content.role = "model"
    mock_response.candidates[0].content.parts = [MagicMock(), MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "This is a thought."
    mock_response.candidates[0].content.parts[0].inline_data = None
    mock_response.candidates[0].content.parts[0].thought = True
    mock_response.candidates[0].content.parts[1].text = "This is not a thought."
    mock_response.candidates[0].content.parts[1].inline_data = None
    mock_response.candidates[0].content.parts[1].thought = None
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    mock_response.function_calls = None
    # No cached_content attribute
    del mock_response.cached_content

    # Convert response
    chat_response = chat_from_gemini_response(mock_response)

    # Verify thoughts in raw response
    assert "thoughts" in chat_response.message.additional_kwargs
    assert chat_response.message.additional_kwargs["thoughts"] == "This is a thought."
    assert chat_response.message.content == "This is not a thought."


def test_thoughts_without_thought_response() -> None:
    """Test response processing when thought summaries are not present."""
    # Mock response without cached_content
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = types.FinishReason.STOP
    mock_response.candidates[0].content.role = "model"
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = "This is not a thought."
    mock_response.candidates[0].content.parts[0].inline_data = None
    mock_response.candidates[0].content.parts[0].thought = None
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    mock_response.function_calls = None
    # No cached_content attribute
    del mock_response.cached_content

    # Convert response
    chat_response = chat_from_gemini_response(mock_response)

    # Verify no cached_content key in raw response
    assert "thoughts" not in chat_response.message.additional_kwargs
    assert chat_response.message.content == "This is not a thought."


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_cached_content_with_generation_config() -> None:
    """Test that cached_content works with custom generation_config."""
    cached_content_value = "projects/test-project/locations/us-central1/cachedContents/cached-content-id-456"

    llm = GoogleGenAI(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
        generation_config=GenerateContentConfig(
            temperature=0.5,
            cached_content=cached_content_value,
        ),
    )

    # Verify both cached_content and custom config are preserved
    assert llm._generation_config["cached_content"] == cached_content_value
    assert llm._generation_config["temperature"] == 0.5


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_cached_content_in_chat_params() -> None:
    """Test that cached_content is properly included in generation config."""
    cached_content_value = (
        "projects/test-project/locations/us-central1/cachedContents/test-cache"
    )

    llm = GoogleGenAI(
        model="models/gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
        cached_content=cached_content_value,
    )

    # Verify cached_content is in the generation config
    assert llm._generation_config["cached_content"] == cached_content_value

    # Test that prepare_chat_params preserves cached_content
    messages = [ChatMessage(content="Test message", role=MessageRole.USER)]

    # Prepare chat params with the LLM's generation config
    next_msg, chat_kwargs = prepare_chat_params(
        llm.model, messages, generation_config=llm._generation_config
    )

    # Verify cached_content is preserved in the config
    assert chat_kwargs["config"].cached_content == cached_content_value


def test_built_in_tool_initialization() -> None:
    """Test GoogleGenAI initialization with built_in_tool parameter."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    # Mock the client
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the model metadata response
        mock_model = MagicMock()
        mock_model.supported_generation_methods = ["generateContent"]
        mock_client.models.get.return_value = mock_model

        llm = GoogleGenAI(
            model="gemini-2.0-flash-001",
            built_in_tool=grounding_tool,
        )

        # Verify built_in_tool is stored in the instance
        assert llm.built_in_tool == grounding_tool

        # Verify built_in_tool is included in generation config tools
        assert "tools" in llm._generation_config
        assert len(llm._generation_config["tools"]) == 1

        # The tool gets converted to dict format in generation config
        tool_dict = llm._generation_config["tools"][0]
        assert isinstance(tool_dict, dict)
        assert "google_search" in tool_dict


def test_built_in_tool_in_response() -> None:
    """Test that built_in_tool information is extracted from Gemini responses."""
    # Mock response with built_in_tool usage metadata
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].finish_reason = types.FinishReason.STOP
    mock_response.candidates[0].content.role = "model"
    mock_response.candidates[0].content.parts = [MagicMock()]
    mock_response.candidates[0].content.parts[
        0
    ].text = "Test response with search results"
    mock_response.candidates[0].content.parts[0].inline_data = None
    mock_response.candidates[0].content.parts[0].thought = None
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.model_dump.return_value = {
        "prompt_token_count": 10,
        "candidates_token_count": 20,
        "total_token_count": 30,
    }
    mock_response.function_calls = None

    # Mock grounding metadata
    grounding_metadata = {
        "web_search_queries": ["test query"],
        "search_entry_point": {"rendered_content": "search results"},
        "grounding_supports": [
            {
                "segment": {"start_index": 0, "end_index": 10, "text": "Test"},
                "grounding_chunk_indices": [0],
            }
        ],
        "grounding_chunks": [
            {"web": {"uri": "https://example.com", "title": "Example"}}
        ],
    }
    mock_response.candidates[0].grounding_metadata = grounding_metadata

    # Mock model_dump to include grounding_metadata
    mock_response.candidates[0].model_dump.return_value = {
        "finish_reason": types.FinishReason.STOP,
        "content": {
            "role": "model",
            "parts": [{"text": "Test response with search results"}],
        },
        "grounding_metadata": grounding_metadata,
    }

    # Convert response
    chat_response = chat_from_gemini_response(mock_response)

    # Verify response is processed correctly
    assert chat_response.message.role == MessageRole.ASSISTANT
    assert len(chat_response.message.blocks) == 1
    assert chat_response.message.blocks[0].text == "Test response with search results"

    # Verify grounding metadata is in raw response
    assert "grounding_metadata" in chat_response.raw
    assert chat_response.raw["grounding_metadata"]["web_search_queries"] == [
        "test query"
    ]


def test_built_in_tool_with_generation_config() -> None:
    """Test that built_in_tool works with custom generation_config."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    # Mock the client
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the model metadata response
        mock_model = MagicMock()
        mock_model.supported_generation_methods = ["generateContent"]
        mock_client.models.get.return_value = mock_model

        llm = GoogleGenAI(
            model="gemini-2.0-flash-001",
            built_in_tool=grounding_tool,
            generation_config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=1000,
            ),
        )

        # Verify built_in_tool is stored in the instance even with custom generation config
        assert llm.built_in_tool == grounding_tool

        # Verify custom config parameters are preserved
        assert llm._generation_config["temperature"] == 0.5
        assert llm._generation_config["max_output_tokens"] == 1000

        # Verify built_in_tool is now properly added to the generation config
        assert "tools" in llm._generation_config
        assert len(llm._generation_config["tools"]) == 1

        # The tool should be preserved as the original Tool object
        tool_obj = llm._generation_config["tools"][0]
        assert isinstance(tool_obj, types.Tool)
        assert tool_obj == grounding_tool


def test_built_in_tool_in_chat_params() -> None:
    """Test that built_in_tool is properly included in chat parameters."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    messages = [
        ChatMessage(role=MessageRole.USER, content="What is the weather today?")
    ]

    # Mock the client
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the model metadata response
        mock_model = MagicMock()
        mock_model.supported_generation_methods = ["generateContent"]
        mock_client.models.get.return_value = mock_model

        llm = GoogleGenAI(
            model="gemini-2.0-flash-001",
            built_in_tool=grounding_tool,
        )

        # Prepare chat params
        next_msg, chat_kwargs = prepare_chat_params(
            llm.model, messages, generation_config=llm._generation_config
        )

        # Verify built_in_tool is in the chat config
        assert hasattr(chat_kwargs["config"], "tools")
        assert chat_kwargs["config"].tools is not None
        assert len(chat_kwargs["config"].tools) == 1

        # The tool should be preserved as the original Tool object in chat config
        assert chat_kwargs["config"].tools[0] == grounding_tool


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_with_invalid_tool() -> None:
    """Test error handling when built_in_tool is invalid or malformed."""
    # Mock the client
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the model metadata response
        mock_model = MagicMock()
        mock_model.supported_generation_methods = ["generateContent"]
        mock_client.models.get.return_value = mock_model

        # Test with None as built_in_tool
        llm = GoogleGenAI(
            model="gemini-2.0-flash-001",
            built_in_tool=None,
        )

        # Should initialize successfully without tools
        assert llm.built_in_tool is None
        assert "tools" in llm._generation_config and not llm._generation_config.get(
            "tools"
        )


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_with_streaming() -> None:
    """Test that built_in_tool works correctly with streaming responses."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
        built_in_tool=grounding_tool,
    )

    # Test streaming chat
    messages = [ChatMessage(content="Who won the Euro 2024?", role=MessageRole.USER)]

    stream_response = llm.stream_chat(messages)

    # Collect all streaming chunks
    chunks = []
    final_response = None
    for chunk in stream_response:
        chunks.append(chunk)
        final_response = chunk

    assert len(chunks) > 0
    assert final_response is not None
    assert final_response.message is not None
    assert len(final_response.message.content) > 0

    # Check if grounding metadata is present in the final response
    if hasattr(final_response, "raw") and final_response.raw:
        raw_response = final_response.raw
        # Grounding metadata may be present depending on whether search was used
        if "grounding_metadata" in raw_response:
            assert isinstance(raw_response["grounding_metadata"], dict)


def test_built_in_tool_config_merge_edge_cases() -> None:
    """Test edge cases in merging built_in_tool with generation_config."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    # Test with generation_config that already has empty tools list
    empty_tools_config = types.GenerateContentConfig(temperature=0.7, tools=[])

    # Mock the client
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the model metadata response
        mock_model = MagicMock()
        mock_model.supported_generation_methods = ["generateContent"]
        mock_client.models.get.return_value = mock_model

        llm = GoogleGenAI(
            model="gemini-2.0-flash-001",
            built_in_tool=grounding_tool,
            generation_config=empty_tools_config,
        )

        # Tool should be added to the empty tools list
        assert "tools" in llm._generation_config
        assert len(llm._generation_config["tools"]) == 1
        assert llm._generation_config["temperature"] == 0.7

        # Test with generation_config that has existing tools
        existing_tool = types.Tool(google_search=types.GoogleSearch())
        existing_tools_config = types.GenerateContentConfig(
            temperature=0.3, tools=[existing_tool]
        )

        # Should raise an error when trying to add another built_in_tool
        with pytest.raises(
            ValueError,
            match="Providing multiple Google GenAI tools or mixing with custom tools is not supported.",
        ):
            GoogleGenAI(
                model="gemini-2.0-flash-001",
                built_in_tool=grounding_tool,
                generation_config=existing_tools_config,
            )


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_error_recovery() -> None:
    """Test error recovery when built_in_tool encounters issues."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
        built_in_tool=grounding_tool,
    )

    # Test with a query that might not trigger search (should still work)
    response = llm.complete("Hello, how are you?")

    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0

    # The LLM should still function even if the tool isn't used
    assert isinstance(response.raw, dict)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_built_in_tool_async_compatibility() -> None:
    """Test that built_in_tool works with async methods."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
        built_in_tool=grounding_tool,
    )

    # Test async complete
    response = await llm.acomplete("What is machine learning?")

    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0

    # Test async chat
    messages = [ChatMessage(content="Explain quantum computing", role=MessageRole.USER)]
    chat_response = await llm.achat(messages)

    assert chat_response is not None
    assert chat_response.message is not None
    assert len(chat_response.message.content) > 0

    # Verify tool configuration persists in async calls
    assert llm.built_in_tool == grounding_tool


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_google_search() -> None:
    """Test Google Search functionality with built_in_tool."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
        built_in_tool=grounding_tool,
    )

    response = llm.complete("What is the current weather in San Francisco?")

    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0

    # Check if grounding metadata is present in the response
    assert "raw" in response.__dict__
    raw_response = response.raw
    assert isinstance(raw_response, dict)

    # Grounding metadata should be present when Google Search is used
    # Note: This may not always be present depending on whether the model
    # decides to use the search tool, so we check if it exists
    if "grounding_metadata" in raw_response:
        grounding_metadata = raw_response["grounding_metadata"]
        assert isinstance(grounding_metadata, dict)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_google_search_grounding_metadata(llm: GoogleGenAI) -> None:
    """Test that Google Search returns comprehensive grounding metadata in response."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    # Create a new LLM instance with the grounding tool
    llm_with_search = GoogleGenAI(
        model=llm.model,
        api_key=os.environ["GOOGLE_API_KEY"],
        built_in_tool=grounding_tool,
    )

    response = llm_with_search.complete("What is the capital of Japan?")

    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0

    raw_response = response.raw
    assert raw_response is not None
    assert isinstance(raw_response, dict)

    # Grounding metadata should be present when Google Search is used
    # Note: Grounding metadata may not always be present depending on
    # whether the model decides to use the search tool
    if "grounding_metadata" in raw_response:
        grounding_metadata = raw_response["grounding_metadata"]
        assert isinstance(grounding_metadata, dict)

        # Web search queries should be present if grounding was used
        if "web_search_queries" in grounding_metadata:
            assert grounding_metadata["web_search_queries"] is not None
            assert isinstance(grounding_metadata["web_search_queries"], list)
            assert len(grounding_metadata["web_search_queries"]) > 0

            # Validate each web search query
            for query in grounding_metadata["web_search_queries"]:
                assert isinstance(query, str)
                assert len(query.strip()) > 0

        # Search entry point should be present if grounding was used
        if "search_entry_point" in grounding_metadata:
            search_entry_point = grounding_metadata["search_entry_point"]
            assert isinstance(search_entry_point, dict)

            # Rendered content should be present
            if "rendered_content" in search_entry_point:
                assert search_entry_point["rendered_content"] is not None
                assert isinstance(search_entry_point["rendered_content"], str)
                assert len(search_entry_point["rendered_content"].strip()) > 0

        # Grounding supports should be present if grounding was used
        if "grounding_supports" in grounding_metadata:
            assert grounding_metadata["grounding_supports"] is not None
            assert isinstance(grounding_metadata["grounding_supports"], list)

            # Validate grounding support structure if present
            for support in grounding_metadata["grounding_supports"]:
                assert isinstance(support, dict)
                if "segment" in support:
                    segment = support["segment"]
                    assert isinstance(segment, dict)

        # Grounding chunks should be present if grounding was used
        if "grounding_chunks" in grounding_metadata:
            assert grounding_metadata["grounding_chunks"] is not None
            assert isinstance(grounding_metadata["grounding_chunks"], list)

            # Validate grounding chunk structure if present
            for chunk in grounding_metadata["grounding_chunks"]:
                assert isinstance(chunk, dict)
                if "web" in chunk:
                    web_chunk = chunk["web"]
                    assert isinstance(web_chunk, dict)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_code_execution() -> None:
    """Test Code Execution functionality with built_in_tool."""
    code_execution_tool = types.Tool(code_execution=types.ToolCodeExecution())

    llm = GoogleGenAI(
        model="gemini-2.0-flash-001",
        api_key=os.environ["GOOGLE_API_KEY"],
        built_in_tool=code_execution_tool,
    )

    response = llm.complete("Calculate 20th fibonacci number.")

    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0

    # The response should contain the calculation result
    assert "6765" in response.text

    # Check if the raw response contains the expected structure
    assert "raw" in response.__dict__
    raw_response = response.raw
    assert isinstance(raw_response, dict)


def test_code_execution_response_parts() -> None:
    """Test that code execution response contains executable_code, code_execution_result, and text parts."""
    code_execution_tool = types.Tool(code_execution=types.ToolCodeExecution())

    # Mock response with code execution parts
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = types.FinishReason.STOP
    mock_candidate.content.role = "model"
    mock_response.candidates = [mock_candidate]

    # Create mock parts for text, executable code, and code execution result
    mock_text_part = MagicMock()
    mock_text_part.text = (
        "I'll calculate the sum of the first 50 prime numbers for you."
    )
    mock_text_part.inline_data = None
    mock_text_part.thought = None

    mock_code_part = MagicMock()
    mock_code_part.text = None
    mock_code_part.inline_data = None
    mock_code_part.thought = None
    mock_code_part.executable_code = {
        "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprimes = []\nn = 2\nwhile len(primes) < 50:\n    if is_prime(n):\n        primes.append(n)\n    n += 1\n\nprint(f'Sum of first 50 primes: {sum(primes)}')",
        "language": types.Language.PYTHON,
    }

    mock_result_part = MagicMock()
    mock_result_part.text = None
    mock_result_part.inline_data = None
    mock_result_part.thought = None
    mock_result_part.code_execution_result = {
        "outcome": types.Outcome.OUTCOME_OK,
        "output": "Sum of first 50 primes: 5117",
    }

    mock_final_text_part = MagicMock()
    mock_final_text_part.text = "The sum of the first 50 prime numbers is 5117."
    mock_final_text_part.inline_data = None
    mock_final_text_part.thought = None

    mock_candidate.content.parts = [
        mock_text_part,
        mock_code_part,
        mock_result_part,
        mock_final_text_part,
    ]
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None
    mock_response.function_calls = None

    # Mock model_dump to return the expected structure
    mock_candidate.model_dump.return_value = {
        "finish_reason": types.FinishReason.STOP,
        "content": {
            "role": "model",
            "parts": [
                {
                    "text": "I'll calculate the sum of the first 50 prime numbers for you."
                },
                {
                    "executable_code": {
                        "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprimes = []\nn = 2\nwhile len(primes) < 50:\n    if is_prime(n):\n        primes.append(n)\n    n += 1\n\nprint(f'Sum of first 50 primes: {sum(primes)}')",
                        "language": types.Language.PYTHON,
                    }
                },
                {
                    "code_execution_result": {
                        "outcome": types.Outcome.OUTCOME_OK,
                        "output": "Sum of first 50 primes: 5117",
                    }
                },
                {"text": "The sum of the first 50 prime numbers is 5117."},
            ],
        },
    }

    # Mock the client and chat method
    with patch("google.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock the model metadata response
        mock_model = MagicMock()
        mock_model.supported_generation_methods = ["generateContent"]
        mock_client.models.get.return_value = mock_model

        # Mock the chat creation and send_message method
        mock_chat = MagicMock()
        mock_client.chats.create.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response

        llm = GoogleGenAI(
            model="gemini-2.0-flash-001",
            built_in_tool=code_execution_tool,
        )

        messages = [
            ChatMessage(
                role="user", content="What is the sum of the first 50 prime numbers?"
            )
        ]
        response = llm.chat(messages)

        assert response is not None
        assert response.message is not None
        assert len(response.message.content) > 0

        # Check the raw response structure
        raw_response = response.raw
        assert isinstance(raw_response, dict)
        assert "content" in raw_response

        content = raw_response["content"]
        assert "parts" in content
        assert isinstance(content["parts"], list)
        assert len(content["parts"]) > 0

        # Analyze each part in the response
        for part in content["parts"]:
            assert isinstance(part, dict)

            # Check for text parts
            if part.get("text") is not None:
                assert isinstance(part["text"], str)
                assert len(part["text"].strip()) > 0

            # Check for executable code parts
            if part.get("executable_code") is not None:
                executable_code_content = part["executable_code"]

                # Validate executable code structure
                assert isinstance(executable_code_content, dict)
                assert "code" in executable_code_content
                assert "language" in executable_code_content

                # Validate the code content
                code = executable_code_content["code"]
                assert isinstance(code, str)
                assert len(code.strip()) > 0

                # Validate language
                assert executable_code_content["language"] == types.Language.PYTHON

            # Check for code execution result parts
            if part.get("code_execution_result") is not None:
                code_execution_result = part["code_execution_result"]

                # Validate code execution result structure
                assert isinstance(code_execution_result, dict)
                assert "outcome" in code_execution_result
                assert "output" in code_execution_result

                # Validate the execution outcome
                assert code_execution_result["outcome"] == types.Outcome.OUTCOME_OK

                # Validate the output
                output = code_execution_result["output"]
                assert isinstance(output, str)
                assert len(output.strip()) > 0

        # The response should mention the final answer
        response_content = response.message.content
        assert "5117" in response_content


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_thoughts_with_streaming() -> None:
    """Test that thought summaries work correctly with streaming responses."""
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
        generation_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(
                include_thoughts=True,
            ),
        ),
    )

    # Test streaming chat
    messages = [ChatMessage(content="What is your name?", role=MessageRole.USER)]

    stream_response = llm.stream_chat(messages)

    # Collect all streaming chunks
    chunks = []
    final_response = None
    for chunk in stream_response:
        chunks.append(chunk)
        final_response = chunk
    print(final_response)
    assert len(chunks) > 0
    assert final_response is not None
    assert final_response.message is not None
    assert len(final_response.message.content) != 0
    assert "thoughts" in final_response.message.additional_kwargs
    assert len(final_response.message.additional_kwargs["thoughts"]) != 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_thoughts_with_async_streaming() -> None:
    """Test that thought summaries work correctly with streaming responses."""
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
        generation_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(
                include_thoughts=True,
            ),
        ),
    )

    # Test streaming chat
    messages = [ChatMessage(content="What is your name?", role=MessageRole.USER)]

    stream_response = await llm.astream_chat(messages)

    # Collect all streaming chunks
    chunks = []
    final_response = None
    async for chunk in stream_response:
        chunks.append(chunk)
        final_response = chunk

    assert len(chunks) > 0
    assert final_response is not None
    assert final_response.message is not None
    assert len(final_response.message.content) != 0
    assert "thoughts" in final_response.message.additional_kwargs
    assert len(final_response.message.additional_kwargs["thoughts"]) != 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_thoughts_with_chat() -> None:
    """Test that thought summaries work correctly with chat responses."""
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
        generation_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(
                include_thoughts=True,
            ),
        ),
    )

    # Test streaming chat
    messages = [ChatMessage(content="What is your name?", role=MessageRole.USER)]

    response = llm.chat(messages)
    final_response = response

    assert final_response is not None
    assert final_response.message is not None
    assert len(final_response.message.content) != 0
    assert "thoughts" in final_response.message.additional_kwargs
    assert len(final_response.message.additional_kwargs["thoughts"]) != 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_thoughts_with_async_chat() -> None:
    """Test that thought summaries work correctly with chat responses."""
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
        generation_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(
                include_thoughts=True,
            ),
        ),
    )

    # Test streaming chat
    messages = [ChatMessage(content="What is your name?", role=MessageRole.USER)]

    response = await llm.achat(messages)
    final_response = response

    assert final_response is not None
    assert final_response.message is not None
    assert len(final_response.message.content) != 0
    assert "thoughts" in final_response.message.additional_kwargs
    assert len(final_response.message.additional_kwargs["thoughts"]) != 0
