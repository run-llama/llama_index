import os
import pytest
from typing import Optional
from pydantic import BaseModel, Field
from google.genai.types import GenerateContentConfig, ThinkingConfig
from unittest.mock import MagicMock

from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
    VideoBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.core.prompts import PromptTemplate, ChatPromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types
from tests.conftest import SKIP_GEMINI, Poem, Schema, BlogPost


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_complete(llm: GoogleGenAI) -> None:
    response = llm.complete("Write a poem about a magic backpack")
    assert response is not None
    assert len(response.text) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_acomplete(llm: GoogleGenAI) -> None:
    response = await llm.acomplete("Write a poem about a magic backpack")
    assert response is not None
    assert len(response.text) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_chat(llm: GoogleGenAI) -> None:
    message = ChatMessage(content="Write a poem about a magic backpack")
    response = llm.chat(messages=[message])
    assert response is not None
    assert response.message.content and len(response.message.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_achat(llm: GoogleGenAI) -> None:
    message = ChatMessage(content="Write a poem about a magic backpack")
    response = await llm.achat(messages=[message])
    assert response is not None
    assert response.message.content and len(response.message.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_stream_chat(llm: GoogleGenAI) -> None:
    message = ChatMessage(content="Write a poem about a magic backpack")
    chunks = list(llm.stream_chat(messages=[message]))
    assert len(chunks) > 0
    assert all(isinstance(chunk.message.content, str) for chunk in chunks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_astream_chat(llm: GoogleGenAI) -> None:
    message = ChatMessage(content="Write a poem about a magic backpack")
    response_gen = await llm.astream_chat(messages=[message])
    chunks = [chunk async for chunk in response_gen]
    assert len(chunks) > 0
    assert all(isinstance(chunk.message.content, str) for chunk in chunks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_stream_complete(llm: GoogleGenAI) -> None:
    prompt = "Write a poem about a magic backpack"
    chunks = list(llm.stream_complete(prompt))
    assert len(chunks) > 0
    assert all(isinstance(chunk.text, str) for chunk in chunks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_astream_complete(llm: GoogleGenAI) -> None:
    prompt = "Write a poem about a magic backpack"
    response_gen = await llm.astream_complete(prompt)
    chunks = [chunk async for chunk in response_gen]
    assert len(chunks) > 0
    assert all(isinstance(chunk.text, str) for chunk in chunks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_structured_predict(llm: GoogleGenAI) -> None:
    response = llm.structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )
    assert isinstance(response, Poem)
    assert len(response.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_astructured_predict(llm: GoogleGenAI) -> None:
    """Integration test for async structured prediction."""
    response = await llm.astructured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )

    assert response is not None
    assert isinstance(response, Poem)
    assert len(response.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_stream_structured_predict(llm: GoogleGenAI) -> None:
    response_gen = llm.stream_structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )
    result = None
    for partial in response_gen:
        result = partial

    assert result is not None
    assert isinstance(result, Poem)
    assert len(result.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_astream_structured_predict(llm: GoogleGenAI) -> None:
    response_gen = await llm.astream_structured_predict(
        output_cls=Poem,
        prompt=PromptTemplate("Write a poem about a magic backpack"),
    )
    result = None
    async for partial in response_gen:
        result = partial

    assert result is not None
    assert isinstance(result, Poem)
    assert len(result.content) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_complex_structured_predict(llm: GoogleGenAI) -> None:
    prompt = PromptTemplate("Generate a simple database structure")
    response = llm.structured_predict(output_cls=Schema, prompt=prompt)
    assert isinstance(response, Schema)
    assert len(response.tables) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_anyof_optional_structured_predict(llm: GoogleGenAI) -> None:
    class Person(BaseModel):
        last_name: str = Field(description="Last name")
        first_name: Optional[str] = Field(None, description="Optional first name")

    prompt = PromptTemplate("Create a fake person with just a last name")
    response = llm.structured_predict(output_cls=Person, prompt=prompt)

    assert response is not None
    assert isinstance(response, Person)
    assert isinstance(response.last_name, str)
    # This might be None or a string, but the key is that it didn't crash
    if response.first_name is not None:
        assert isinstance(response.first_name, str)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_as_structured_llm(llm: GoogleGenAI) -> None:
    prompt = PromptTemplate("Generate content")

    # Simple
    poem_resp = llm.as_structured_llm(output_cls=Poem, prompt=prompt).complete(
        "Write a poem"
    )
    assert isinstance(poem_resp.raw, Poem)

    # Complex
    schema_resp = llm.as_structured_llm(output_cls=Schema, prompt=prompt).complete(
        "Generate a simple database structure"
    )
    assert isinstance(schema_resp.raw, Schema)


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
def test_as_structure_llm_with_config(llm: GoogleGenAI) -> None:
    # Test overriding config in complete call
    response = (
        llm.as_structured_llm(output_cls=Poem)
        .complete(
            prompt="Write a poem about a magic backpack",
            config={"temperature": 0.1},
        )
        .raw
    )
    assert isinstance(response, Poem)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_as_structured_llm_async(llm: GoogleGenAI) -> None:
    prompt = PromptTemplate("Generate content")

    poem_resp = await llm.as_structured_llm(output_cls=Poem, prompt=prompt).acomplete(
        "Write a poem about a magic backpack"
    )
    assert isinstance(poem_resp.raw, Poem)
    assert len(poem_resp.raw.content) > 0

    schema_resp = await llm.as_structured_llm(
        output_cls=Schema, prompt=prompt
    ).acomplete("Generate a simple database structure")
    assert isinstance(schema_resp.raw, Schema)
    assert len(schema_resp.raw.schema_name) > 0
    assert len(schema_resp.raw.tables) > 0


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_as_structured_llm_async_with_config(llm: GoogleGenAI) -> None:
    resp = await llm.as_structured_llm(output_cls=Poem).acomplete(
        prompt="Write a poem about a magic backpack",
        config={"temperature": 0.1},
    )

    assert isinstance(resp.raw, Poem)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_multimodal_predict(llm: GoogleGenAI) -> None:
    # Test with Image
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

    resp = llm.structured_predict(
        output_cls=Response, prompt=ChatPromptTemplate(message_templates=chat_messages)
    )
    assert isinstance(resp, Response)
    assert "wiki" in resp.answer.lower()


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_video_predict(llm: GoogleGenAI) -> None:
    chat_messages = [
        ChatMessage(
            content=[
                TextBlock(text="where is this scene happening?"),
                VideoBlock(
                    url="https://upload.wikimedia.org/wikipedia/commons/transcoded/2/28/"
                    "TikTok_and_YouTube_Shorts_example.webm/TikTok_and_YouTube_Shorts_example.webm.720p.vp9.webm"
                ),
            ],
            role=MessageRole.USER,
        ),
    ]

    # Using standard predict string response
    answer = llm.predict(prompt=ChatPromptTemplate(message_templates=chat_messages))
    assert "space" in answer.lower() or "rocket" in answer.lower()


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_predict_with_large_video(llm: GoogleGenAI) -> None:
    client = llm._client
    before_file_names = {file.name for file in client.files.list()}

    chat_messages = [
        ChatMessage(
            content=[
                TextBlock(text="what is this video about?"),
                VideoBlock(
                    url="https://upload.wikimedia.org/wikipedia/commons/transcoded/f/f0/Die_Franz%C3%B6sische_Revolution_und_Napoleon_-_Planet_Wissen.webm/Die_Franz%C3%B6sische_Revolution_und_Napoleon_-_Planet_Wissen.webm.720p.vp9.webm"
                ),
            ],
            role=MessageRole.USER,
        ),
    ]
    answer = llm.predict(prompt=ChatPromptTemplate(message_templates=chat_messages))
    assert "revolution" in answer.lower()

    after_file_names = {file.name for file in client.files.list()}
    assert before_file_names == after_file_names


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_google_search(llm: GoogleGenAI) -> None:
    # We need to re-init LLM with the tool
    search_llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=types.Tool(google_search=types.GoogleSearch()),
    )

    response = search_llm.complete("What is the current weather in San Francisco?")
    assert len(response.text) > 0

    # Check grounding metadata if present
    if hasattr(response, "raw") and isinstance(response.raw, dict):
        # It's possible the model didn't use search, but we check structure if it did
        if "grounding_metadata" in response.raw:
            assert isinstance(response.raw["grounding_metadata"], dict)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_google_search_grounding_metadata(llm: GoogleGenAI) -> None:
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm_with_search = GoogleGenAI(
        model=llm.model,
        built_in_tool=grounding_tool,
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    response = llm_with_search.complete("What is the capital of Japan?")
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0

    raw_response = response.raw
    assert raw_response is not None
    assert isinstance(raw_response, dict)

    # Grounding metadata may not always be present depending on whether the model uses search.
    if "grounding_metadata" in raw_response:
        grounding_metadata = raw_response["grounding_metadata"]
        assert isinstance(grounding_metadata, dict)

        if "web_search_queries" in grounding_metadata:
            assert isinstance(grounding_metadata["web_search_queries"], list)
            assert len(grounding_metadata["web_search_queries"]) > 0
            for query in grounding_metadata["web_search_queries"]:
                assert isinstance(query, str)
                assert len(query.strip()) > 0

        if "search_entry_point" in grounding_metadata:
            assert isinstance(grounding_metadata["search_entry_point"], dict)

        if "grounding_supports" in grounding_metadata:
            assert isinstance(grounding_metadata["grounding_supports"], list)

        if "grounding_chunks" in grounding_metadata:
            assert isinstance(grounding_metadata["grounding_chunks"], list)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_code_execution() -> None:
    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=types.Tool(code_execution=types.ToolCodeExecution()),
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    response = llm.complete("Calculate 20th fibonacci number.")
    assert len(response.text) > 0
    assert "6765" in response.text
    assert isinstance(response.raw, dict)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_with_streaming() -> None:
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=grounding_tool,
    )

    messages = [ChatMessage(content="Who won the Euro 2024?", role=MessageRole.USER)]
    stream_response = llm.stream_chat(messages)

    chunks = []
    final_response = None
    for chunk in stream_response:
        chunks.append(chunk)
        final_response = chunk

    assert len(chunks) > 0
    assert final_response is not None
    assert final_response.message is not None
    assert len(final_response.message.content) > 0

    if hasattr(final_response, "raw") and final_response.raw:
        raw_response = final_response.raw
        if "grounding_metadata" in raw_response:
            assert isinstance(raw_response["grounding_metadata"], dict)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_built_in_tool_error_recovery() -> None:
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=grounding_tool,
    )

    response = llm.complete("Hello, how are you?")
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0
    assert isinstance(response.raw, dict)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_built_in_tool_async_compatibility() -> None:
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=grounding_tool,
    )

    response = await llm.acomplete("What is machine learning?")
    assert response is not None
    assert response.text is not None
    assert len(response.text) > 0

    messages = [ChatMessage(content="Explain quantum computing", role=MessageRole.USER)]
    chat_response = await llm.achat(messages)
    assert chat_response is not None
    assert chat_response.message is not None
    assert len(chat_response.message.content) > 0

    assert llm.built_in_tool == grounding_tool


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_thoughts_with_streaming() -> None:
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        generation_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
    )

    messages = [ChatMessage(content="What is your name?", role=MessageRole.USER)]
    chunks = list(llm.stream_chat(messages))
    assert len(chunks) > 0

    final = chunks[-1]
    assert len(final.message.content) != 0
    assert any(isinstance(b, ThinkingBlock) for b in final.message.blocks)
    assert (
        len(
            "".join(
                [
                    b.content or ""
                    for b in final.message.blocks
                    if isinstance(b, ThinkingBlock)
                ]
            )
        )
        != 0
    )


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_thoughts_with_async_streaming() -> None:
    # flash lite doesn't bring thought blocks
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        generation_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
    )

    messages = [ChatMessage(content="What is your name?", role=MessageRole.USER)]
    response_gen = await llm.astream_chat(messages)
    chunks = [chunk async for chunk in response_gen]
    assert len(chunks) > 0

    final = chunks[-1]
    assert len(final.message.content) != 0
    assert any(isinstance(b, ThinkingBlock) for b in final.message.blocks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_thoughts_with_chat() -> None:
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        generation_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
    )

    messages = [ChatMessage(content="What is your name?", role=MessageRole.USER)]
    resp = llm.chat(messages)
    assert len(resp.message.content) != 0
    assert any(isinstance(b, ThinkingBlock) for b in resp.message.blocks)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_thoughts_with_async_chat() -> None:
    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        generation_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
    )

    messages = [ChatMessage(content="What is your name?", role=MessageRole.USER)]
    resp = await llm.achat(messages)
    assert len(resp.message.content) != 0
    assert any(isinstance(b, ThinkingBlock) for b in resp.message.blocks)


def test_built_in_tool_in_response() -> None:
    """
    Validate grounding_metadata extraction from response conversion.

    This only checks response conversion and does not require API access.
    """
    from llama_index.llms.google_genai.conversion.responses import (
        ResponseConverter,
        GeminiResponseParseState,
    )

    converter = ResponseConverter()
    state = GeminiResponseParseState()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = types.FinishReason.STOP
    mock_candidate.content.role = "model"
    mock_candidate.content.parts = [
        types.Part(text="Test response with search results")
    ]
    mock_response.candidates = [mock_candidate]
    mock_response.prompt_feedback = None
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.model_dump.return_value = {
        "prompt_token_count": 10,
        "candidates_token_count": 20,
        "total_token_count": 30,
    }

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

    mock_candidate.model_dump.return_value = {
        "finish_reason": types.FinishReason.STOP,
        "content": {
            "role": "model",
            "parts": [{"text": "Test response with search results"}],
        },
        "grounding_metadata": grounding_metadata,
    }

    chat_response = converter.to_chat_response(mock_response, state=state)
    assert chat_response.message.role == MessageRole.ASSISTANT
    assert len(chat_response.message.blocks) == 1
    assert chat_response.message.blocks[0].text == "Test response with search results"
    assert "grounding_metadata" in chat_response.raw
    assert chat_response.raw["grounding_metadata"]["web_search_queries"] == [
        "test query"
    ]


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

    tool_calls = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"
    assert tool_calls[0].tool_kwargs == {"a": 2, "b": 3}


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_tool_required_integration(llm: GoogleGenAI) -> None:
    response = llm.chat_with_tools(
        user_msg="What is the weather in Paris?",
        tools=[search_tool],
        tool_required=True,
    )

    assert len([b for b in response.message.blocks if isinstance(b, ToolCallBlock)]) > 0

    response = llm.chat_with_tools(
        user_msg="Say hello!",
        tools=[search_tool],
        tool_required=False,
    )
    assert response is not None


@pytest.mark.skipif(SKIP_GEMINI, reason="GOOGLE_API_KEY not set")
def test_optional_lists_nested_gemini(llm: GoogleGenAI) -> None:
    """Integration test for nested optional list structured generation."""
    blogpost = (
        llm.as_structured_llm(output_cls=BlogPost)
        .complete(prompt="Write a blog post with at least 3 contents")
        .raw
    )
    assert isinstance(blogpost, BlogPost)
    assert len(blogpost.contents) >= 3
