import pytest
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ToolCallBlock
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


def test_prepare_chat_with_tools_tool_required(mocked_llm):
    """Test that tool_required is correctly passed to the API request when True."""
    # We access the internal method via the public chat_with_tools flow or directly for testing
    # Since _prepare_chat_with_tools is the logic we want to verify:

    result = mocked_llm._prepare_chat_with_tools(
        tools=[search_tool], tool_required=True
    )

    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.ANY
    )
    assert len(result["tools"]) == 1
    # In the new SDK/Implementation, tools are converted to types.Tool objects containing function_declarations
    assert len(result["tools"][0].function_declarations) == 1
    assert result["tools"][0].function_declarations[0].name == "search_tool"


def test_prepare_chat_with_tools_tool_not_required(mocked_llm):
    result = mocked_llm._prepare_chat_with_tools(
        tools=[search_tool], tool_required=False
    )
    # "auto" is the default when tools are present but not forced
    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.AUTO
    )


def test_prepare_chat_with_tools_explicit_tool_choice(mocked_llm):
    # Test explicit "none"
    result = mocked_llm._prepare_chat_with_tools(
        tools=[search_tool], tool_choice="none"
    )
    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.NONE
    )

    # Test explicit "any"
    result = mocked_llm._prepare_chat_with_tools(tools=[search_tool], tool_choice="any")
    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.ANY
    )

    # Test explicit tool name
    result = mocked_llm._prepare_chat_with_tools(
        tools=[search_tool], tool_choice="search_tool"
    )
    # When a specific tool is chosen, the mode is ANY, and allowed_function_names is set
    assert (
        result["tool_config"].function_calling_config.mode
        == types.FunctionCallingConfigMode.ANY
    )
    assert result["tool_config"].function_calling_config.allowed_function_names == [
        "search_tool"
    ]


def test_get_tool_calls_from_response(mocked_llm):
    # Create a dummy ChatResponse with a ToolCallBlock
    from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole

    tool_block = ToolCallBlock(
        tool_name="search_tool",
        tool_kwargs={"query": "python"},
        tool_call_id="call_123",
    )

    response = ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[tool_block],
        )
    )

    tool_calls = mocked_llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "search_tool"
    assert tool_calls[0].tool_kwargs == {"query": "python"}


def test_cached_content_initialization(mock_genai_client_factory):
    cached_content_value = "projects/p/locations/l/cachedContents/c123"

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        cached_content=cached_content_value,
        api_key="dummy",
    )

    # Verify it's stored in the generation config dict
    assert llm._generation_config["cached_content"] == cached_content_value


def test_built_in_tool_initialization(mock_genai_client_factory):
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite", built_in_tool=grounding_tool, api_key="dummy"
    )

    assert llm.built_in_tool == grounding_tool
    assert "tools" in llm._generation_config
    # GenerateContentConfig.model_dump() serializes Tool objects into dicts.
    tools_dump = llm._generation_config["tools"]
    assert isinstance(tools_dump, list)
    assert tools_dump and tools_dump[0].get("google_search") is not None
    # Ensure the stored built-in tool matches by type/shape.
    assert tools_dump[0] == grounding_tool.model_dump(exclude_none=False)


def test_built_in_tool_merge_error(mock_genai_client_factory):
    """Test that we cannot mix built-in tools with existing tools in generation_config."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    existing_config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())]
    )

    with pytest.raises(ValueError, match="Providing multiple Google GenAI tools"):
        GoogleGenAI(
            model="gemini-2.5-flash-lite",
            built_in_tool=grounding_tool,
            generation_config=existing_config,
            api_key="dummy",
        )


def test_built_in_tool_with_invalid_tool(mock_genai_client_factory):
    """built_in_tool=None should not break tools config."""
    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=None,
        api_key="dummy",
    )

    assert llm.built_in_tool is None
    # Implementation stores a tools key only when tool is set.
    # Ensure we don't accidentally create a non-empty tools list.
    tools = llm._generation_config.get("tools")
    assert tools is None or tools == []


def test_built_in_tool_config_merge_edge_cases(mock_genai_client_factory):
    """Edge case tests for tool merging."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    # generation_config already has empty tools list
    empty_tools_config = types.GenerateContentConfig(temperature=0.7, tools=[])
    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=grounding_tool,
        generation_config=empty_tools_config,
        api_key="dummy",
    )

    assert "tools" in llm._generation_config
    assert len(llm._generation_config["tools"]) == 1
    assert llm._generation_config["temperature"] == 0.7

    # generation_config already has an existing tool -> error
    existing_tool = types.Tool(google_search=types.GoogleSearch())
    existing_tools_config = types.GenerateContentConfig(
        temperature=0.3,
        tools=[existing_tool],
    )

    with pytest.raises(ValueError, match="Providing multiple Google GenAI tools"):
        GoogleGenAI(
            model="gemini-2.5-flash-lite",
            built_in_tool=grounding_tool,
            generation_config=existing_tools_config,
            api_key="dummy",
        )


@pytest.mark.asyncio
async def test_built_in_tool_in_chat_params(mock_genai_client_factory):
    """
    Verify built-in tool makes it into the config used for chats.

    Verifies the built-in tool makes it into the config used for chats.
    """
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=grounding_tool,
        api_key="dummy",
    )

    prepared = await llm._chat_runner.prepare(
        messages=[
            ChatMessage(role=MessageRole.USER, content="What is the weather today?")
        ],
        generation_config=llm._generation_config,
    )

    assert prepared.chat_kwargs["config"].tools is not None
    assert len(prepared.chat_kwargs["config"].tools) == 1
    assert prepared.chat_kwargs["config"].tools[0] == grounding_tool


@pytest.mark.asyncio
async def test_cached_content_in_chat_params(mock_genai_client_factory):
    """Verify cached_content makes it into the config used for chats."""
    cached_content_value = (
        "projects/test-project/locations/us-central1/cachedContents/test-cache"
    )

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        cached_content=cached_content_value,
        api_key="dummy",
    )

    prepared = await llm._chat_runner.prepare(
        messages=[ChatMessage(role=MessageRole.USER, content="Test message")],
        generation_config=llm._generation_config,
    )

    assert prepared.chat_kwargs["config"].cached_content == cached_content_value


def test_built_in_tool_with_generation_config(mock_genai_client_factory):
    """
    Ensure generation_config values are preserved and built_in_tool is appended.

    Ensures custom generation_config values are preserved and built_in_tool is
    appended when generation_config has no tools.
    """
    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    llm = GoogleGenAI(
        model="gemini-2.5-flash-lite",
        built_in_tool=grounding_tool,
        generation_config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=1000,
        ),
        api_key="dummy",
    )

    assert llm.built_in_tool == grounding_tool
    assert llm._generation_config["temperature"] == 0.5
    assert llm._generation_config["max_output_tokens"] == 1000
    assert "tools" in llm._generation_config
    assert len(llm._generation_config["tools"]) == 1
    assert llm._generation_config["tools"][0] == grounding_tool
