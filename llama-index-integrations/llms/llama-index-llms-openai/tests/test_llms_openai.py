import os

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.tools import FunctionTool

from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import resolve_tool_choice


def test_text_inference_embedding_class():
    names_of_base_classes = [b.__name__ for b in OpenAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def search(query: str) -> str:
    """Search for information about a query."""
    return f"Results for {query}"


# Shared tool for all tests
search_tool = FunctionTool.from_defaults(
    fn=search, name="search_tool", description="A tool for searching information"
)


def test_resolve_tool_choice_utility():
    """Test the resolve_tool_choice utility function directly."""

    # Test with tool_required=True and no explicit tool_choice
    result = resolve_tool_choice(None, tool_required=True)
    assert result == "required"

    # Test with tool_required=False and no explicit tool_choice
    result = resolve_tool_choice(None, tool_required=False)
    assert result == "auto"

    # Test with explicit tool_choice overriding tool_required
    result = resolve_tool_choice("none", tool_required=True)
    assert result == "none"

    # Test with function name tool_choice
    result = resolve_tool_choice("search_tool", tool_required=False)
    assert result == {"type": "function", "function": {"name": "search_tool"}}

    # Test with dict tool_choice
    tool_choice_dict = {"type": "function", "function": {"name": "custom_tool"}}
    result = resolve_tool_choice(tool_choice_dict, tool_required=True)
    assert result == tool_choice_dict


def test_prepare_chat_with_tools_tool_required():
    """Test that tool_required=True is correctly passed to the API request."""
    llm = OpenAI(api_key="test-key")

    result = llm._prepare_chat_with_tools(
        tools=[search_tool], user_msg="Search for Python tutorials", tool_required=True
    )

    assert "messages" in result
    assert "tools" in result
    assert "tool_choice" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["function"]["name"] == "search_tool"
    assert result["tool_choice"] == "required"
    assert result["parallel_tool_calls"] is False


def test_prepare_chat_with_tools_tool_not_required():
    """Test that tool_required=False is correctly passed to the API request."""
    llm = OpenAI(api_key="test-key")

    result = llm._prepare_chat_with_tools(
        tools=[search_tool], user_msg="Search for Python tutorials", tool_required=False
    )

    assert "messages" in result
    assert "tools" in result
    assert "tool_choice" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["function"]["name"] == "search_tool"
    assert result["tool_choice"] == "auto"


def test_prepare_chat_with_tools_default_behavior():
    """Test default behavior when tool_required is not specified (should default to False/auto)."""
    llm = OpenAI(api_key="test-key")

    result = llm._prepare_chat_with_tools(
        tools=[search_tool], user_msg="Search for Python tutorials"
    )

    assert "messages" in result
    assert "tools" in result
    assert "tool_choice" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["function"]["name"] == "search_tool"
    # Should default to "auto" when tool_required=False (default)
    assert result["tool_choice"] == "auto"
    assert result["parallel_tool_calls"] is False


def test_prepare_chat_with_tools_allow_parallel_tool_calls():
    """Test that allow_parallel_tool_calls is forwarded for OpenAI chat completions."""
    llm = OpenAI(api_key="test-key")

    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
        user_msg="Search for Python tutorials",
        allow_parallel_tool_calls=True,
    )

    assert result["parallel_tool_calls"] is True


def test_prepare_chat_with_tools_no_tools():
    """Test _prepare_chat_with_tools with no tools."""
    llm = OpenAI(api_key="test-key")

    result = llm._prepare_chat_with_tools(
        tools=[],
        user_msg="Just a regular message",
        tool_required=True,  # Should be ignored when no tools
    )

    assert "messages" in result
    assert result["tools"] is None
    assert result["tool_choice"] is None
    assert "parallel_tool_calls" not in result


def test_prepare_chat_with_tools_explicit_tool_choice_overrides_tool_required():
    """Test that explicit tool_choice parameter overrides tool_required."""
    llm = OpenAI(api_key="test-key")

    # Test that explicit tool_choice="none" overrides tool_required=True
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
        user_msg="Search for Python tutorials",
        tool_required=True,
        tool_choice="none",
    )

    assert result["tool_choice"] == "none"  # Should be "none", not "required"

    # Test with function name tool_choice
    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
        user_msg="Search for Python tutorials",
        tool_required=True,
        tool_choice="search_tool",
    )

    assert result["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_tool"},
    }


def test_prepare_chat_with_tools_explicit_tool_choice_required():
    """Test that explicit tool_choice="required" works even when tool_required=False."""
    llm = OpenAI(api_key="test-key")

    result = llm._prepare_chat_with_tools(
        tools=[search_tool],
        user_msg="Search for Python tutorials",
        tool_required=False,
        tool_choice="required",
    )

    assert result["tool_choice"] == "required"


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None, reason="OpenAI API key not available"
)
def test_tool_required():
    llm = OpenAI(model="gpt-4.1-mini")
    response = llm.chat_with_tools(
        user_msg="What is the capital of France?",
        tools=[search_tool],
        tool_required=True,
    )
    print(repr(response))
    assert len(response.message.additional_kwargs["tool_calls"]) == 1
    assert (
        len(
            [
                block
                for block in response.message.blocks
                if isinstance(block, ToolCallBlock)
            ]
        )
        == 1
    )


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None, reason="OpenAI API key not available"
)
def test_streaming_with_usage_tokens():
    llm = OpenAI(
        model="gpt-4.1-mini",
        additional_kwargs={"stream_options": {"include_usage": True}},
    )
    response_gen = llm.stream_complete("What is the capital of France?")
    intermediate_response = None
    for chunk in response_gen:
        intermediate_response = chunk

    assert intermediate_response.additional_kwargs["prompt_tokens"] > 0
    assert intermediate_response.additional_kwargs["completion_tokens"] > 0
    assert intermediate_response.additional_kwargs["total_tokens"] > 0


def test_structured_predict_tool_choice_pydantic_program_mode_llm():
    import json

    import httpx
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.types import PydanticProgramMode
    from pydantic import BaseModel

    class Total(BaseModel):
        total_units: int

    sent = []

    def capture(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"total_units": 30}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key="fake-key",
        pydantic_program_mode=PydanticProgramMode.LLM,
        http_client=httpx.Client(transport=httpx.MockTransport(capture)),
        max_retries=0,
    )

    result = llm.structured_predict(
        Total, PromptTemplate("Sum: {text}"), text="A: 10, B: 20"
    )
    assert result.total_units == 30
    assert len(sent) == 1
    assert "tool_choice" not in sent[0] or sent[0].get("tool_choice") is None
    assert sent[0].get("tools") is None


@pytest.mark.asyncio
async def test_astructured_predict_tool_choice_pydantic_program_mode_llm():
    import json

    import httpx
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.types import PydanticProgramMode
    from pydantic import BaseModel

    class Total(BaseModel):
        total_units: int

    sent = []

    async def capture(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"total_units": 42}',
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key="fake-key",
        pydantic_program_mode=PydanticProgramMode.LLM,
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(capture)),
        max_retries=0,
    )

    result = await llm.astructured_predict(
        Total, PromptTemplate("Sum: {text}"), text="A: 20, B: 22"
    )
    assert result.total_units == 42
    assert len(sent) == 1
    assert "tool_choice" not in sent[0] or sent[0].get("tool_choice") is None
    assert sent[0].get("tools") is None
