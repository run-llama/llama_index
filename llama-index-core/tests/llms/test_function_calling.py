from typing import Any, AsyncGenerator, Coroutine, Dict, List, Optional, Sequence, Union
from unittest.mock import patch

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.program.function_program import FunctionTool, get_function_tool
from llama_index.core.tools.types import BaseTool
from pydantic import BaseModel, Field


class MockFunctionCallingLLM(FunctionCallingLLM):
    def __init__(self, tool_selection: List[ToolSelection]):
        super().__init__()
        self._tool_selection = tool_selection

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Coroutine[Any, Any, ChatResponse]:
        return ChatResponse(message=ChatMessage(role="user", content=""))

    def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Coroutine[Any, Any, CompletionResponse]:
        pass

    def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Coroutine[Any, Any, AsyncGenerator[ChatResponse, None]]:
        pass

    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Coroutine[Any, Any, AsyncGenerator[CompletionResponse, None]]:
        pass

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="user", content=""))

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        pass

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        pass

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> ChatResponseGen:
        pass

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"messages": []}

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolSelection]:
        return self._tool_selection


class MockFunctionCallingLLMWithoutToolRequired(MockFunctionCallingLLM):
    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Note: no tool_required parameter in signature
        return {"messages": []}


class MockStreamingFunctionCallingLLM(MockFunctionCallingLLM):
    """Mock LLM that streams cumulative responses and records validation calls."""

    def __init__(
        self,
        tool_selection: List[ToolSelection],
        stream_responses: List[ChatResponse],
    ):
        super().__init__(tool_selection)
        self._stream_responses = stream_responses
        self._validated: List[ChatResponse] = []

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        def gen() -> ChatResponseGen:
            yield from self._stream_responses

        return gen()

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for response in self._stream_responses:
                yield response

        return gen()

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: Sequence["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        self._validated.append(response)
        # mimic in-place normalization such as force_single_tool_call
        response.message.additional_kwargs["validated"] = True
        return response


class Person(BaseModel):
    name: str = Field(description="Person name")


@pytest.fixture()
def person_tool() -> FunctionTool:
    return get_function_tool(Person)


@pytest.fixture()
def person_tool_selection(person_tool: FunctionTool) -> ToolSelection:
    return ToolSelection(
        tool_id="",
        tool_name=person_tool.metadata.name,
        tool_kwargs={},
    )


def test_predict_and_call(
    person_tool: FunctionTool, person_tool_selection: ToolSelection
) -> None:
    """Test predict_and_call will return ToolOutput with error rather than raising one."""
    llm = MockFunctionCallingLLM([person_tool_selection])
    response = llm.predict_and_call(tools=[person_tool])
    assert all(tool_output.is_error for tool_output in response.sources)


def test_predict_and_call_throws_if_error_on_tool(
    person_tool: FunctionTool, person_tool_selection: ToolSelection
) -> None:
    """Test predict_and_call will raise an error."""
    llm = MockFunctionCallingLLM([person_tool_selection])
    with pytest.raises(ValueError):
        llm.predict_and_call(tools=[person_tool], error_on_tool_error=True)


@pytest.mark.asyncio
async def test_apredict_and_call(
    person_tool: FunctionTool, person_tool_selection: ToolSelection
) -> None:
    """Test apredict_and_call will return ToolOutput with error rather than raising one."""
    llm = MockFunctionCallingLLM([person_tool_selection])
    response = await llm.apredict_and_call(tools=[person_tool])
    assert all(tool_output.is_error for tool_output in response.sources)


@pytest.mark.asyncio
async def test_apredict_and_call_throws_if_error_on_tool(
    person_tool: FunctionTool, person_tool_selection: ToolSelection
) -> None:
    """Test apredict_and_call will raise an error."""
    llm = MockFunctionCallingLLM([person_tool_selection])
    with pytest.raises(ValueError):
        await llm.apredict_and_call(tools=[person_tool], error_on_tool_error=True)


@pytest.fixture()
def stream_responses() -> List[ChatResponse]:
    # streaming responses are cumulative: each chunk carries the full message
    # so far plus the new delta
    return [
        ChatResponse(message=ChatMessage(role="assistant", content="foo"), delta="foo"),
        ChatResponse(
            message=ChatMessage(role="assistant", content="foobar"), delta="bar"
        ),
    ]


def test_stream_chat_with_tools_validates_final_response(
    person_tool: FunctionTool,
    person_tool_selection: ToolSelection,
    stream_responses: List[ChatResponse],
) -> None:
    """Test that the final streamed response gets validated once the stream is exhausted."""
    llm = MockStreamingFunctionCallingLLM([person_tool_selection], stream_responses)

    chunks = list(llm.stream_chat_with_tools(tools=[person_tool]))

    assert chunks == stream_responses
    assert llm._validated == [stream_responses[-1]]
    # in-place changes made by validation are visible on the final response
    assert chunks[-1].message.additional_kwargs.get("validated") is True
    # intermediate chunks are passed through untouched
    assert "validated" not in chunks[0].message.additional_kwargs


@pytest.mark.asyncio
async def test_astream_chat_with_tools_validates_final_response(
    person_tool: FunctionTool,
    person_tool_selection: ToolSelection,
    stream_responses: List[ChatResponse],
) -> None:
    """Test that the async streaming path validates the final response."""
    llm = MockStreamingFunctionCallingLLM([person_tool_selection], stream_responses)

    response_gen = await llm.astream_chat_with_tools(tools=[person_tool])
    chunks = [chunk async for chunk in response_gen]

    assert chunks == stream_responses
    assert llm._validated == [stream_responses[-1]]
    assert chunks[-1].message.additional_kwargs.get("validated") is True


def test_stream_chat_with_tools_partial_consumption_skips_validation(
    person_tool: FunctionTool,
    person_tool_selection: ToolSelection,
    stream_responses: List[ChatResponse],
) -> None:
    """Test that validation only runs when the stream is fully consumed."""
    llm = MockStreamingFunctionCallingLLM([person_tool_selection], stream_responses)

    response_gen = llm.stream_chat_with_tools(tools=[person_tool])
    next(response_gen)
    response_gen.close()

    assert llm._validated == []


def test_stream_chat_with_tools_empty_stream(
    person_tool: FunctionTool, person_tool_selection: ToolSelection
) -> None:
    """Test that an empty stream neither errors nor validates."""
    llm = MockStreamingFunctionCallingLLM([person_tool_selection], [])

    assert list(llm.stream_chat_with_tools(tools=[person_tool])) == []
    assert llm._validated == []


def test_stream_chat_with_tools_validation_error_raised_at_stream_end(
    person_tool: FunctionTool,
    person_tool_selection: ToolSelection,
    stream_responses: List[ChatResponse],
) -> None:
    """Test that a raising validator surfaces the error when the stream ends."""
    llm = MockStreamingFunctionCallingLLM([person_tool_selection], stream_responses)

    with patch.object(
        llm,
        "_validate_chat_with_tools_response",
        side_effect=ValueError("invalid tool call"),
    ):
        response_gen = llm.stream_chat_with_tools(tools=[person_tool])
        chunks = [next(response_gen), next(response_gen)]
        with pytest.raises(ValueError, match="invalid tool call"):
            next(response_gen)

    assert chunks == stream_responses


def test_tool_required_compatibility_without_support(
    person_tool: FunctionTool, person_tool_selection: ToolSelection
) -> None:
    """Test that tool_required parameter is not passed to LLMs that don't support it."""
    llm = MockFunctionCallingLLMWithoutToolRequired([person_tool_selection])

    # Mock the _prepare_chat_with_tools method to capture what arguments it receives
    with patch.object(
        llm, "_prepare_chat_with_tools", wraps=llm._prepare_chat_with_tools
    ) as mock_prepare:
        llm.chat_with_tools(tools=[person_tool], tool_required=True)

        # Verify that tool_required was NOT passed to _prepare_chat_with_tools
        args, kwargs = mock_prepare.call_args
        assert "tool_required" not in kwargs


def test_tool_required_compatibility_with_support(
    person_tool: FunctionTool, person_tool_selection: ToolSelection
) -> None:
    """Test that tool_required parameter is passed to LLMs that support it."""
    llm = MockFunctionCallingLLM([person_tool_selection])

    # Mock the _prepare_chat_with_tools method to capture what arguments it receives
    with patch.object(
        llm, "_prepare_chat_with_tools", wraps=llm._prepare_chat_with_tools
    ) as mock_prepare:
        llm.chat_with_tools(tools=[person_tool], tool_required=True)

        # Verify that tool_required was passed to _prepare_chat_with_tools
        args, kwargs = mock_prepare.call_args
        assert "tool_required" in kwargs
        assert kwargs["tool_required"] is True
