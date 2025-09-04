from typing import Any, AsyncGenerator, Coroutine, Dict, List, Optional, Sequence, Union
from unittest.mock import patch

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
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
