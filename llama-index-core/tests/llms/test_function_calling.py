from typing import Any, AsyncGenerator, Coroutine, List, Optional, Sequence, Union, Dict

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
from llama_index.core.program.function_program import get_function_tool
from llama_index.core.tools.types import BaseTool
from pydantic import BaseModel, Field


class MockFunctionCallingLLM(FunctionCallingLLM):
    def __init__(self, tool_selection: List[ToolSelection]):
        super().__init__()
        self._tool_selection = tool_selection

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="user", content=""))

    def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Coroutine[Any, Any, ChatResponse]:
        pass

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
        pass

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


class Person(BaseModel):
    name: str = Field(description="Person name")


@pytest.fixture()
def person_tool():
    return get_function_tool(Person)


@pytest.fixture()
def person_tool_selection(person_tool):
    return ToolSelection(
        tool_id="",
        tool_name=person_tool.metadata.name,
        tool_kwargs={},
    )


def test_predict_and_call(person_tool, person_tool_selection):
    llm = MockFunctionCallingLLM([person_tool_selection])
    llm.predict_and_call(tools=[person_tool])


def test_predict_and_call_throws_if_error_on_tool(person_tool, person_tool_selection):
    llm = MockFunctionCallingLLM([person_tool_selection])
    with pytest.raises(ValueError):
        llm.predict_and_call(tools=[person_tool], error_on_tool_error=True)
