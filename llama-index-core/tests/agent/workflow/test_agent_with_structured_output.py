import pytest
from typing import Any, List, Type, Optional, Dict
from typing_extensions import override
from pydantic import BaseModel

from llama_index.core.types import Model
from llama_index.core.llms import (
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    LLM,
)
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.tools import ToolSelection
from llama_index.core.agent.workflow import FunctionAgent


class TestLLM(LLM):
    def __init__(self, responses: List[ChatMessage], structured_response: str):
        super().__init__()
        self._responses = responses
        self._structured_response = structured_response
        self._response_index = 0

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    async def astream_chat(
        self, messages: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = None
        if self._responses:
            response_msg = self._responses[self._response_index]
            self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
            if response_msg:
                yield ChatResponse(
                    message=response_msg,
                    delta=response_msg.content,
                    raw={"content": response_msg.content},
                )

        return _gen()

    async def astream_chat_with_tools(
        self, tools: List[Any], chat_history: List[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        response_msg = None
        if self._responses:
            response_msg = self._responses[self._response_index]
            self._response_index = (self._response_index + 1) % len(self._responses)

        async def _gen():
            if response_msg:
                yield ChatResponse(
                    message=response_msg,
                    delta=response_msg.content,
                    raw={"content": response_msg.content},
                )

        return _gen()

    def get_tool_calls_from_response(
        self, response: ChatResponse, **kwargs: Any
    ) -> List[ToolSelection]:
        return response.message.additional_kwargs.get("tool_calls", [])

    @override
    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        return output_cls.model_validate_json(self._structured_response)

    @override
    async def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        return output_cls.model_validate_json(self._structured_response)

    async def achat(self, *args, **kwargs):
        pass

    def chat(self, *args, **kwargs):
        pass

    def stream_chat(self, *args, **kwargs):
        pass

    def complete(self, *args, **kwargs):
        pass

    async def acomplete(self, *args, **kwargs):
        pass

    def stream_complete(self, *args, **kwargs):
        pass

    async def astream_complete(self, *args, **kwargs):
        pass

    def _prepare_chat_with_tools(self, *args, **kwargs):
        return {}


class Structure(BaseModel):
    hello: str
    world: int


@pytest.fixture()
def function_agent():
    return FunctionAgent(
        name="retriever",
        description="Manages data retrieval",
        system_prompt="You are a retrieval assistant.",
        llm=TestLLM(
            responses=[
                ChatMessage(role="assistant", content="Success with the FunctionAgent")
            ],
            structured_response='{"hello":"hello","world":1}',
        ),
        output_cls=Structure,
    )


@pytest.mark.asyncio
async def test_single_function_agent(function_agent):
    """Test single agent with state management."""
    handler = function_agent.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.structured_response == Structure(hello="hello", world=1)
