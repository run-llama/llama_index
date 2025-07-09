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
from llama_index.core.agent.workflow import AgentWorkflow
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
def function_agent_output_cls():
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


def structured_function_fn(*args, **kwargs) -> Structure:
    return Structure(hello="bonjour", world=2)


async def astructured_function_fn(*args, **kwargs) -> Structure:
    return Structure(hello="guten tag", world=3)


@pytest.fixture()
def function_agent_struct_fn():
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
        structured_output_fn=structured_function_fn,
    )


@pytest.fixture()
def function_agent_astruct_fn():
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
        structured_output_fn=astructured_function_fn,
    )


@pytest.mark.asyncio
async def test_output_cls_agent(function_agent_output_cls: FunctionAgent):
    """Test single agent with state management."""
    handler = function_agent_output_cls.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(hello="hello", world=1)


@pytest.mark.asyncio
async def test_structured_fn_agent(function_agent_struct_fn: FunctionAgent):
    """Test single agent with state management."""
    handler = function_agent_struct_fn.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(hello="bonjour", world=2)


@pytest.mark.asyncio
async def test_astructured_fn_agent(function_agent_astruct_fn: FunctionAgent):
    """Test single agent with state management."""
    handler = function_agent_astruct_fn.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(
        hello="guten tag", world=3
    )


@pytest.mark.asyncio
async def test_structured_output_agentworkflow(
    function_agent_output_cls: FunctionAgent,
) -> None:
    wf = AgentWorkflow(
        agents=[function_agent_output_cls],
        root_agent=function_agent_output_cls.name,
        output_cls=Structure,
    )
    handler = wf.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(hello="hello", world=1)


@pytest.mark.asyncio
async def test_structured_output_fn_agentworkflow(
    function_agent_output_cls: FunctionAgent,
) -> None:
    wf = AgentWorkflow(
        agents=[function_agent_output_cls],
        root_agent=function_agent_output_cls.name,
        structured_output_fn=structured_function_fn,
    )
    handler = wf.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(hello="bonjour", world=2)


@pytest.mark.asyncio
async def test_astructured_output_fn_agentworkflow(
    function_agent_output_cls: FunctionAgent,
) -> None:
    wf = AgentWorkflow(
        agents=[function_agent_output_cls],
        root_agent=function_agent_output_cls.name,
        structured_output_fn=astructured_function_fn,
    )
    handler = wf.run(user_msg="test")
    async for _ in handler.stream_events():
        pass

    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(
        hello="guten tag", world=3
    )
