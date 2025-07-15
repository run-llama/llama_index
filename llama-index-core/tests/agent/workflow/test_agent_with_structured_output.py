import pytest
import os
from typing import Any, List, Type, Optional, Dict
from typing_extensions import override
from pydantic import BaseModel, Field

from llama_index.core.types import Model
from llama_index.core.llms import (
    LLMMetadata,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    LLM,
)
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    AgentOutput,
    AgentStreamStructuredOutput,
)
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.tools import ToolSelection
from llama_index.core.agent.workflow import FunctionAgent


skip_condition = os.getenv("OPENAI_API_KEY", None) is None


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


def structured_function_fn(*args, **kwargs) -> dict:
    return Structure(hello="bonjour", world=2).model_dump()


async def astructured_function_fn(*args, **kwargs) -> dict:
    return Structure(hello="guten tag", world=3).model_dump()


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
    streaming_event = False
    async for event in handler.stream_events():
        if isinstance(event, AgentStreamStructuredOutput):
            streaming_event = True
    assert streaming_event
    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(hello="hello", world=1)


@pytest.mark.asyncio
async def test_structured_fn_agent(function_agent_struct_fn: FunctionAgent):
    """Test single agent with state management."""
    handler = function_agent_struct_fn.run(user_msg="test")
    streaming_event = False
    async for event in handler.stream_events():
        if isinstance(event, AgentStreamStructuredOutput):
            streaming_event = True
    assert streaming_event
    response = await handler
    assert "Success with the FunctionAgent" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(hello="bonjour", world=2)


@pytest.mark.asyncio
async def test_astructured_fn_agent(function_agent_astruct_fn: FunctionAgent):
    """Test single agent with state management."""
    handler = function_agent_astruct_fn.run(user_msg="test")
    async for event in handler.stream_events():
        if isinstance(event, AgentStreamStructuredOutput):
            streaming_event = True
    assert streaming_event
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
    streaming_event = False
    async for event in handler.stream_events():
        if isinstance(event, AgentStreamStructuredOutput):
            streaming_event = True
    assert streaming_event

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


@pytest.mark.asyncio
@pytest.mark.skipif(condition=skip_condition, reason="OPENAI_API_KEY is not available.")
async def test_multi_agent_openai() -> None:
    from llama_index.llms.openai import OpenAI

    class MathResult(BaseModel):
        operation: str = Field(description="The operation performed")
        result: int = Field(description="The result of the operation")

    main_agent = FunctionAgent(
        llm=OpenAI(model="gpt-4.1"),
        name="MainAgent",
        description="Useful for dispatching tasks.",
        system_prompt="You are the MainAgent. Your task is to distribute tasks to other agents. You must always dispatch tasks to secondary agents. You must never perform tasks yourself.",
        tools=[],
        can_handoff_to=["CalculatorAgent"],
    )

    def multiply(x: int, j: int) -> int:
        """
        Multiply two numbers together.

        Args:
            x (int): first factor
            j (int): second factor
        Returns:
            int: the result of the multiplication

        """
        return x * j

    multiplication_agent = FunctionAgent(
        llm=OpenAI(model="gpt-4.1"),
        name="CalculatorAgent",
        description="Useful for performing operations.",
        system_prompt="You are the CalculatorAgent. Your task is to calculate the results of an operation, if needed using the `multiply`tool you are provided with.",
        tools=[multiply],
    )

    workflow = AgentWorkflow(
        agents=[main_agent, multiplication_agent],
        root_agent=main_agent.name,
        output_cls=MathResult,
    )

    result = await workflow.run(user_msg="What is 30 multiplied by 60?")
    assert isinstance(result, AgentOutput)
    assert isinstance(result.structured_response, dict)
    assert isinstance(result.get_pydantic_model(MathResult), MathResult)
    assert result.get_pydantic_model(MathResult).result == 1800


@pytest.mark.asyncio
async def test_from_tools_or_functions() -> None:
    def multiply(x: int, j: int) -> int:
        """
        Multiply two numbers together.

        Args:
            x (int): first factor
            j (int): second factor
        Returns:
            int: the result of the multiplication

        """
        return x * j

    wf: AgentWorkflow = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[multiply],
        system_prompt="You are an agent.",
        output_cls=Structure,
        llm=TestLLM(
            responses=[
                ChatMessage(role="assistant", content="Success with the workflow!")
            ],
            structured_response=Structure(hello="hello", world=3).model_dump_json(),
        ),
    )
    response = await wf.run(user_msg="Hello world!")
    assert "Success with the workflow!" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(hello="hello", world=3)
    wf1: AgentWorkflow = AgentWorkflow.from_tools_or_functions(
        tools_or_functions=[multiply],
        system_prompt="You are an agent.",
        structured_output_fn=structured_function_fn,
        llm=TestLLM(
            responses=[
                ChatMessage(role="assistant", content="Success with the workflow!")
            ],
            structured_response=Structure(hello="hello", world=3).model_dump_json(),
        ),
    )
    response = await wf1.run(user_msg="Hello world!")
    assert "Success with the workflow!" in str(response.response)
    assert response.get_pydantic_model(Structure) == Structure(hello="bonjour", world=2)


@pytest.mark.asyncio
@pytest.mark.skipif(condition=skip_condition, reason="OPENAI_API_KEY is not available.")
async def test_multi_agent_openai_from_tools() -> None:
    from llama_index.llms.openai import OpenAI

    class MathResult(BaseModel):
        operation: str = Field(description="The operation performed")
        result: int = Field(description="The result of the operation")

    def multiply(x: int, j: int) -> int:
        """
        Multiply two numbers together.

        Args:
            x (int): first factor
            j (int): second factor
        Returns:
            int: the result of the multiplication

        """
        return x * j

    multiplication_wf: AgentWorkflow = AgentWorkflow.from_tools_or_functions(
        llm=OpenAI(model="gpt-4.1"),
        system_prompt="You are the CalculatorAgent. Your task is to calculate the results of an operation, if needed using the `multiply`tool you are provided with.",
        tools_or_functions=[multiply],
        output_cls=MathResult,
    )

    result = await multiplication_wf.run(user_msg="What is 30 multiplied by 60?")
    assert isinstance(result, AgentOutput)
    assert isinstance(result.structured_response, dict)
    assert isinstance(result.get_pydantic_model(MathResult), MathResult)
    assert result.get_pydantic_model(MathResult).result == 1800
