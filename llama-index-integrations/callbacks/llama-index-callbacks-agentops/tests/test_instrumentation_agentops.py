import pytest
import uuid
import llama_index.core.instrumentation as instrument
from agentops import LLMEvent, ToolEvent, ErrorEvent
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Generator,
    List,
    Optional,
    Sequence,
    Union,
)
from llama_index.core.agent.function_calling.step import FunctionCallingAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.chatml_utils import completion_to_prompt, messages_to_prompt
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import LLM, ToolSelection
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import BaseTool, ToolMetadata
from llama_index.callbacks.agentops.agentops import AgentOpsHandler
from unittest.mock import patch, MagicMock


MOCK_MODEL_NAME = "MockLLMModelName"
MOCK_EXCEPTION_MESSAGE = "This is a test exception"
MOCK_AGENT_PROMPT = "This is a test prompt"
MOCK_AGENT_RESPONSE = "This is a test response"
MOCK_AGENT_TOOL_RESPONSE = "This is a test tool response"
MOCK_TOOL_NAME = "Mock Tool Name"
MOCK_TOOL_ID = "MockToolID"
MOCK_TOOL_SELECTION = ToolSelection(
    tool_id=MOCK_TOOL_ID, tool_name=MOCK_TOOL_NAME, tool_kwargs={"arg1": "val1"}
)
MOCK_TOOL = FunctionTool(
    lambda x: x, ToolMetadata(description="", name=MOCK_TOOL_NAME, return_direct=True)
)

dispatcher = instrument.get_dispatcher("test")


class MockLLM(CustomLLM):
    model: str = Field(default=MOCK_MODEL_NAME)
    throw_chat_error: bool = Field(default=False)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        if self.throw_chat_error:
            raise Exception(MOCK_EXCEPTION_MESSAGE)
        return ChatResponse(message=ChatMessage(content=MOCK_AGENT_RESPONSE))

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        ...

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        ...


class MockFunctionCallingLLM(FunctionCallingLLM):
    model: str = Field(default=MOCK_MODEL_NAME)
    use_tools: bool = Field(default=False)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=ChatMessage(content=MOCK_AGENT_RESPONSE))

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        ...

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        ...

    def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Coroutine[Any, Any, ChatResponse]:
        ...

    def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Coroutine[Any, Any, CompletionResponse]:
        ...

    def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Coroutine[Any, Any, AsyncGenerator[ChatResponse, None]]:
        ...

    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Coroutine[Any, Any, AsyncGenerator[CompletionResponse, None]]:
        ...

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        ...

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        ...

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> ChatResponseGen:
        ...

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_function_calling_model=True)

    def chat_with_tools(
        self,
        tools: List[BaseTool],
        user_msg: Union[str, ChatMessage, None] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any
    ) -> ChatResponse:
        if not chat_history:
            raise ValueError
        return self.chat(chat_history)

    async def achat_with_tools(
        self,
        tools: List[BaseTool],
        user_msg: Union[str, ChatMessage, None] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any
    ) -> ChatResponse:
        return self.chat_with_tools(
            tools, user_msg, chat_history, verbose, allow_parallel_tool_calls, **kwargs
        )

    def get_tool_calls_from_response(
        self,
        response: AgentChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any
    ) -> List[ToolSelection]:
        return [MOCK_TOOL_SELECTION] if self.use_tools else []


class MockAgentWorker(BaseAgentWorker):
    def __init__(self, llm: LLM):
        """Initialize."""
        self._llm = llm

    def initialize_step(self, task: Task, **kwargs: Any) -> TaskStep:
        """Initialize step from task."""
        return TaskStep(
            task_id=task.task_id,
            step_id=str(uuid.uuid4()),
            input=task.input,
            memory=task.memory,
        )

    def run_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        chat_res = self._llm.chat([ChatMessage(content=task.input)])
        chat_agent_res = AgentChatResponse(chat_res.message.content)
        return TaskStepOutput(
            output=chat_agent_res,
            task_step=step,
            is_last=True,
            next_steps=[],
        )

    async def arun_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        return self.run_step(step=step, task=task, **kwargs)

    def stream_step(self, step: TaskStep, task: Task, **kwargs: Any) -> TaskStepOutput:
        chat_res = self._llm.chat([ChatMessage(content=task.input)])
        chat_agent_res = StreamingAgentChatResponse(chat_res.message.content)
        return TaskStepOutput(
            output=chat_agent_res,
            task_step=step,
            is_last=True,
            next_steps=[],
        )

    async def astream_step(
        self, step: TaskStep, task: Task, **kwargs: Any
    ) -> TaskStepOutput:
        return self.stream_step(step=step, task=task, **kwargs)

    def finalize_task(self, task: Task, **kwargs: Any) -> None:
        ...


@pytest.fixture()
def mock_agent() -> AgentRunner:
    return AgentRunner(agent_worker=MockAgentWorker(llm=MockLLM()))


@pytest.fixture()
def mock_error_throwing_agent() -> AgentRunner:
    llm = MockLLM()
    llm.throw_chat_error = True
    return AgentRunner(agent_worker=MockAgentWorker(llm=llm))


@pytest.fixture()
def mock_basic_function_calling_agent() -> AgentRunner:
    return AgentRunner(
        agent_worker=FunctionCallingAgentWorker.from_tools(llm=MockFunctionCallingLLM())
    )


@pytest.fixture()
def mock_function_calling_agent() -> AgentRunner:
    llm = MockFunctionCallingLLM()
    llm.use_tools = True
    return AgentRunner(
        agent_worker=FunctionCallingAgentWorker.from_tools(tools=[MOCK_TOOL], llm=llm)
    )


def test_class():
    names_of_base_classes = [b.__name__ for b in AgentOpsHandler.__mro__]
    assert AgentOpsHandler.__name__ in names_of_base_classes


@pytest.mark.asyncio()
@pytest.mark.parametrize("method", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize(
    "agent_runner_fixture", ["mock_agent", "mock_basic_function_calling_agent"]
)
@patch("llama_index.callbacks.agentops.agentops.base.AOClient")
async def test_agentops_event_handler_emits_llmevents(
    mock_ao_client: MagicMock, method: str, agent_runner_fixture: str, request
):
    # Function calling agent doesn't support stream chat, skip these tests
    if agent_runner_fixture == "mock_basic_function_calling_agent" and (
        method == "stream_chat" or method == "astream_chat"
    ):
        pytest.skip("Stream not supported for function calling agent")

    agent_runner: AgentRunner = request.getfixturevalue(agent_runner_fixture)

    mock_ao_client_instance = MagicMock()
    mock_ao_client.return_value = mock_ao_client_instance

    # Initialize the AgentOps handler
    AgentOpsHandler.init()

    # Initiate a chat with the agent
    if method == "chat":
        agent_runner.chat(MOCK_AGENT_PROMPT)
    elif method == "achat":
        await agent_runner.achat(MOCK_AGENT_PROMPT)
    elif method == "stream_chat":
        agent_runner.stream_chat(MOCK_AGENT_PROMPT)
    else:
        await agent_runner.astream_chat(MOCK_AGENT_PROMPT)

    # We expect one LLMEvent for the prompt, then one for the LLM response
    calls_to_agentops = mock_ao_client_instance.record.call_args_list
    assert len(calls_to_agentops) == 2
    agent_start_chat_event: LLMEvent = calls_to_agentops[0][0][0]
    agent_end_chat_event: LLMEvent = calls_to_agentops[1][0][0]

    expected_agent_prompt = messages_to_prompt([ChatMessage(content=MOCK_AGENT_PROMPT)])
    expected_agent_completion = completion_to_prompt(MOCK_AGENT_RESPONSE)
    assert agent_start_chat_event.prompt == expected_agent_prompt
    assert not agent_start_chat_event.completion
    assert agent_start_chat_event.model == MOCK_MODEL_NAME
    assert agent_end_chat_event.prompt == expected_agent_prompt
    assert agent_end_chat_event.completion == expected_agent_completion
    assert agent_end_chat_event.model == MOCK_MODEL_NAME


@pytest.mark.asyncio()
@pytest.mark.parametrize("method", ["chat", "achat"])
@patch("llama_index.callbacks.agentops.agentops.base.AOClient")
async def test_agentops_event_handler_emits_toolevents(
    mock_ao_client: MagicMock, method: str, mock_function_calling_agent: AgentRunner
):
    mock_ao_client_instance = MagicMock()
    mock_ao_client.return_value = mock_ao_client_instance

    # Initialize the AgentOps handler
    AgentOpsHandler.init()

    # Initiate a chat with the agent
    if method == "chat":
        mock_function_calling_agent.chat(MOCK_AGENT_PROMPT)
    else:
        await mock_function_calling_agent.achat(MOCK_AGENT_PROMPT)

    # Expect event for user query, LLM response, then LLM tool call
    calls_to_agentops = mock_ao_client_instance.record.call_args_list
    assert len(calls_to_agentops) == 3
    agent_start_chat_event: LLMEvent = calls_to_agentops[0][0][0]
    agent_end_chat_event: LLMEvent = calls_to_agentops[1][0][0]
    agent_tool_event: ToolEvent = calls_to_agentops[2][0][0]

    expected_agent_prompt = messages_to_prompt([ChatMessage(content=MOCK_AGENT_PROMPT)])
    expected_agent_completion = completion_to_prompt(MOCK_AGENT_RESPONSE)
    expected_tool_event_name = MOCK_TOOL_NAME
    assert agent_start_chat_event.prompt == expected_agent_prompt
    assert not agent_start_chat_event.completion
    assert agent_start_chat_event.model == MOCK_MODEL_NAME
    assert agent_end_chat_event.prompt == expected_agent_prompt
    assert agent_end_chat_event.completion == expected_agent_completion
    assert agent_end_chat_event.model == MOCK_MODEL_NAME
    assert agent_tool_event.name == expected_tool_event_name


@pytest.mark.asyncio()
@pytest.mark.parametrize("method", ["chat", "achat", "stream_chat", "astream_chat"])
@patch("llama_index.callbacks.agentops.agentops.base.AOClient")
async def test_agentops_event_handler_emits_errorevents(
    mock_ao_client: MagicMock, method: str, mock_error_throwing_agent: AgentRunner
):
    mock_ao_client_instance = MagicMock()
    mock_ao_client.return_value = mock_ao_client_instance

    # Initialize the AgentOps handler
    AgentOpsHandler.init()

    # Initiate a chat with the agent
    with pytest.raises(Exception):
        if method == "chat":
            mock_error_throwing_agent.chat(MOCK_AGENT_PROMPT)
        elif method == "achat":
            await mock_error_throwing_agent.achat(MOCK_AGENT_PROMPT)
        elif method == "stream_chat":
            mock_error_throwing_agent.stream_chat(MOCK_AGENT_PROMPT)
        else:
            await mock_error_throwing_agent.astream_chat(MOCK_AGENT_PROMPT)

    # We expect one LLMEvent for the prompt, then an ErrorEvent for the exception thrown
    calls_to_agentops = mock_ao_client_instance.record.call_args_list
    assert len(calls_to_agentops) == 2
    agent_start_chat_event: LLMEvent = calls_to_agentops[0][0][0]
    agent_error_event: ErrorEvent = calls_to_agentops[1][0][0]

    expected_agent_prompt = messages_to_prompt([ChatMessage(content=MOCK_AGENT_PROMPT)])
    assert agent_start_chat_event.prompt == expected_agent_prompt
    assert not agent_start_chat_event.completion
    assert agent_start_chat_event.model == MOCK_MODEL_NAME
    assert agent_error_event.details == MOCK_EXCEPTION_MESSAGE
