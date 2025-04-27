import pytest
import uuid
import llama_index.core.instrumentation as instrument
from agentops import LLMEvent
from typing import (
    Any,
    Generator,
    Sequence,
)
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
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.llm import LLM, ToolSelection
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import ToolMetadata
from llama_index.callbacks.agentops import AgentOpsHandler
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


def test_class():
    names_of_base_classes = [b.__name__ for b in AgentOpsHandler.__mro__]
    assert AgentOpsHandler.__name__ in names_of_base_classes


@pytest.mark.asyncio()
@pytest.mark.parametrize("method", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize("agent_runner_fixture", ["mock_agent"])
@patch("llama_index.callbacks.agentops.base.AOClient")
async def test_agentops_event_handler_emits_llmevents(
    mock_ao_client: MagicMock, method: str, agent_runner_fixture: str, request
):
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

    # We expect one LLMEvent for the prompt and response
    calls_to_agentops = mock_ao_client_instance.record.call_args_list
    assert len(calls_to_agentops) == 1
    agent_event: LLMEvent = calls_to_agentops[0][0][0]

    expected_agent_prompt = [{"content": MOCK_AGENT_PROMPT, "role": MessageRole.USER}]
    expected_agent_completion = {
        "content": MOCK_AGENT_RESPONSE,
        "role": MessageRole.USER,
    }
    assert agent_event.prompt == expected_agent_prompt
    assert agent_event.model == MOCK_MODEL_NAME
    assert agent_event.completion == expected_agent_completion
