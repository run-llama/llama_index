import uuid
import pytest
from typing import Any, AsyncGenerator, Coroutine, List, Optional, Sequence, Union, Dict
from llama_index.core.agent.function_calling.step import (
    FunctionCallingAgentWorker,
    build_missing_tool_message,
    build_missing_tool_output,
    get_function_by_name,
)
from llama_index.core.base.agent.types import Task, TaskStepOutput
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    LLMMetadata,
)
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.tools.types import BaseTool, ToolMetadata


NONEXISTENT_TOOL_ID = "NonexistentToolID"
NONEXISTENT_TOOL_NAME = "NonexistentToolName"
NONEXISTENT_TOOL_ERR_MSG = build_missing_tool_message(NONEXISTENT_TOOL_NAME)
NONEXISTENT_TOOL_SELECTION = ToolSelection(
    tool_id=NONEXISTENT_TOOL_ID,
    tool_name=NONEXISTENT_TOOL_NAME,
    tool_kwargs={},
)
NONEXISTENT_TOOL_OUTPUT = build_missing_tool_output(NONEXISTENT_TOOL_SELECTION)

TOOL_1_NAME = "Tool 1"
TOOL_1 = FunctionTool(lambda: None, ToolMetadata("", TOOL_1_NAME))
TOOL_1_SELECTION = ToolSelection(
    tool_id=TOOL_1_NAME,
    tool_name=TOOL_1_NAME,
    tool_kwargs={},
)

TOOL_2_NAME = "Tool 2"
TOOL_2 = FunctionTool(lambda x: None, ToolMetadata("", TOOL_2_NAME))


class MockFunctionCallingLLM(FunctionCallingLLM):
    use_nonexistent_tool: bool = False

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
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare chat with tools."""
        return {}

    def chat_with_tools(
        self,
        tools: List[BaseTool],
        user_msg: Union[str, ChatMessage, None] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="user", content=""))

    async def achat_with_tools(
        self,
        tools: List[BaseTool],
        user_msg: Union[str, ChatMessage, None] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="user", content=""))

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> List[ToolSelection]:
        return [
            (
                NONEXISTENT_TOOL_SELECTION
                if self.use_nonexistent_tool
                else TOOL_1_SELECTION
            )
        ]


@pytest.fixture()
def agent_worker() -> FunctionCallingAgentWorker:
    llm = MockFunctionCallingLLM(use_nonexistent_tool=False)
    return FunctionCallingAgentWorker([TOOL_1], llm, [])


@pytest.fixture()
def missing_function_agent_worker() -> FunctionCallingAgentWorker:
    llm = MockFunctionCallingLLM(use_nonexistent_tool=True)
    return FunctionCallingAgentWorker([TOOL_1], llm, [])


def test_get_function_by_name_finds_existing_tool() -> None:
    tools = [TOOL_1, TOOL_2]
    assert get_function_by_name(tools, TOOL_1.metadata.name) == TOOL_1
    assert get_function_by_name(tools, TOOL_1.metadata.name) == TOOL_1


def test_get_function_by_name_returns_none_for_nonexistent_tool() -> None:
    assert get_function_by_name([TOOL_1], "Name of a Nonexistent Tool") is None
    assert get_function_by_name([], "Name of a Nonexistent Tool") is None


def test_run_step_returns_message_if_function_not_found(
    missing_function_agent_worker: FunctionCallingAgentWorker,
) -> None:
    task = Task(input="", memory=ChatMemoryBuffer.from_defaults(), extra_state={})
    step = missing_function_agent_worker.initialize_step(task)
    output: TaskStepOutput = missing_function_agent_worker.run_step(step, task)
    output_chat_response: AgentChatResponse = output.output

    assert not output.is_last
    assert len(output.next_steps) == 1
    assert len(output_chat_response.sources) == 1
    assert output_chat_response.sources[0] == NONEXISTENT_TOOL_OUTPUT


@pytest.mark.asyncio
async def test_arun_step_returns_message_if_function_not_found(
    missing_function_agent_worker: FunctionCallingAgentWorker,
) -> None:
    task = Task(input="", memory=ChatMemoryBuffer.from_defaults(), extra_state={})
    step = missing_function_agent_worker.initialize_step(task)
    output: TaskStepOutput = await missing_function_agent_worker.arun_step(step, task)
    output_chat_response: AgentChatResponse = output.output

    assert not output.is_last
    assert len(output.next_steps) == 1
    assert len(output_chat_response.sources) == 1
    assert output_chat_response.sources[0] == NONEXISTENT_TOOL_OUTPUT


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["run_step", "arun_step"])
async def test_run_step_returns_correct_sources_history(
    method: str,
    agent_worker: FunctionCallingAgentWorker,
) -> None:
    num_steps = 4
    task = Task(input="", memory=ChatMemoryBuffer.from_defaults(), extra_state={})
    step_outputs: List[TaskStepOutput] = []

    # Create steps
    steps = [agent_worker.initialize_step(task)]
    for step_idx in range(num_steps - 1):
        steps.append(
            steps[-1].get_next_step(
                step_id=str(uuid.uuid4()),
                input=None,
            )
        )

    # Run each step, invoking a single tool call each time
    for step_idx in range(num_steps):
        step_outputs.append(
            agent_worker.run_step(steps[step_idx], task)
            if method == "run_step"
            else await agent_worker.arun_step(steps[step_idx], task)
        )

    # Ensure that each step only has one source for its one tool call
    for step_idx in range(num_steps):
        assert len(step_outputs[step_idx].output.sources) == 1
