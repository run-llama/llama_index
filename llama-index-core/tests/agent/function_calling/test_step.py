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


class MockBadFunctionCallingLLM(FunctionCallingLLM):
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
        **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="user", content=""))

    async def achat_with_tools(
        self,
        tools: List[BaseTool],
        user_msg: Union[str, ChatMessage, None] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any
    ) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role="user", content=""))

    def get_tool_calls_from_response(
        self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any
    ) -> List[ToolSelection]:
        return [NONEXISTENT_TOOL_SELECTION]


@pytest.fixture()
def missing_function_agent_worker() -> FunctionCallingAgentWorker:
    llm = MockBadFunctionCallingLLM()
    tool = FunctionTool(lambda x: x, ToolMetadata("", "Tool 1"))
    return FunctionCallingAgentWorker([tool], llm, [])


def test_get_function_by_name_finds_existing_tool() -> None:
    first_tool = FunctionTool(lambda x: x, ToolMetadata("", "Tool 1"))
    second_tool = FunctionTool(lambda x: x, ToolMetadata("", "Tool 2"))
    tools = [first_tool, second_tool]
    assert get_function_by_name(tools, first_tool.metadata.name) == first_tool
    assert get_function_by_name(tools, first_tool.metadata.name) == first_tool


def test_get_function_by_name_returns_none_for_nonexistent_tool() -> None:
    tool = FunctionTool(lambda x: x, ToolMetadata("", "Tool 1"))
    assert get_function_by_name([tool], "Name of a Nonexistent Tool") is None
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


@pytest.mark.asyncio()
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
