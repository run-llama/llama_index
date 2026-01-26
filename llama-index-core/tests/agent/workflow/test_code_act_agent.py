from dataclasses import dataclass, field
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Sequence


from llama_index.core.agent.workflow.codeact_agent import CodeActAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput, ToolCallResult
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import ChatMessage, LLMMetadata
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import ToolOutput
from llama_index.core.memory import BaseMemory


@dataclass
class MockStore:
    """A simple mock store that implements get/set."""

    data: dict = field(default_factory=dict)

    async def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        self.data[key] = value


@dataclass
class MockContext:
    """A mock context that captures events and provides a simple store."""

    events: list = field(default_factory=list)
    store: MockStore = field(default_factory=MockStore)

    def write_event_to_stream(self, event: Any) -> None:
        self.events.append(event)


def mock_context() -> MockContext:
    return MockContext()


@pytest.fixture()
def mock_llm():
    # Create a mock that inherits from FunctionCallingLLM
    class MockFunctionCallingLLM(FunctionCallingLLM):
        get_tool_calls_from_response: Any = MagicMock(return_value=[])

        def __init__(self) -> None:
            super().__init__()
            self._responses = []

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                is_function_calling_model=True,
            )

        async def astream_chat(self, *args, **kwargs):
            # Return an async generator that yields each response
            async def gen():
                for response in self._responses:
                    yield response

            return gen()

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

    return MockFunctionCallingLLM()


@pytest.fixture()
def mock_code_execute_fn():
    return lambda code: "Code executed"


@pytest.fixture()
def mock_memory():
    memory = AsyncMock(spec=BaseMemory)
    memory.aput = AsyncMock()
    return memory


@pytest.mark.asyncio
async def test_code_act_agent_basic_execution(
    mock_llm, mock_code_execute_fn, mock_memory
):
    # Setup mock response
    mock_response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            content="Let me calculate that for you.\n<execute>\nprint('Hello World')\n</execute>",
        ),
        delta="Let me calculate that for you.\n<execute>\nprint('Hello World')\n</execute>",
    )
    mock_llm._responses = [mock_response]  # Set the responses to be yielded

    # Create agent
    agent = CodeActAgent(
        code_execute_fn=mock_code_execute_fn,
        llm=mock_llm,
    )

    # Create context
    mock_ctx = mock_context()

    # Take step
    output = await agent.take_step(
        ctx=mock_ctx,
        llm_input=[ChatMessage(role="user", content="Say hello")],
        tools=[],
        memory=mock_memory,
    )

    # Verify output
    assert isinstance(output, AgentOutput)
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].tool_name == "execute"
    assert "print('Hello World')" in output.tool_calls[0].tool_kwargs["code"]


@pytest.mark.asyncio
async def test_code_act_agent_tool_handling(
    mock_llm, mock_code_execute_fn, mock_memory
):
    # Setup mock response
    mock_response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            content="Let me calculate that for you.\n<execute>\nresult = 2 + 2\nprint(result)\n</execute>",
        ),
        delta="Let me calculate that for you.\n<execute>\nresult = 2 + 2\nprint(result)\n</execute>",
    )
    mock_llm._responses = [mock_response]  # Set the responses to be yielded

    # Create agent
    agent = CodeActAgent(
        code_execute_fn=mock_code_execute_fn,
        llm=mock_llm,
    )

    # Create context
    mock_ctx = mock_context()

    # Take step
    output = await agent.take_step(
        ctx=mock_ctx,
        llm_input=[ChatMessage(role="user", content="What is 2 + 2?")],
        tools=[],
        memory=mock_memory,
    )

    # Handle tool results
    tool_results = [
        ToolCallResult(
            tool_id=output.tool_calls[0].tool_id,
            tool_name="execute",
            tool_kwargs={"code": "result = 2 + 2\nprint(result)\n"},
            tool_output=ToolOutput(
                content="4", tool_name="execute", raw_input={}, raw_output={}
            ),
            return_direct=False,
        )
    ]
    await agent.handle_tool_call_results(mock_ctx, tool_results, mock_memory)

    # Verify scratchpad was updated
    scratchpad = await mock_ctx.store.get("scratchpad")
    assert len(scratchpad) == 2  # User message and assistant response
    assert "4" in scratchpad[1].content  # Verify the result was added to scratchpad

    # Finalize
    final_output = await agent.finalize(mock_ctx, output, mock_memory)
    assert isinstance(final_output, AgentOutput)
    assert mock_memory.aput_messages.called  # Verify memory was updated


@pytest.mark.asyncio
async def test_code_act_agent_workflow_integration():
    """
    Integration test that runs the agent through the full workflow,
    verifying proper integration with the workflows library.
    """
    # Track call count to return different responses
    call_count = [0]

    def multi_response_generator(messages: Sequence[ChatMessage]) -> ChatMessage:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return code to execute
            return ChatMessage(
                role="assistant",
                content="Let me calculate that.\n<execute>\nresult = 2 + 2\nprint(result)\n</execute>",
            )
        else:
            # Second call: return final answer
            return ChatMessage(
                role="assistant",
                content="The answer is 4.",
            )

    mock_llm = MockFunctionCallingLLM(response_generator=multi_response_generator)

    # Create agent with a code_execute_fn that returns a result
    def execute_code(code: str) -> str:
        return "4"

    agent = CodeActAgent(
        code_execute_fn=execute_code,
        llm=mock_llm,
    )

    # Run the agent through the full workflow
    handler = agent.run(user_msg="What is 2 + 2?")
    result = await handler

    # Verify we got an AgentOutput
    assert isinstance(result, AgentOutput)
    assert result.response.content == "The answer is 4."
