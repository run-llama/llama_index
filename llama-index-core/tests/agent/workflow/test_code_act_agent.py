import pytest
from unittest.mock import MagicMock
from typing import Any

from llama_index.core.agent.workflow.codeact_agent import CodeActAgent
from llama_index.core.agent.workflow.workflow_events import AgentOutput, ToolCall
from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.llms import ChatMessage, LLMMetadata
from llama_index.core.llms.function_calling import FunctionCallingLLM


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


@pytest.mark.asyncio
async def test_code_act_agent_basic_execution(mock_llm, mock_code_execute_fn):
    """Test that CodeActAgent correctly parses execute blocks and creates tool calls."""
    # Setup mock response with execute block
    mock_response = ChatResponse(
        message=ChatMessage(
            role="assistant",
            content="Let me calculate that for you.\n<execute>\nprint('Hello World')\n</execute>",
        ),
        delta="Let me calculate that for you.\n<execute>\nprint('Hello World')\n</execute>",
    )
    mock_llm._responses = [mock_response]

    # Create agent
    agent = CodeActAgent(
        code_execute_fn=mock_code_execute_fn,
        llm=mock_llm,
    )

    # Run the agent through the proper workflow flow
    handler = agent.run(user_msg="Say hello")

    # Collect events to find tool calls
    tool_calls_found = []
    async for event in handler.stream_events():
        if isinstance(event, AgentOutput) and event.tool_calls:
            tool_calls_found.extend(event.tool_calls)

    # The agent should have identified the execute block
    assert len(tool_calls_found) >= 1
    execute_calls = [tc for tc in tool_calls_found if tc.tool_name == "execute"]
    assert len(execute_calls) >= 1
    assert "print('Hello World')" in execute_calls[0].tool_kwargs["code"]


@pytest.mark.asyncio
async def test_code_act_agent_tool_handling(mock_llm, mock_code_execute_fn):
    """Test that CodeActAgent correctly handles tool execution and results."""
    # Setup mock responses - first with execute block, then final response without execute block
    mock_responses = [
        ChatResponse(
            message=ChatMessage(
                role="assistant",
                content="Let me calculate that for you.\n<execute>\nresult = 2 + 2\nprint(result)\n</execute>",
            ),
            delta="Let me calculate that for you.\n<execute>\nresult = 2 + 2\nprint(result)\n</execute>",
        ),
    ]
    # Provide enough final responses for the agent to complete (no execute block = done)
    for _ in range(25):  # Extra responses to ensure agent has enough to complete
        mock_responses.append(
            ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content="The result is 4.",
                ),
                delta="The result is 4.",
            )
        )
    mock_llm._responses = mock_responses

    # Create agent with a code execute function that returns "4"
    def execute_fn(code: str) -> str:
        return "4"

    agent = CodeActAgent(
        code_execute_fn=execute_fn,
        llm=mock_llm,
    )

    # Run the agent
    handler = agent.run(user_msg="What is 2 + 2?")

    # Consume events and collect outputs
    outputs = []
    async for event in handler.stream_events():
        if isinstance(event, AgentOutput):
            outputs.append(event)

    # Get final result
    result = await handler

    # Verify we got a result
    assert isinstance(result, AgentOutput)
    # Verify we saw at least one tool call (the execute)
    assert any(o.tool_calls for o in outputs)
