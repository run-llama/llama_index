from unittest.mock import MagicMock, patch

import openai
import pytest
from llama_index.agent import OpenAIAssistantAgent
from llama_index.agent.openai_assistant_agent import acall_function
from llama_index.llms import ChatMessage
from llama_index.tools import FunctionTool, ToolOutput
from openai.types.beta.threads.required_action_function_tool_call import Function


def test_from_existing_no_tools() -> None:
    assistant_id = "test-id"
    api_key = "test-api-key"
    mock_assistant = MagicMock()

    with patch.object(openai, "OpenAI") as mock_openai:
        mock_openai.return_value.beta.assistants.retrieve.return_value = mock_assistant
        agent = OpenAIAssistantAgent.from_existing(
            assistant_id=assistant_id,
            thread_id="your_thread_id",
            instructions_prefix="your_instructions_prefix",
            run_retrieve_sleep_time=0,
            api_key=api_key,
        )

    mock_openai.assert_called_once_with(api_key=api_key)
    mock_openai.return_value.beta.assistants.retrieve.assert_called_once_with(
        assistant_id
    )
    assert isinstance(agent, OpenAIAssistantAgent)


@pytest.fixture()
def add_tool() -> FunctionTool:
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer."""
        return a + b

    return FunctionTool.from_defaults(fn=add)


@pytest.fixture()
def add_function_call() -> Function:
    return Function(
        name="add",
        arguments='{"a": 1, "b": 2}',
    )


@pytest.mark.asyncio()
async def test_acall_function(
    add_tool: FunctionTool, add_function_call: Function
) -> None:
    tools = [add_tool]
    chat_message, tool_output = await acall_function(tools, add_function_call)  # type: ignore
    assert isinstance(chat_message, ChatMessage)
    assert isinstance(tool_output, ToolOutput)
    assert tool_output.raw_output == 3
