from unittest.mock import MagicMock, patch

import pytest
from llama_index.agent.openai.openai_assistant_agent import (
    OpenAIAssistantAgent,
    acall_function,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool, ToolOutput

import openai
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


def test_from_new():
    name = "Math Tutor"
    instructions = (
        "You are a personal math tutor. Write and run code to answer math questions."
    )
    openai_tools = [{"type": "code_interpreter"}]
    instructions_prefix = (
        "Please address the user as Jane Doe. The user has a premium account."
    )
    run_retrieve_sleep_time = 0.5
    verbose = True
    api_key = "test-api-key"

    mock_assistant = MagicMock()
    with patch.object(openai, "OpenAI") as mock_openai:
        mock_openai.return_value.beta.assistants.create.return_value = mock_assistant
        agent = OpenAIAssistantAgent.from_new(
            name=name,
            instructions=instructions,
            openai_tools=openai_tools,
            instructions_prefix=instructions_prefix,
            run_retrieve_sleep_time=run_retrieve_sleep_time,
            verbose=verbose,
            api_key=api_key,
        )

    assert isinstance(agent, OpenAIAssistantAgent)
    assert agent.assistant == mock_assistant
    assert agent.client == mock_openai.return_value
    assert agent._instructions_prefix == instructions_prefix
    assert agent._run_retrieve_sleep_time == run_retrieve_sleep_time
    assert agent._verbose == verbose

    mock_openai.assert_called_once_with(api_key=api_key)
    mock_openai.return_value.beta.assistants.create.assert_called_once_with(
        model="gpt-4-1106-preview",
        name=name,
        instructions=instructions,
        tools=openai_tools,
    )


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
