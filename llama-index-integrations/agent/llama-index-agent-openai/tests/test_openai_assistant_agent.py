from typing import List
from unittest.mock import MagicMock, patch

from llama_index.core.agent.function_calling.step import (
    build_missing_tool_message,
    build_missing_tool_output,
)
from llama_index.core.llms.llm import ToolSelection
import pytest
from llama_index.agent.openai.openai_assistant_agent import (
    OpenAIAssistantAgent,
    acall_function,
    call_function,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool, ToolOutput

import openai
from openai.types.beta.threads import Run
from openai.types.beta.threads.run import (
    RequiredAction,
    RequiredActionSubmitToolOutputs,
    RequiredActionFunctionToolCall,
)
from openai.types.beta.threads.required_action_function_tool_call import Function

NONEXISTENT_TOOL_ID = "NonexistentToolID"
NONEXISTENT_TOOL_NAME = "NonexistentToolName"
NONEXISTENT_TOOL_ERR_MSG = build_missing_tool_message(NONEXISTENT_TOOL_NAME)
NONEXISTENT_TOOL_SELECTION = ToolSelection(
    tool_id=NONEXISTENT_TOOL_ID,
    tool_name=NONEXISTENT_TOOL_NAME,
    tool_kwargs={},
)
NONEXISTENT_TOOL_OUTPUT = build_missing_tool_output(NONEXISTENT_TOOL_SELECTION)
NONEXISTENT_FUNCTION = Function(name=NONEXISTENT_TOOL_NAME, arguments="{}")


def mock_nonexistent_function_run(completed: bool) -> Run:
    return Run(
        id="",
        assistant_id="",
        created_at=0,
        instructions="",
        model="",
        object="thread.run",
        status="completed" if completed else "requires_action",
        thread_id="",
        tools=[],
        required_action=RequiredAction(
            type="submit_tool_outputs",
            submit_tool_outputs=RequiredActionSubmitToolOutputs(
                tool_calls=[
                    RequiredActionFunctionToolCall(
                        id="", function=NONEXISTENT_FUNCTION, type="function"
                    )
                ]
            ),
        ),
        parallel_tool_calls=False,
    )


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
        top_p=None,
        temperature=None,
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


@pytest.fixture()
def nonexistent_function() -> Function:
    return NONEXISTENT_FUNCTION


@pytest.fixture()
def requires_action_nonexistent_function_run() -> Run:
    return mock_nonexistent_function_run(False)


@pytest.fixture()
def completed_nonexistent_function_run() -> Run:
    return mock_nonexistent_function_run(True)


def test_call_function(add_tool: FunctionTool, add_function_call: Function) -> None:
    tools = [add_tool]
    chat_message, tool_output = call_function(tools, add_function_call)  # type: ignore
    assert isinstance(chat_message, ChatMessage)
    assert isinstance(tool_output, ToolOutput)
    assert tool_output.raw_output == 3


def test_call_function_returns_message_if_tool_not_found(
    nonexistent_function: Function,
) -> None:
    chat_message, tool_output = call_function([], nonexistent_function)  # type: ignore
    assert tool_output == NONEXISTENT_TOOL_OUTPUT
    assert chat_message.content == NONEXISTENT_TOOL_ERR_MSG


@pytest.mark.asyncio
async def test_acall_function(
    add_tool: FunctionTool, add_function_call: Function
) -> None:
    tools = [add_tool]
    chat_message, tool_output = await acall_function(tools, add_function_call)  # type: ignore
    assert isinstance(chat_message, ChatMessage)
    assert isinstance(tool_output, ToolOutput)
    assert tool_output.raw_output == 3


@pytest.mark.asyncio
async def test_acall_function_returns_message_if_tool_not_found(
    nonexistent_function: Function,
) -> None:
    chat_message, tool_output = await acall_function([], nonexistent_function)  # type: ignore
    assert tool_output == NONEXISTENT_TOOL_OUTPUT
    assert chat_message.content == NONEXISTENT_TOOL_ERR_MSG


def test_run_assistant_returns_message_if_tool_not_found(
    requires_action_nonexistent_function_run: Run,
    completed_nonexistent_function_run: Run,
) -> None:
    with patch.object(openai, "OpenAI") as mock_openai:
        mock_openai.return_value.beta.assistants.retrieve.return_value = MagicMock()
        mock_openai.return_value.beta.threads.runs.create.return_value = (
            requires_action_nonexistent_function_run
        )
        mock_openai.return_value.beta.threads.runs.retrieve.side_effect = [
            requires_action_nonexistent_function_run,
            completed_nonexistent_function_run,
        ]
        mock_openai.return_value.beta.threads.runs.submit_tool_outputs = MagicMock()

        agent = OpenAIAssistantAgent.from_existing(
            assistant_id="",
            thread_id="",
            instructions_prefix="your_instructions_prefix",
            run_retrieve_sleep_time=0,
            api_key="",
            tools=[],
        )
        run, sources_dict = agent.run_assistant()
        tool_outputs: List[ToolOutput] = sources_dict["sources"]
        assert len(tool_outputs) == 1
        assert tool_outputs[0] == NONEXISTENT_TOOL_OUTPUT


@pytest.mark.asyncio
async def test_arun_assistant_returns_message_if_tool_not_found(
    requires_action_nonexistent_function_run: Run,
    completed_nonexistent_function_run: Run,
) -> None:
    with patch.object(openai, "OpenAI") as mock_openai:
        mock_openai.return_value.beta.assistants.retrieve.return_value = MagicMock()
        mock_openai.return_value.beta.threads.runs.create.return_value = (
            requires_action_nonexistent_function_run
        )
        mock_openai.return_value.beta.threads.runs.retrieve.side_effect = [
            requires_action_nonexistent_function_run,
            completed_nonexistent_function_run,
        ]
        mock_openai.return_value.beta.threads.runs.submit_tool_outputs = MagicMock()

        agent = OpenAIAssistantAgent.from_existing(
            assistant_id="",
            thread_id="",
            instructions_prefix="your_instructions_prefix",
            run_retrieve_sleep_time=0,
            api_key="",
            tools=[],
        )
        run, sources_dict = await agent.arun_assistant()
        tool_outputs: List[ToolOutput] = sources_dict["sources"]
        assert len(tool_outputs) == 1
        assert tool_outputs[0] == NONEXISTENT_TOOL_OUTPUT


def test_add_message_calls_agent_create_with_default_tool():
    default_tool = [{"type": "file_search"}]
    thread_id = "test_thread_id"
    message_content = "Hello, this is a test message."
    fake_message_response = {
        "thread_id": thread_id,
        "role": "user",
        "content": message_content,
    }
    file_ids = ["file123", "file456"]

    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value.beta.threads.messages.create.return_value = (
            fake_message_response
        )

        agent = OpenAIAssistantAgent.from_existing(
            assistant_id="",
            thread_id=thread_id,
            instructions_prefix="your_instructions_prefix",
            run_retrieve_sleep_time=0,
            api_key="",
            tools=[],
        )

        result = agent.add_message(message=message_content, file_ids=file_ids, tools=[])

        attachments = [
            {"file_id": file_id, "tools": default_tool} for file_id in file_ids
        ]
        mock_openai.return_value.beta.threads.messages.create.assert_called_once_with(
            thread_id=thread_id,
            role="user",
            content=message_content,
            attachments=attachments,
        )
        assert result == fake_message_response
