import pytest
from llama_index.llms.google_genai.orchestration.chat_session import (
    ChatSessionRunner,
    PreparedChat,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.types import ToolCallBlock
import google.genai.types as types
from llama_index.llms.google_genai.conversion.messages import MessageConverter


@pytest.mark.asyncio
async def test_chat_runner_prepare(
    mock_genai_client,
    mock_file_manager,
    mock_message_converter,
    mock_response_converter,
) -> None:
    """
    Test that ChatSessionRunner.prepare correctly coordinates converters
    and separates the last message for the API call.
    """
    mock_message_converter.to_gemini_content.side_effect = [
        (types.Content(role="user", parts=[types.Part(text="Hello")]), []),
        (types.Content(role="model", parts=[types.Part(text="Hi")]), []),
        (types.Content(role="user", parts=[types.Part(text="How are you?")]), []),
    ]

    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="gemini-pro",
        file_manager=mock_file_manager,
        message_converter=mock_message_converter,
        response_converter=mock_response_converter,
    )

    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi"),
        ChatMessage(role=MessageRole.USER, content="How are you?"),
    ]

    prepared = await runner.prepare(messages=messages)

    # Assert
    assert isinstance(prepared, PreparedChat)

    # The last message should be popped for the API call
    assert prepared.next_msg.parts[0].text == "How are you?"

    # The history should contain the previous messages
    assert len(prepared.chat_kwargs["history"]) == 2
    assert prepared.chat_kwargs["history"][0].parts[0].text == "Hello"
    assert prepared.chat_kwargs["model"] == "gemini-pro"


@pytest.mark.asyncio
async def test_prepare_with_system_message(
    mock_genai_client,
    mock_file_manager,
    mock_message_converter,
    mock_response_converter,
) -> None:
    """Test that a system message is extracted to system_instruction."""
    mock_message_converter.to_gemini_content.side_effect = [
        (types.Content(role="user", parts=[types.Part(text="Hello")]), []),
    ]

    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="gemini-pro",
        file_manager=mock_file_manager,
        message_converter=mock_message_converter,
        response_converter=mock_response_converter,
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="Be helpful"),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]

    prepared = await runner.prepare(messages=messages)

    # System message should be popped and put into config
    assert prepared.chat_kwargs["config"].system_instruction == "Be helpful"
    # Only user message remains in history processing (and becomes next_msg)
    assert prepared.next_msg.parts[0].text == "Hello"


@pytest.mark.asyncio
async def test_prepare_tool_history_merging(
    mock_genai_client,
    mock_file_manager,
    mock_message_converter,
    mock_response_converter,
) -> None:
    """Test merging of sequential tool outputs."""
    # Simulate: User -> Model (Call) -> User (Result 1) -> User (Result 2) -> Model (Next)
    # The two User (Result) messages should be merged into one Content with 2 parts

    # We just mock the returns of to_gemini_content
    mock_message_converter.to_gemini_content.side_effect = [
        (types.Content(role="user", parts=[types.Part(text="Start")]), []),
        (
            types.Content(
                role="model",
                parts=[
                    types.Part(function_call=types.FunctionCall(name="t1", args={}))
                ],
            ),
            [],
        ),
        (
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="t1", response={"r": 1}
                        )
                    )
                ],
            ),
            [],
        ),
        (
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="t2", response={"r": 2}
                        )
                    )
                ],
            ),
            [],
        ),
        (types.Content(role="model", parts=[types.Part(text="Done")]), []),
    ]

    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="gemini-pro",
        file_manager=mock_file_manager,
        message_converter=mock_message_converter,
        response_converter=mock_response_converter,
    )

    messages = [
        ChatMessage(role=MessageRole.USER, content="Start"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[ToolCallBlock(tool_name="t1", tool_kwargs={})],
        ),
        ChatMessage(
            role=MessageRole.TOOL, content="1", additional_kwargs={"tool_call_id": "t1"}
        ),
        ChatMessage(
            role=MessageRole.TOOL, content="2", additional_kwargs={"tool_call_id": "t2"}
        ),
        ChatMessage(role=MessageRole.ASSISTANT, content="Done"),
    ]

    prepared = await runner.prepare(messages=messages)

    history = prepared.chat_kwargs["history"]
    # Expected: User(Start), Model(Call), User(Resp1 + Resp2)
    assert len(history) == 3
    assert history[2].role == "user"
    assert len(history[2].parts) == 2
    assert history[2].parts[0].function_response.name == "t1"
    assert history[2].parts[1].function_response.name == "t2"


@pytest.mark.asyncio
async def test_prepare_more_than_2_tool_calls(
    mock_genai_client,
    mock_file_manager,
    mock_message_converter,
    mock_response_converter,
) -> None:
    """
    Ensure assistant tool_calls are normalized and tool responses are merged.

    Ensures that:
    - assistant additional_kwargs tool_calls are normalized into ToolCallBlocks
    - tool responses are merged into a single user Content with multiple FR parts
    - ordering is preserved and next_msg is final assistant content
    """
    # Use the real converter for correctness; we are testing orchestration and
    # normalization behavior. `mock_file_manager` avoids any IO.
    real_converter = MessageConverter(file_manager=mock_file_manager)

    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="models/gemini-foo",
        file_manager=mock_file_manager,
        message_converter=real_converter,
        response_converter=mock_response_converter,
    )

    test_messages = [
        ChatMessage(content="Find me a puppy.", role=MessageRole.USER),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                # Thinking block becomes a thought=True text part.
                # Kept as-is in the mocked converter output above.
            ],
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[ToolCallBlock(tool_name="get_available_tools", tool_kwargs={})],
        ),
        ChatMessage(
            content="Let me search for puppies.",
            role=MessageRole.ASSISTANT,
            additional_kwargs={
                "tool_calls": [
                    {"name": "tool_1", "args": {}},
                    {"name": "tool_2", "args": {}},
                    {"name": "tool_3", "args": {}},
                ]
            },
        ),
        ChatMessage(
            content="Tool 1 Response",
            role=MessageRole.TOOL,
            additional_kwargs={"tool_call_id": "tool_1"},
        ),
        ChatMessage(
            content="Tool 2 Response",
            role=MessageRole.TOOL,
            additional_kwargs={"tool_call_id": "tool_2"},
        ),
        ChatMessage(
            content="Tool 3 Response",
            role=MessageRole.TOOL,
            additional_kwargs={"tool_call_id": "tool_3"},
        ),
        ChatMessage(content="Here is a list of puppies.", role=MessageRole.ASSISTANT),
    ]

    prepared = await runner.prepare(messages=test_messages)

    assert prepared.chat_kwargs["model"] == "models/gemini-foo"
    assert isinstance(prepared.chat_kwargs["config"], types.GenerateContentConfig)

    # Next msg is the last assistant output.
    assert isinstance(prepared.next_msg, types.Content)
    assert prepared.next_msg.role == "model"
    assert prepared.next_msg.parts[0].text == "Here is a list of puppies."

    history = prepared.chat_kwargs["history"]
    # After normalization/merging, history should include:
    # 1) user prompt
    # 2) model: get_available_tools tool call
    # 3) model: "Let me search..." + tool_1/tool_2/tool_3 function calls
    # 4) user: merged tool responses (FR parts)
    assert len(history) == 4

    # user prompt
    assert history[0].role == "user"
    assert history[0].parts[0].text == "Find me a puppy."

    # assistant: get_available_tools tool call
    assert history[1].role == "model"
    tool_call_names_step1 = [
        p.function_call.name for p in (history[1].parts or []) if p.function_call
    ]
    assert tool_call_names_step1 == ["get_available_tools"]

    # assistant: "Let me search..." + tool_1/tool_2/tool_3 function calls
    assert history[2].role == "model"
    tool_call_names_step2 = [
        p.function_call.name for p in (history[2].parts or []) if p.function_call
    ]
    assert tool_call_names_step2 == ["tool_1", "tool_2", "tool_3"]

    # merged tool responses
    assert history[3].role == "user"
    fr_names = [
        p.function_response.name
        for p in (history[3].parts or [])
        if p.function_response
    ]
    assert fr_names == ["tool_1", "tool_2", "tool_3"]
