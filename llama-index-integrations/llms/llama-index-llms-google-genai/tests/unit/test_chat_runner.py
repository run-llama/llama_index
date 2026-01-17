import pytest
from llama_index.llms.google_genai.orchestration.chat_session import (
    ChatSessionRunner,
    PreparedChat,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.base.llms.types import TextBlock
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
async def test_prepare_merges_adjacent_non_tool_same_role_safely(
    mock_genai_client,
    mock_file_manager,
    mock_message_converter,
    mock_response_converter,
) -> None:
    """
    Adjacent same-role non-tool contents should merge into a single Content.

    This matches the old `merge_neighboring_same_role_messages` behavior while
    respecting Gemini thought signature/function calling constraints.
    """
    mock_message_converter.to_gemini_content.side_effect = [
        (types.Content(role="user", parts=[types.Part(text="Hello")]), []),
        (types.Content(role="user", parts=[types.Part(text="World")]), []),
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
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.USER, content="World"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Done"),
    ]

    prepared = await runner.prepare(messages=messages)
    history = prepared.chat_kwargs["history"]

    # The two user messages should be merged into one history Content.
    assert len(history) == 1
    assert history[0].role == "user"
    assert [p.text for p in (history[0].parts or [])] == ["Hello", "World"]

    # Next msg is the final assistant message.
    assert isinstance(prepared.next_msg, types.Content)
    assert prepared.next_msg.role == "model"
    assert prepared.next_msg.parts[0].text == "Done"


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


@pytest.mark.asyncio
async def test_prepare_build_history_contents_existing_uploaded_file_names(
    mock_genai_client,
    mock_file_manager,
    mock_response_converter,
) -> None:
    """
    Ensure _build_history_contents returns the uploaded_file_names list.

    This is a regression test for the tuple returned by:
        history_contents, uploaded_file_names = await self._build_history_contents(msgs)

    In fileapi/hybrid modes, this list is later used for cleanup.
    """
    # Use real converter to exercise orchestration end-to-end.
    real_converter = MessageConverter(file_manager=mock_file_manager)

    # Build a fake file upload part without IO.
    async def _fake_create_part(file_buffer, mime_type: str):
        return (
            types.Part.from_uri(file_uri="gs://bucket/file.png", mime_type=mime_type),
            "files/file-123",
        )

    mock_file_manager.create_part.side_effect = _fake_create_part

    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="models/gemini-foo",
        file_manager=mock_file_manager,
        message_converter=real_converter,
        response_converter=mock_response_converter,
    )

    # Use a real ImageBlock but override resolver so we don't fetch anything.
    from llama_index.core.base.llms.types import ImageBlock, TextBlock

    class FakeImageBlock(ImageBlock):
        def resolve_image(self, as_base64: bool = False):
            # The FileManager mock ignores the buffer content.
            return b"fake"

    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text="what is this?"), FakeImageBlock(image=b"fake")],
        ),
        ChatMessage(role=MessageRole.ASSISTANT, content="Looks like a logo"),
    ]

    prepared = await runner.prepare(messages=messages)

    assert prepared.uploaded_file_names == ["files/file-123"]
    # Also verify the content made it into next_msg/history correctly.
    assert isinstance(prepared.next_msg, types.Content)
    assert prepared.next_msg.role == "model"


@pytest.mark.asyncio
async def test_build_history_contents_aggregates_multiple_uploaded_file_names(
    mock_genai_client,
    mock_file_manager,
    mock_response_converter,
) -> None:
    """Ensure uploaded_file_names are aggregated across multiple messages."""
    real_converter = MessageConverter(file_manager=mock_file_manager)

    async def _fake_create_part(file_buffer, mime_type: str):
        # Return a different name per call.
        call_idx = mock_file_manager.create_part.call_count
        return (
            types.Part.from_uri(
                file_uri=f"gs://bucket/file-{call_idx}.png", mime_type=mime_type
            ),
            f"files/file-{call_idx}",
        )

    mock_file_manager.create_part.side_effect = _fake_create_part

    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="models/gemini-foo",
        file_manager=mock_file_manager,
        message_converter=real_converter,
        response_converter=mock_response_converter,
    )

    from llama_index.core.base.llms.types import ImageBlock, TextBlock

    class FakeImageBlock(ImageBlock):
        def resolve_image(self, as_base64: bool = False):
            return b"fake"

    # Two user messages with images -> two uploads.
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text="img1"), FakeImageBlock(image=b"fake")],
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=[TextBlock(text="img2"), FakeImageBlock(image=b"fake")],
        ),
    ]

    history, uploaded = await runner._build_history_contents(messages)

    assert len(history) == 2
    assert uploaded == ["files/file-1", "files/file-2"]


def test_chat_message_content_joins_text_blocks_with_newlines() -> None:
    r"""
    ChatMessage.content joins multiple TextBlocks using '\n'.

    This matters when rendering messages that intentionally keep a 1:1 mapping
    between Gemini Parts and LlamaIndex blocks (e.g., for thought_signature
    positional correctness).
    """
    msg = ChatMessage(
        role=MessageRole.ASSISTANT,
        blocks=[TextBlock(text="Hello"), TextBlock(text="World")],
    )

    assert msg.content == "Hello\nWorld"

    # Single TextBlock: no extra newline.
    assert (
        ChatMessage(role=MessageRole.ASSISTANT, blocks=[TextBlock(text="Hi")]).content
        == "Hi"
    )

    # No TextBlocks: None.
    assert ChatMessage(role=MessageRole.ASSISTANT, blocks=[]).content is None


def _content(*, role: str, parts: list[types.Part]) -> types.Content:
    return types.Content(role=role, parts=parts)


def _text_part(text: str, *, thought_signature: str | None = None) -> types.Part:
    part = types.Part.from_text(text=text)
    if thought_signature is not None:
        part.thought_signature = thought_signature
    return part


def _fc_part(
    *, name: str, args: dict, thought_signature: str | None = None
) -> types.Part:
    part = types.Part.from_function_call(name=name, args=args)
    if thought_signature is not None:
        part.thought_signature = thought_signature
    return part


def _fr_part(*, name: str, response: dict) -> types.Part:
    return types.Part.from_function_response(name=name, response=response)


@pytest.mark.asyncio
async def test_prepare_rejects_content_that_mixes_function_call_and_function_response(
    mock_genai_client,
    mock_file_manager,
    mock_response_converter,
) -> None:
    """Gemini 3: A single Content must not mix FC and FR parts."""
    mock_message_converter = pytest.MonkeyPatch()

    converter = MessageConverter(file_manager=mock_file_manager)
    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="models/gemini-foo",
        file_manager=mock_file_manager,
        message_converter=converter,
        response_converter=mock_response_converter,
    )

    # Build a message history that results in a single *Gemini* Content with both
    # function_call and function_response parts (invalid).
    invalid_content = _content(
        role="user",
        parts=[
            _fc_part(name="t1", args={"x": 1}),
            _fr_part(name="t1", response={"ok": True}),
        ],
    )

    # Patch the runner's internal history builder to bypass MessageConverter.
    async def _fake_build_history_contents(_msgs):
        return [invalid_content], []

    runner._build_history_contents = _fake_build_history_contents  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="functionCall and functionResponse"):
        await runner.prepare(messages=[ChatMessage(role=MessageRole.USER, content="x")])


@pytest.mark.asyncio
async def test_normalize_history_does_not_merge_contents_with_thought_signatures(
    mock_genai_client,
    mock_file_manager,
    mock_message_converter,
    mock_response_converter,
) -> None:
    """Ensure we never merge across thought_signature boundaries."""
    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="models/gemini-foo",
        file_manager=mock_file_manager,
        message_converter=mock_message_converter,
        response_converter=mock_response_converter,
    )

    c1 = _content(role="model", parts=[_text_part("a", thought_signature="sigA")])
    c2 = _content(role="model", parts=[_text_part("b")])

    normalized = runner._normalize_history([c1, c2])
    assert len(normalized) == 2


@pytest.mark.asyncio
async def test_normalize_history_does_not_merge_contents_with_function_calls(
    mock_genai_client,
    mock_file_manager,
    mock_message_converter,
    mock_response_converter,
) -> None:
    """Ensure we never merge adjacent model contents when either has a function_call."""
    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="models/gemini-foo",
        file_manager=mock_file_manager,
        message_converter=mock_message_converter,
        response_converter=mock_response_converter,
    )

    c1 = _content(role="model", parts=[_fc_part(name="t1", args={})])
    c2 = _content(role="model", parts=[_text_part("after")])

    normalized = runner._normalize_history([c1, c2])
    assert len(normalized) == 2


@pytest.mark.asyncio
async def test_normalize_history_merges_adjacent_tool_responses_only(
    mock_genai_client,
    mock_file_manager,
    mock_message_converter,
    mock_response_converter,
) -> None:
    """Adjacent tool-response contents should merge into one user Content."""
    runner = ChatSessionRunner(
        client=mock_genai_client,
        model="models/gemini-foo",
        file_manager=mock_file_manager,
        message_converter=mock_message_converter,
        response_converter=mock_response_converter,
    )

    u1 = _content(role="user", parts=[_fr_part(name="t1", response={"r": 1})])
    u2 = _content(role="user", parts=[_fr_part(name="t2", response={"r": 2})])
    normalized = runner._normalize_history([u1, u2])

    assert len(normalized) == 1
    assert normalized[0].role == "user"
    assert len(normalized[0].parts) == 2
    assert normalized[0].parts[0].function_response.name == "t1"
    assert normalized[0].parts[1].function_response.name == "t2"
