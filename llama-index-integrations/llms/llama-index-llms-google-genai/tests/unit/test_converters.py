import pytest
from unittest.mock import MagicMock
from llama_index.llms.google_genai.conversion.messages import MessageConverter
from llama_index.llms.google_genai.conversion.responses import (
    ResponseConverter,
    GeminiResponseParseState,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
    ThinkingBlock,
)
from google.genai import types


@pytest.mark.asyncio
async def test_message_converter_text_only(mock_file_manager) -> None:
    """Test converting a simple text message without file operations."""
    converter = MessageConverter(file_manager=mock_file_manager)
    msg = ChatMessage(role=MessageRole.USER, content="Hello")
    content, file_names = await converter.to_gemini_content(msg)

    assert isinstance(content, types.Content)
    assert content.role == "user"
    assert len(content.parts) == 1
    assert content.parts[0].text == "Hello"
    assert file_names == []


@pytest.mark.asyncio
async def test_message_converter_with_image(mock_file_manager) -> None:
    """Test converting a message with an image block, verifying file manager interaction."""
    converter = MessageConverter(file_manager=mock_file_manager)

    mock_file_manager.create_part.return_value = (
        types.Part.from_text(text="img"),
        None,
    )

    msg = ChatMessage(
        role=MessageRole.USER,
        content=[ImageBlock(image=b"123", image_mimetype="image/png")],
    )

    content, file_names = await converter.to_gemini_content(msg)

    assert content is not None
    assert content.role == "user"
    assert len(content.parts) == 1
    assert file_names == []
    mock_file_manager.create_part.assert_called_once()


def test_response_converter_thinking() -> None:
    """Test that ResponseConverter extracts thoughts into ThinkingBlocks."""
    converter = ResponseConverter()

    # Mock a response with a thought part and a text part
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = types.FinishReason.STOP
    mock_candidate.content.role = "model"

    part1 = types.Part(text="Thinking process...", thought=True)
    part2 = types.Part(text="Final Answer")

    mock_candidate.content.parts = [part1, part2]
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None

    state = GeminiResponseParseState()
    chat_response = converter.to_chat_response(mock_response, state=state)

    blocks = chat_response.message.blocks
    assert len(blocks) == 2
    assert isinstance(blocks[0], ThinkingBlock)
    assert blocks[0].content == "Thinking process..."
    assert isinstance(blocks[1], TextBlock)
    assert blocks[1].text == "Final Answer"


def test_thoughts_in_response_converter() -> None:
    """Real test: thought parts are converted EXACTLY into blocks + signatures alignment."""
    converter = ResponseConverter()
    state = GeminiResponseParseState()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = types.FinishReason.STOP
    mock_candidate.content.role = "model"

    thought_part = types.Part(
        text="This is a thought.",
        thought=True,
        thought_signature=b"sig-1",
    )
    text_part = types.Part(text="This is not a thought.")

    mock_candidate.content.parts = [thought_part, text_part]
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None
    mock_response.prompt_feedback = None

    chat_response = converter.to_chat_response(mock_response, state=state)

    blocks = chat_response.message.blocks
    assert len(blocks) == 2

    assert isinstance(blocks[0], ThinkingBlock)
    assert blocks[0].content == "This is a thought."
    assert blocks[0].additional_information.get("thought_signature") == b"sig-1"
    assert blocks[0].additional_information.get("thought") is True

    assert isinstance(blocks[1], TextBlock)
    assert blocks[1].text == "This is not a thought."

    assert chat_response.message.additional_kwargs.get("thought_signatures") == [
        b"sig-1",
        None,
    ]


def test_response_converter_code_execution() -> None:
    """
    Validate raw extraction preserves code execution parts.

    ResponseConverter does not map executable_code / code_execution_result into
    dedicated LlamaIndex blocks today, but raw extraction must preserve them.
    """
    converter = ResponseConverter()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = types.FinishReason.STOP
    mock_candidate.content.role = "model"

    # Provide raw parts for executable_code and code_execution_result.
    # We keep the Parts as MagicMock to avoid relying on SDK constructors.
    part_text_1 = types.Part(text="I'll calculate primes.")

    part_code = MagicMock()
    part_code.text = None
    part_code.thought = None
    part_code.function_call = None
    part_code.function_response = None
    part_code.inline_data = None
    part_code.thought_signature = None
    part_code.executable_code = {
        "code": "print('hello')",
        "language": types.Language.PYTHON,
    }

    part_result = MagicMock()
    part_result.text = None
    part_result.thought = None
    part_result.function_call = None
    part_result.function_response = None
    part_result.inline_data = None
    part_result.thought_signature = None
    part_result.code_execution_result = {
        "outcome": types.Outcome.OUTCOME_OK,
        "output": "hello",
    }

    part_text_2 = types.Part(text="Done.")

    mock_candidate.content.parts = [part_text_1, part_code, part_result, part_text_2]
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None

    # Ensure raw extraction retains the code execution structures.
    mock_candidate.model_dump.return_value = {
        "finish_reason": types.FinishReason.STOP,
        "content": {
            "role": "model",
            "parts": [
                {"text": "I'll calculate primes."},
                {
                    "executable_code": {
                        "code": "print('hello')",
                        "language": types.Language.PYTHON,
                    }
                },
                {
                    "code_execution_result": {
                        "outcome": types.Outcome.OUTCOME_OK,
                        "output": "hello",
                    }
                },
                {"text": "Done."},
            ],
        },
    }

    state = GeminiResponseParseState()
    chat_response = converter.to_chat_response(mock_response, state=state)

    blocks = chat_response.message.blocks

    # Converter should keep the user-visible text parts.
    assert any(
        isinstance(b, TextBlock) and b.text == "I'll calculate primes." for b in blocks
    )
    assert any(isinstance(b, TextBlock) and b.text == "Done." for b in blocks)

    # Raw should retain executable_code and code_execution_result parts.
    assert isinstance(chat_response.raw, dict)
    parts = chat_response.raw.get("content", {}).get("parts", [])
    assert isinstance(parts, list)
    assert any("executable_code" in p for p in parts)
    assert any("code_execution_result" in p for p in parts)


def test_code_execution_response_parts() -> None:
    """
    Test that code execution response contains executable_code, code_execution_result, and text parts.

    This is a unit-level test that validates ResponseConverter raw extraction and
    that the user-visible message content includes the final answer.
    """
    converter = ResponseConverter()
    state = GeminiResponseParseState()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = types.FinishReason.STOP
    mock_candidate.content.role = "model"
    mock_response.candidates = [mock_candidate]

    mock_candidate.content.parts = [
        types.Part(
            text="I'll calculate the sum of the first 50 prime numbers for you."
        ),
        MagicMock(
            text=None,
            inline_data=None,
            thought=None,
            function_call=None,
            function_response=None,
            executable_code={
                "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprimes = []\nn = 2\nwhile len(primes) < 50:\n    if is_prime(n):\n        primes.append(n)\n    n += 1\n\nprint(f'Sum of first 50 primes: {sum(primes)}')",
                "language": types.Language.PYTHON,
            },
        ),
        MagicMock(
            text=None,
            inline_data=None,
            thought=None,
            function_call=None,
            function_response=None,
            code_execution_result={
                "outcome": types.Outcome.OUTCOME_OK,
                "output": "Sum of first 50 primes: 5117",
            },
        ),
        types.Part(text="The sum of the first 50 prime numbers is 5117."),
    ]

    mock_response.prompt_feedback = None
    mock_response.usage_metadata = None

    mock_candidate.model_dump.return_value = {
        "finish_reason": types.FinishReason.STOP,
        "content": {
            "role": "model",
            "parts": [
                {
                    "text": "I'll calculate the sum of the first 50 prime numbers for you."
                },
                {
                    "executable_code": {
                        "code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprimes = []\nn = 2\nwhile len(primes) < 50:\n    if is_prime(n):\n        primes.append(n)\n    n += 1\n\nprint(f'Sum of first 50 primes: {sum(primes)}')",
                        "language": types.Language.PYTHON,
                    }
                },
                {
                    "code_execution_result": {
                        "outcome": types.Outcome.OUTCOME_OK,
                        "output": "Sum of first 50 primes: 5117",
                    }
                },
                {"text": "The sum of the first 50 prime numbers is 5117."},
            ],
        },
    }

    response = converter.to_chat_response(mock_response, state=state)

    assert response is not None
    assert response.message is not None
    assert "5117" in response.message.content

    raw = response.raw
    assert isinstance(raw, dict)
    content = raw.get("content")
    assert isinstance(content, dict)
    parts = content.get("parts")
    assert isinstance(parts, list)
    assert parts

    # Validate each part's shape.
    for part in parts:
        assert isinstance(part, dict)

        if part.get("text") is not None:
            assert isinstance(part["text"], str)
            assert part["text"].strip()

        if part.get("executable_code") is not None:
            executable_code = part["executable_code"]
            assert isinstance(executable_code, dict)
            assert isinstance(executable_code.get("code"), str)
            assert executable_code["code"].strip()
            assert executable_code.get("language") == types.Language.PYTHON

        if part.get("code_execution_result") is not None:
            code_execution_result = part["code_execution_result"]
            assert isinstance(code_execution_result, dict)
            assert code_execution_result.get("outcome") == types.Outcome.OUTCOME_OK
            output = code_execution_result.get("output")
            assert isinstance(output, str)
            assert output.strip()


def test_response_converter_streaming_thoughts_accumulate() -> None:
    """Ensure streaming state accumulates thought and text blocks across chunks."""
    converter = ResponseConverter()
    state = GeminiResponseParseState()

    # Chunk 1: thought
    r1 = MagicMock()
    c1 = MagicMock()
    c1.finish_reason = types.FinishReason.STOP
    c1.content.role = "model"
    c1.content.parts = [types.Part(text="This is a thought.", thought=True)]
    r1.candidates = [c1]
    r1.usage_metadata = None
    r1.prompt_feedback = None

    resp1 = converter.to_chat_response(r1, state=state)
    assert any(isinstance(b, ThinkingBlock) for b in resp1.message.blocks)

    # Chunk 2: non-thought text delta
    r2 = MagicMock()
    c2 = MagicMock()
    c2.finish_reason = types.FinishReason.STOP
    c2.content.role = "model"
    c2.content.parts = [types.Part(text="This is not a thought.")]
    r2.candidates = [c2]
    r2.usage_metadata = None
    r2.prompt_feedback = None

    resp2 = converter.to_chat_response(r2, state=state)
    assert any(isinstance(b, ThinkingBlock) for b in resp2.message.blocks)
    assert any(isinstance(b, TextBlock) for b in resp2.message.blocks)

    # Ensure accumulated blocks preserve order and count == 2
    assert len(resp2.message.blocks) == 2
    assert isinstance(resp2.message.blocks[0], ThinkingBlock)
    assert isinstance(resp2.message.blocks[1], TextBlock)
