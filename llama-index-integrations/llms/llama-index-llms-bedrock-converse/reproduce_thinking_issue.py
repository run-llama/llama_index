"""Reproduce issue #20489: Nova models embed <thinking> tags in text blocks."""

from unittest.mock import MagicMock

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.bedrock_converse import BedrockConverse

NOVA_MODEL = "amazon.nova-2-lite-v1:0"
RAW_TEXT = (
    "<thinking>Let me reason about this step by step.</thinking>"
    "The capital of France is Paris."
)
EXPECTED_TEXT = "The capital of France is Paris."


def _assert_no_thinking_tags(text: str, context: str) -> None:
    assert "<thinking>" not in text, f"{context}: found opening <thinking> tag in {text!r}"
    assert "</thinking>" not in text, f"{context}: found closing </thinking> tag in {text!r}"


def test_non_streaming_chat(mock_client: MagicMock) -> None:
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": RAW_TEXT}],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
    }

    llm = BedrockConverse(model=NOVA_MODEL, client=mock_client)
    response = llm.chat([ChatMessage(role=MessageRole.USER, content="Capital of France?")])

    text_blocks = [b for b in response.message.blocks if isinstance(b, TextBlock)]
    assert len(text_blocks) == 1, f"Expected one TextBlock, got {text_blocks}"
    _assert_no_thinking_tags(text_blocks[0].text, "non-streaming chat")
    assert text_blocks[0].text.strip() == EXPECTED_TEXT


def test_streaming_chat(mock_client: MagicMock) -> None:
    mock_client.converse_stream.return_value = {
        "stream": [
            {
                "contentBlockDelta": {
                    "delta": {"text": "<thinking>Let me reason"},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": " about this step by step.</thinking>"},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": "The capital of France is Paris."},
                    "contentBlockIndex": 0,
                }
            },
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 20,
                        "totalTokens": 30,
                    }
                }
            },
        ]
    }

    llm = BedrockConverse(model=NOVA_MODEL, client=mock_client)
    responses = list(
        llm.stream_chat([ChatMessage(role=MessageRole.USER, content="Capital of France?")])
    )

    final_response = responses[-1]
    text_blocks = [b for b in final_response.message.blocks if isinstance(b, TextBlock)]
    assert len(text_blocks) >= 1, f"Expected TextBlock in final response, got {text_blocks}"
    _assert_no_thinking_tags(text_blocks[0].text, "streaming chat final content")
    assert text_blocks[0].text.strip() == EXPECTED_TEXT

    concatenated_deltas = "".join(r.delta for r in responses if r.delta)
    _assert_no_thinking_tags(concatenated_deltas, "streaming chat deltas")


def main() -> None:
    mock_client = MagicMock()
    print("Testing non-streaming chat...")
    test_non_streaming_chat(mock_client)
    print("  PASS: thinking tags stripped from non-streaming response")

    mock_client = MagicMock()
    print("Testing streaming chat...")
    test_streaming_chat(mock_client)
    print("  PASS: thinking tags stripped from streaming response")

    print("\nAll reproduction checks passed.")


if __name__ == "__main__":
    main()
