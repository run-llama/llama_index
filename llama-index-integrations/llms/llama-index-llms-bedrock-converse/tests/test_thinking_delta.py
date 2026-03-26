from unittest.mock import MagicMock

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    ThinkingBlock,
    TextBlock,
)
from llama_index.llms.bedrock_converse import BedrockConverse


@pytest.fixture
def mock_bedrock_client():
    return MagicMock()


@pytest.fixture
def bedrock_with_thinking(mock_bedrock_client):
    return BedrockConverse(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        thinking={"type": "enabled", "budget_tokens": 1024},
        client=mock_bedrock_client,
    )


def test_thinking_delta_populated_in_stream_chat(
    bedrock_with_thinking, mock_bedrock_client
):
    mock_bedrock_client.converse_stream.return_value = {
        "stream": [
            {
                "contentBlockDelta": {
                    "delta": {
                        "reasoningContent": {
                            "text": "Let me think",
                            "signature": "sig1",
                        }
                    },
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {
                        "reasoningContent": {
                            "text": " about this",
                            "signature": "sig2",
                        }
                    },
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": "The answer is"},
                    "contentBlockIndex": 1,
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

    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    responses = list(bedrock_with_thinking.stream_chat(messages))

    assert len(responses) > 0

    thinking_responses = [
        r for r in responses if r.additional_kwargs.get("thinking_delta") is not None
    ]
    assert len(thinking_responses) == 2
    assert thinking_responses[0].additional_kwargs["thinking_delta"] == "Let me think"
    assert thinking_responses[1].additional_kwargs["thinking_delta"] == " about this"

    text_responses = [
        r
        for r in responses
        if r.delta and r.additional_kwargs.get("thinking_delta") is None
    ]
    assert len(text_responses) >= 1
    assert text_responses[0].delta == "The answer is"


@pytest.mark.asyncio
async def test_thinking_delta_populated_in_astream_chat(
    bedrock_with_thinking, monkeypatch
):
    events = [
        {
            "contentBlockDelta": {
                "delta": {
                    "reasoningContent": {
                        "text": "Let me think",
                        "signature": "sig1",
                    }
                },
                "contentBlockIndex": 0,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {
                    "reasoningContent": {
                        "text": " about this",
                        "signature": "sig2",
                    }
                },
                "contentBlockIndex": 0,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"text": "The answer is"},
                "contentBlockIndex": 1,
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

    async def _fake_converse_with_retry_async(**_kwargs):
        async def _gen():
            for event in events:
                yield event

        return _gen()

    monkeypatch.setattr(
        "llama_index.llms.bedrock_converse.base.converse_with_retry_async",
        _fake_converse_with_retry_async,
    )

    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    response_stream = await bedrock_with_thinking.astream_chat(messages)
    responses = [r async for r in response_stream]

    assert len(responses) > 0

    thinking_responses = [
        r for r in responses if r.additional_kwargs.get("thinking_delta") is not None
    ]
    assert len(thinking_responses) == 2
    assert thinking_responses[0].additional_kwargs["thinking_delta"] == "Let me think"
    assert thinking_responses[1].additional_kwargs["thinking_delta"] == " about this"

    text_responses = [
        r
        for r in responses
        if r.delta and r.additional_kwargs.get("thinking_delta") is None
    ]
    assert len(text_responses) >= 1
    assert text_responses[0].delta == "The answer is"


def test_thinking_delta_none_for_non_thinking_content(
    bedrock_with_thinking, mock_bedrock_client
):
    mock_bedrock_client.converse_stream.return_value = {
        "stream": [
            {
                "contentBlockStart": {
                    "start": {"text": ""},
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": "Regular text"},
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

    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    responses = list(bedrock_with_thinking.stream_chat(messages))

    text_responses = [r for r in responses if r.delta]
    assert all(
        r.additional_kwargs.get("thinking_delta") is None for r in text_responses
    )


def test_thinking_block_in_message_blocks(bedrock_with_thinking, mock_bedrock_client):
    mock_bedrock_client.converse_stream.return_value = {
        "stream": [
            {
                "contentBlockDelta": {
                    "delta": {
                        "reasoningContent": {
                            "text": "Thinking content",
                            "signature": "sig",
                        }
                    },
                    "contentBlockIndex": 0,
                }
            },
            {
                "contentBlockDelta": {
                    "delta": {"text": "Text content"},
                    "contentBlockIndex": 1,
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

    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    responses = list(bedrock_with_thinking.stream_chat(messages))

    final_response = responses[-1]
    assert len(final_response.message.blocks) >= 2

    thinking_blocks = [
        b for b in final_response.message.blocks if isinstance(b, ThinkingBlock)
    ]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].content == "Thinking content"

    text_blocks = [b for b in final_response.message.blocks if isinstance(b, TextBlock)]
    assert len(text_blocks) >= 1


def test_thinking_delta_populated_in_chat(bedrock_with_thinking, mock_bedrock_client):
    mock_bedrock_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": "I am thinking",
                                "signature": "sig",
                            }
                        }
                    },
                    {"text": "The answer is 42"},
                ],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
    }

    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    response = bedrock_with_thinking.chat(messages)

    # In non-streaming chat, thinking_delta should NOT be in additional_kwargs
    assert "thinking_delta" not in response.additional_kwargs
    # But it should be in blocks as a ThinkingBlock
    assert any(isinstance(b, ThinkingBlock) for b in response.message.blocks)
    thinking_block = next(
        b for b in response.message.blocks if isinstance(b, ThinkingBlock)
    )
    assert thinking_block.content == "I am thinking"


def test_thinking_block_round_trip(bedrock_with_thinking, mock_bedrock_client):
    from llama_index.llms.bedrock_converse.utils import messages_to_converse_messages

    messages = [
        ChatMessage(role=MessageRole.USER, content="Explain 42"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ThinkingBlock(
                    content="I need to calculate",
                    additional_information={"signature": "sig123"},
                ),
                TextBlock(text="It is the meaning of life"),
            ],
        ),
        ChatMessage(role=MessageRole.USER, content="Thanks"),
    ]

    converse_messages, _ = messages_to_converse_messages(messages, "some-model")

    # The assistant message should have 2 content blocks in Bedrock format
    assistant_msg = converse_messages[1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["content"]) == 2
    assert "reasoningContent" in assistant_msg["content"][0]
    assert (
        assistant_msg["content"][0]["reasoningContent"]["reasoningText"]["text"]
        == "I need to calculate"
    )
    assert (
        assistant_msg["content"][0]["reasoningContent"]["reasoningText"]["signature"]
        == "sig123"
    )
    assert assistant_msg["content"][1]["text"] == "It is the meaning of life"
