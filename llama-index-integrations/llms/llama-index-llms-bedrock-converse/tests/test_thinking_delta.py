from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.llms.bedrock_converse.utils import messages_to_converse_messages


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


def _usage_event():
    return {
        "metadata": {
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30,
            }
        }
    }


def _interleaved_tool_loop_events():
    """
    Realistic ConverseStream wire shape for a Claude tool-loop turn:
    thinking -> text -> signature-only thinking -> text -> tool call -> tool call.

    Reasoning text deltas are followed by a separate signature-only delta, and
    every block ends with `contentBlockStop`.
    """
    return [
        # Index 0: reasoning block with text, then its own signature
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"text": "Check the"}},
                "contentBlockIndex": 0,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"text": " weather."}},
                "contentBlockIndex": 0,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"signature": "sigA"}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        # Index 1: text
        {
            "contentBlockDelta": {
                "delta": {"text": "Let me check Paris."},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        # Index 2: omitted-mode reasoning block (signature only)
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"signature": "sigB"}},
                "contentBlockIndex": 2,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 2}},
        # Index 3: text
        {
            "contentBlockDelta": {
                "delta": {"text": "And Tokyo."},
                "contentBlockIndex": 3,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 3}},
        # Index 4: tool call
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tool-1", "name": "get_weather"}},
                "contentBlockIndex": 4,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"city": "Pa'}},
                "contentBlockIndex": 4,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": 'ris"}'}},
                "contentBlockIndex": 4,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 4}},
        # Index 5: tool call
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tool-2", "name": "get_time"}},
                "contentBlockIndex": 5,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"city": "Tokyo"}'}},
                "contentBlockIndex": 5,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 5}},
        {"messageStop": {"stopReason": "tool_use"}},
        _usage_event(),
    ]


async def _stream_responses(llm, mock_bedrock_client, monkeypatch, events, use_async):
    messages = [ChatMessage(role=MessageRole.USER, content="Test")]
    if not use_async:
        mock_bedrock_client.converse_stream.return_value = {"stream": events}
        return list(llm.stream_chat(messages))

    async def _fake_converse_with_retry_async(**_kwargs):
        async def _gen():
            for event in events:
                yield event

        return _gen()

    monkeypatch.setattr(
        "llama_index.llms.bedrock_converse.base.converse_with_retry_async",
        _fake_converse_with_retry_async,
    )
    response_stream = await llm.astream_chat(messages)
    return [r async for r in response_stream]


def _find_response(responses, event_type, index):
    """
    Find the response yielded for the last `event_type` event at `index`.
    """
    matches = [
        r
        for r in responses
        if event_type in r.raw and r.raw[event_type].get("contentBlockIndex") == index
    ]
    assert matches, f"No response yielded for {event_type} at index {index}"
    return matches[-1]


def _summarize_blocks(blocks):
    summary = []
    for block in blocks:
        if isinstance(block, ThinkingBlock):
            summary.append(
                ("thinking", block.content, block.additional_information["signature"])
            )
        elif isinstance(block, TextBlock):
            summary.append(("text", block.text))
        elif isinstance(block, ToolCallBlock):
            summary.append(
                ("tool", block.tool_call_id, block.tool_name, block.tool_kwargs)
            )
    return summary


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_stream_keeps_block_order_and_separate_signatures(
    bedrock_with_thinking, mock_bedrock_client, monkeypatch, use_async
):
    responses = await _stream_responses(
        bedrock_with_thinking,
        mock_bedrock_client,
        monkeypatch,
        _interleaved_tool_loop_events(),
        use_async,
    )

    # contentBlockDelta yield: text at index 3, before any tool call has started
    delta_response = _find_response(responses, "contentBlockDelta", 3)
    assert delta_response.delta == "And Tokyo."
    assert _summarize_blocks(delta_response.message.blocks) == [
        ("thinking", "Check the weather.", "sigA"),
        ("text", "Let me check Paris."),
        ("thinking", "", "sigB"),
        ("text", "And Tokyo."),
    ]
    assert delta_response.message.additional_kwargs["tool_call_id"] == []

    # contentBlockStart yield: second tool call just started, no input yet
    start_response = _find_response(responses, "contentBlockStart", 5)
    assert _summarize_blocks(start_response.message.blocks) == [
        ("thinking", "Check the weather.", "sigA"),
        ("text", "Let me check Paris."),
        ("thinking", "", "sigB"),
        ("text", "And Tokyo."),
        ("tool", "tool-1", "get_weather", {"city": "Paris"}),
        ("tool", "tool-2", "get_time", {}),
    ]
    assert start_response.message.additional_kwargs["tool_call_id"] == [
        "tool-1",
        "tool-2",
    ]

    # metadata yield: final message
    final_response = responses[-1]
    assert "metadata" in final_response.raw
    assert _summarize_blocks(final_response.message.blocks) == [
        ("thinking", "Check the weather.", "sigA"),
        ("text", "Let me check Paris."),
        ("thinking", "", "sigB"),
        ("text", "And Tokyo."),
        ("tool", "tool-1", "get_weather", {"city": "Paris"}),
        ("tool", "tool-2", "get_time", {"city": "Tokyo"}),
    ]
    assert final_response.message.additional_kwargs["tool_call_id"] == [
        "tool-1",
        "tool-2",
    ]
    # Separate text blocks are joined by core with a newline
    assert final_response.message.content == "Let me check Paris.\nAnd Tokyo."


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_stream_signature_only_thinking_block_persists(
    bedrock_with_thinking, mock_bedrock_client, monkeypatch, use_async
):
    events = [
        {
            "contentBlockDelta": {
                "delta": {"reasoningContent": {"signature": "omitted-sig"}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tool-1", "name": "get_weather"}},
                "contentBlockIndex": 1,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"city": "Paris"}'}},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        _usage_event(),
    ]

    responses = await _stream_responses(
        bedrock_with_thinking, mock_bedrock_client, monkeypatch, events, use_async
    )

    # The signature-only delta is visible as soon as it arrives
    delta_response = _find_response(responses, "contentBlockDelta", 0)
    assert _summarize_blocks(delta_response.message.blocks) == [
        ("thinking", "", "omitted-sig"),
    ]

    start_response = _find_response(responses, "contentBlockStart", 1)
    assert _summarize_blocks(start_response.message.blocks) == [
        ("thinking", "", "omitted-sig"),
        ("tool", "tool-1", "get_weather", {}),
    ]

    final_message = responses[-1].message
    assert _summarize_blocks(final_message.blocks) == [
        ("thinking", "", "omitted-sig"),
        ("tool", "tool-1", "get_weather", {"city": "Paris"}),
    ]
    # No text was streamed, so there is no TextBlock (matches non-streaming `chat`)
    assert final_message.content is None

    # The streamed message replays the signature-only block unchanged
    converse_messages, _ = messages_to_converse_messages(
        [ChatMessage(role=MessageRole.USER, content="Test"), final_message],
        "some-model",
    )
    assistant_msg = converse_messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"][0] == {
        "reasoningContent": {"reasoningText": {"text": "", "signature": "omitted-sig"}}
    }
    assert assistant_msg["content"][1]["toolUse"]["toolUseId"] == "tool-1"
    assert len(assistant_msg["content"]) == 2


def test_signature_only_thinking_block_round_trip():
    messages = [
        ChatMessage(role=MessageRole.USER, content="Weather in Paris?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            blocks=[
                ThinkingBlock(content="", additional_information={"signature": "sig1"}),
                TextBlock(text="Let me check."),
                ThinkingBlock(content="", additional_information={"signature": "sig2"}),
            ],
        ),
        ChatMessage(role=MessageRole.USER, content="Thanks"),
    ]

    converse_messages, _ = messages_to_converse_messages(messages, "some-model")

    assert converse_messages[1]["content"] == [
        {"reasoningContent": {"reasoningText": {"text": "", "signature": "sig1"}}},
        {"text": "Let me check."},
        {"reasoningContent": {"reasoningText": {"text": "", "signature": "sig2"}}},
    ]
