import asyncio
from typing import Any, List, Optional, Union

import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llama_index.llms.mistralai import MistralAI
from mistralai.client import models


def make_event(
    content: Optional[Union[str, List[Any]]] = None,
    tool_calls: Optional[List[Any]] = None,
) -> models.CompletionEvent:
    return models.CompletionEvent(
        data=models.CompletionChunk(
            id="cmpl-1",
            model="mistral-large-latest",
            choices=[
                models.CompletionResponseStreamChoice(
                    index=0,
                    delta=models.DeltaMessage(
                        role="assistant", content=content, tool_calls=tool_calls
                    ),
                    finish_reason=None,
                )
            ],
        )
    )


def make_tool_call(
    call_id: str = "call_1", name: str = "add", arguments: str = '{"a": 1, "b": 2}'
) -> models.ToolCall:
    return models.ToolCall(
        id=call_id,
        function=models.FunctionCall(name=name, arguments=arguments),
    )


class FakeStream:
    def __init__(self, events: List[Any]) -> None:
        self.events = events

    def __iter__(self) -> Any:
        return iter(self.events)

    async def __aiter__(self) -> Any:
        for event in self.events:
            yield event


def make_llm(monkeypatch: pytest.MonkeyPatch, events: List[Any], **kwargs: Any):
    kwargs.setdefault("model", "mistral-large-latest")
    llm = MistralAI(api_key="test", **kwargs)

    def stream(*args: Any, **kw: Any) -> FakeStream:
        return FakeStream(events)

    async def stream_async(*args: Any, **kw: Any) -> FakeStream:
        return FakeStream(events)

    monkeypatch.setattr(llm._client.chat, "stream", stream)
    monkeypatch.setattr(llm._client.chat, "stream_async", stream_async)
    return llm


MESSAGES = [ChatMessage(role="user", content="hi")]


def run_sync(llm: MistralAI) -> list:
    return list(llm.stream_chat(MESSAGES))


async def run_async(llm: MistralAI) -> list:
    return [r async for r in await llm.astream_chat(MESSAGES)]


def test_text_inference_embedding_class() -> None:
    names_of_base_classes = [b.__name__ for b in MistralAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def assert_text_deltas(responses: list, events: list) -> None:
    assert [r.message.content for r in responses] == [
        "Hello",
        "Hello world",
        "Hello world!",
    ]
    assert [r.delta for r in responses] == ["Hello", " world", "!"]
    last = responses[-1]
    assert [type(b) for b in last.message.blocks] == [TextBlock]
    assert last.message.blocks[0].text == "Hello world!"
    for r, e in zip(responses, events):
        assert r.message.role == MessageRole.ASSISTANT
        assert r.raw is e
        assert [type(b) for b in r.message.blocks] == [TextBlock]


def test_stream_chat_text_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [make_event("Hello"), make_event(" world"), make_event("!")]
    responses = run_sync(make_llm(monkeypatch, events))
    assert_text_deltas(responses, events)


def test_astream_chat_text_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [make_event("Hello"), make_event(" world"), make_event("!")]
    responses = asyncio.run(run_async(make_llm(monkeypatch, events)))
    assert_text_deltas(responses, events)


def list_events() -> list:
    return [
        make_event([models.TextChunk(text="Hello")]),
        make_event([models.TextChunk(text="world")]),
    ]


def assert_list_content(responses: list, events: list) -> None:
    for r, e in zip(responses, events):
        assert r.raw is e
        assert isinstance(r.raw, models.CompletionEvent)
    assert [r.delta for r in responses] == ["Hello\n", "world\n"]
    assert [r.message.content for r in responses] == ["Hello\n", "Hello\nworld\n"]
    for r in responses:
        assert len([b for b in r.message.blocks if isinstance(b, TextBlock)]) == 1


def test_stream_chat_list_content_raw_is_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = list_events()
    assert_list_content(run_sync(make_llm(monkeypatch, events)), events)


def test_astream_chat_list_content_raw_is_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = list_events()
    assert_list_content(asyncio.run(run_async(make_llm(monkeypatch, events))), events)


def tool_events() -> list:
    return [
        make_event("Calling"),
        make_event(None, [make_tool_call()]),
        make_event(" tool"),
        make_event("!"),
    ]


def assert_tool_preserved(responses: list) -> None:
    assert [r.message.content for r in responses] == [
        "Calling",
        "Calling",
        "Calling tool",
        "Calling tool!",
    ]
    assert [len(r.message.blocks) for r in responses] == [1, 2, 2, 2]
    for r in responses[1:]:
        text_blocks = [b for b in r.message.blocks if isinstance(b, TextBlock)]
        tool_blocks = [b for b in r.message.blocks if isinstance(b, ToolCallBlock)]
        assert len(text_blocks) == 1
        assert len(tool_blocks) == 1
        assert tool_blocks[0].tool_call_id == "call_1"
        assert tool_blocks[0].tool_name == "add"
        assert tool_blocks[0].tool_kwargs == '{"a": 1, "b": 2}'
    assert responses[-1].message.blocks[0].text == "Calling tool!"


def test_stream_chat_tool_call_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    assert_tool_preserved(run_sync(make_llm(monkeypatch, tool_events())))


def test_astream_chat_tool_call_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_tool_preserved(asyncio.run(run_async(make_llm(monkeypatch, tool_events()))))


def think_events() -> list:
    return [
        make_event("<think>\npondering\n</think>\nHello"),
        make_event(" world"),
    ]


def assert_thinking_hidden(responses: list) -> None:
    for r in responses:
        assert [type(b) for b in r.message.blocks] == [ThinkingBlock, TextBlock]
        assert r.message.blocks[0].content == "pondering"
    assert [r.message.content for r in responses] == ["Hello", "Hello world"]
    assert responses[-1].message.blocks[1].text == "Hello world"


def assert_thinking_shown(responses: list) -> None:
    full = "<think>\npondering\n</think>\nHello world"
    for r in responses:
        assert [type(b) for b in r.message.blocks] == [ThinkingBlock, TextBlock]
        assert r.message.blocks[0].content == "pondering"
    assert responses[-1].message.content == full


def test_stream_chat_thinking_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = make_llm(monkeypatch, think_events(), model="magistral-medium-2506")
    assert_thinking_hidden(run_sync(llm))


def test_astream_chat_thinking_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = make_llm(monkeypatch, think_events(), model="magistral-medium-2506")
    assert_thinking_hidden(asyncio.run(run_async(llm)))


def test_stream_chat_thinking_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = make_llm(
        monkeypatch, think_events(), model="magistral-medium-2506", show_thinking=True
    )
    assert_thinking_shown(run_sync(llm))


def test_astream_chat_thinking_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = make_llm(
        monkeypatch, think_events(), model="magistral-medium-2506", show_thinking=True
    )
    assert_thinking_shown(asyncio.run(run_async(llm)))


def two_tool_events() -> list:
    return [
        make_event("Calling"),
        make_event(None, [make_tool_call("call_1", "add", '{"a": 1}')]),
        make_event(
            None,
            [
                make_tool_call("call_2", "sub", '{"a": 2}'),
                make_tool_call("call_3", "mul", '{"a": 3}'),
            ],
        ),
        make_event("!"),
    ]


def assert_two_tool_calls(responses: list) -> None:
    assert [len(r.message.blocks) for r in responses] == [1, 2, 4, 4]
    last = responses[-1].message.blocks
    assert isinstance(last[0], TextBlock)
    assert last[0].text == "Calling!"
    tools = [b for b in last if isinstance(b, ToolCallBlock)]
    assert [(t.tool_call_id, t.tool_name, t.tool_kwargs) for t in tools] == [
        ("call_1", "add", '{"a": 1}'),
        ("call_2", "sub", '{"a": 2}'),
        ("call_3", "mul", '{"a": 3}'),
    ]


def test_stream_chat_two_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    assert_two_tool_calls(run_sync(make_llm(monkeypatch, two_tool_events())))


def test_astream_chat_two_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    assert_two_tool_calls(
        asyncio.run(run_async(make_llm(monkeypatch, two_tool_events())))
    )
