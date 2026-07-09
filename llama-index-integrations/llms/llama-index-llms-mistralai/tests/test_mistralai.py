import asyncio
from types import SimpleNamespace

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.mistralai import MistralAI


def test_embedding_class():
    names_of_base_classes = [b.__name__ for b in MistralAI.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


def _text_chunk(text: str) -> SimpleNamespace:
    """Mimic a Mistral stream chunk carrying a plain-string content delta."""
    delta = SimpleNamespace(role=None, tool_calls=None, content=text)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(data=SimpleNamespace(choices=[choice]))


def test_stream_chat_aggregates_single_text_block(monkeypatch):
    """The aggregated streamed message must carry one cumulative TextBlock.

    Regression test: `stream_chat` used to append a new TextBlock holding the
    whole accumulated text on every chunk, so `ChatMessage.content` (which joins
    all TextBlock texts with "\n") became the newline-joined concatenation of
    every growing prefix instead of the reply.
    """
    llm = MistralAI(api_key="fake", model="mistral-large-latest")

    def fake_stream(**kwargs):
        return iter([_text_chunk("Hello"), _text_chunk(" "), _text_chunk("world")])

    monkeypatch.setattr(llm._client.chat, "stream", fake_stream)

    responses = list(
        llm.stream_chat([ChatMessage(role=MessageRole.USER, content="hi")])
    )
    final = responses[-1]

    text_blocks = [b for b in final.message.blocks if isinstance(b, TextBlock)]
    assert len(text_blocks) == 1
    assert final.message.content == "Hello world"
    # the per-chunk delta stays correct throughout the stream
    assert "".join(r.delta or "" for r in responses) == "Hello world"


def test_astream_chat_aggregates_single_text_block(monkeypatch):
    """Async counterpart of the aggregation regression test."""
    llm = MistralAI(api_key="fake", model="mistral-large-latest")

    async def fake_stream_async(**kwargs):
        async def agen():
            for chunk in [_text_chunk("Hello"), _text_chunk(" "), _text_chunk("world")]:
                yield chunk

        return agen()

    monkeypatch.setattr(llm._client.chat, "stream_async", fake_stream_async)

    async def collect():
        return [
            r
            async for r in await llm.astream_chat(
                [ChatMessage(role=MessageRole.USER, content="hi")]
            )
        ]

    responses = asyncio.run(collect())
    final = responses[-1]

    text_blocks = [b for b in final.message.blocks if isinstance(b, TextBlock)]
    assert len(text_blocks) == 1
    assert final.message.content == "Hello world"
    assert "".join(r.delta or "" for r in responses) == "Hello world"
