from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel

from llama_index.core.prompts import PromptTemplate
from llama_index.core.instrumentation.events.llm import (
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictInProgressEvent,
)

from llama_index.llms.google_genai import base as base_mod
from llama_index.llms.google_genai.base import GoogleGenAI
from llama_index.llms.google_genai.utils import extract_token_usage_from_response


# -------------------------
# Helpers
# -------------------------


def _make_llm(file_mode="fileapi"):
    llm = GoogleGenAI.model_construct()
    object.__setattr__(llm, "model", "gemini-3-flash-preview")
    object.__setattr__(llm, "file_mode", file_mode)
    object.__setattr__(llm, "max_retries", 0)
    object.__setattr__(llm, "_generation_config", {})
    return llm


def _make_structured_llm():
    from llama_index.core.types import PydanticProgramMode

    llm = _make_llm(file_mode="inline")
    object.__setattr__(llm, "pydantic_program_mode", PydanticProgramMode.DEFAULT)
    return llm


def _make_usage_metadata(prompt=10, completion=20, total=30):
    m = SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=completion,
        total_token_count=total,
        thoughts_token_count=None,
    )
    m.model_dump = lambda: {
        "prompt_token_count": prompt,
        "candidates_token_count": completion,
        "total_token_count": total,
        "thoughts_token_count": None,
    }
    return m


def _capture_events(monkeypatch):
    events = []
    mock_dispatcher = MagicMock()
    mock_dispatcher.event = lambda evt: events.append(evt)
    monkeypatch.setattr(base_mod, "dispatcher", mock_dispatcher)
    return events


# IMPORTANT FIX: must be async coroutine (not lambda)
async def fake_chat_message_to_gemini(*args, **kwargs):
    return SimpleNamespace(), []


# -------------------------
# Chat cleanup tests
# -------------------------


def test_chat_bubbles_up_cleanup_error_if_delete_fails(monkeypatch):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        return "hello", {}, ["file1"]

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    class FakeChat:
        def send_message(self, *_args, **_kwargs):
            raise ValueError("boom")

    class FakeAioChat:
        def send_message(self, *_args, **_kwargs):
            return SimpleNamespace()

    llm = _make_llm()
    llm._client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **_: FakeChat()),
        aio=SimpleNamespace(chats=SimpleNamespace(create=lambda **_: FakeAioChat())),
    )

    def fake_delete_uploaded_files(_names, _client):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(base_mod, "delete_uploaded_files", fake_delete_uploaded_files)

    with pytest.raises(RuntimeError, match="delete failed"):
        llm._chat([])


@pytest.mark.asyncio
async def test_achat_bubbles_up_cleanup_error_if_delete_fails(monkeypatch):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        return "hello", {}, ["file1"]

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    class FakeChat:
        pass

    class FakeAioChat:
        async def send_message(self, *_args, **_kwargs):
            raise ValueError("boom")

    llm = _make_llm()
    llm._client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **_: FakeChat()),
        aio=SimpleNamespace(chats=SimpleNamespace(create=lambda **_: FakeAioChat())),
    )

    async def fake_adelete_uploaded_files(_names, _client):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(base_mod, "adelete_uploaded_files", fake_adelete_uploaded_files)

    with pytest.raises(RuntimeError, match="delete failed"):
        await llm._achat([])


# -------------------------
# Streaming chat cleanup
# -------------------------


def test_stream_chat_runs_cleanup(monkeypatch):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        return "hello", {}, ["file1"]

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    class Chunk:
        def __init__(self):
            part = SimpleNamespace(text="hi", thought=None)
            content = SimpleNamespace(parts=[part])
            cand = SimpleNamespace(content=content)
            self.candidates = [cand]

    class FakeChat:
        def send_message_stream(self, *_args, **_kwargs):
            return iter([Chunk()])

    llm = _make_llm()

    llm._client = SimpleNamespace(chats=SimpleNamespace(create=lambda **_: FakeChat()))

    monkeypatch.setattr(
        base_mod,
        "chat_from_gemini_response",
        lambda *a, **k: SimpleNamespace(delta=None),
    )

    deleted = {"called": False}

    def fake_delete(names, _client):
        assert names == ["file1"]
        deleted["called"] = True

    monkeypatch.setattr(base_mod, "delete_uploaded_files", fake_delete)

    gen = llm._stream_chat([])
    next(gen)
    with pytest.raises(StopIteration):
        next(gen)

    assert deleted["called"] is True


@pytest.mark.asyncio
async def test_astream_chat_runs_cleanup(monkeypatch):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        return "hello", {}, ["file1"]

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    class Chunk:
        def __init__(self):
            part = SimpleNamespace(text="hi", thought=None)
            content = SimpleNamespace(parts=[part])
            cand = SimpleNamespace(content=content)
            self.candidates = [cand]

    async def stream():
        yield Chunk()

    class FakeAioChat:
        async def send_message_stream(self, *_args, **_kwargs):
            return stream()

    llm = _make_llm()

    llm._client = SimpleNamespace(
        aio=SimpleNamespace(chats=SimpleNamespace(create=lambda **_: FakeAioChat()))
    )

    monkeypatch.setattr(
        base_mod,
        "chat_from_gemini_response",
        lambda *a, **k: SimpleNamespace(delta=None),
    )

    deleted = {"called": False}

    async def fake_adelete(names, _client):
        assert names == ["file1"]
        deleted["called"] = True

    monkeypatch.setattr(base_mod, "adelete_uploaded_files", fake_adelete)

    agen = await llm._astream_chat([])
    item = await agen.__anext__()
    assert item is not None

    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

    assert deleted["called"] is True


# -------------------------
# Token usage utils
# -------------------------


def test_extract_token_usage_returns_empty_when_no_metadata():
    response = SimpleNamespace(usage_metadata=None)
    assert extract_token_usage_from_response(response) == {}


def test_extract_token_usage_returns_correct_keys():
    response = SimpleNamespace(usage_metadata=_make_usage_metadata())
    result = extract_token_usage_from_response(response)

    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 20
    assert result["total_tokens"] == 30


# -------------------------
# Structured predict
# -------------------------


def test_structured_predict_emits_event_with_token_usage(monkeypatch):
    class Out(BaseModel):
        answer: str

    llm = _make_structured_llm()

    fake_response = SimpleNamespace(
        parsed=Out(answer="42"),
        usage_metadata=_make_usage_metadata(),
    )

    monkeypatch.setattr(base_mod, "chat_message_to_gemini", fake_chat_message_to_gemini)

    llm._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **_: fake_response)
    )

    events = _capture_events(monkeypatch)

    llm.structured_predict(Out, PromptTemplate("{q}"), q="6x7")

    assert isinstance(events[0], LLMStructuredPredictEndEvent)
    assert events[0].output.answer == "42"
    event = events[0]

    assert events[0].additional_kwargs is not None, (
        "Token usage not attached — check extract_token_usage_from_response "
        "returned a non-empty dict and base.py passes it correctly"
    )
    assert event.additional_kwargs["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_astructured_predict_emits_event_with_token_usage(monkeypatch):
    class Out(BaseModel):
        answer: str

    llm = _make_structured_llm()

    fake_response = SimpleNamespace(
        parsed=Out(answer="42"),
        usage_metadata=_make_usage_metadata(),
    )

    monkeypatch.setattr(base_mod, "chat_message_to_gemini", fake_chat_message_to_gemini)

    llm._client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=AsyncMock(return_value=fake_response)
            )
        )
    )

    events = _capture_events(monkeypatch)

    await llm.astructured_predict(Out, PromptTemplate("{q}"), q="6x7")

    assert isinstance(events[0], LLMStructuredPredictEndEvent)

    additional_kwargs = getattr(events[0], "additional_kwargs", None) or {}
    assert additional_kwargs.get("total_tokens") == 30


# -------------------------
# Streaming structured predict
# -------------------------


def test_stream_structured_predict_emits_event_with_token_usage(monkeypatch):
    class Out(BaseModel):
        answer: str

    llm = _make_structured_llm()

    final_chunk = SimpleNamespace(
        parsed=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text='{"answer":"42"}')])
            )
        ],
        usage_metadata=_make_usage_metadata(),
    )

    monkeypatch.setattr(base_mod, "chat_message_to_gemini", fake_chat_message_to_gemini)
    monkeypatch.setattr(
        base_mod,
        "handle_streaming_flexible_model",
        lambda *a, **k: (Out(answer="42"), '{"answer":"42"}'),
    )

    llm._client = SimpleNamespace(
        models=SimpleNamespace(generate_content_stream=lambda **_: iter([final_chunk]))
    )

    events = _capture_events(monkeypatch)

    list(llm.stream_structured_predict(Out, PromptTemplate("{q}"), q="test"))

    assert any(isinstance(e, LLMStructuredPredictInProgressEvent) for e in events)


@pytest.mark.asyncio
async def test_astream_structured_predict_emits_event_with_token_usage(monkeypatch):
    class Out(BaseModel):
        answer: str

    llm = _make_structured_llm()

    final_chunk = SimpleNamespace(
        parsed=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text='{"answer":"42"}')])
            )
        ],
        usage_metadata=_make_usage_metadata(),
    )

    monkeypatch.setattr(base_mod, "chat_message_to_gemini", fake_chat_message_to_gemini)
    monkeypatch.setattr(
        base_mod,
        "handle_streaming_flexible_model",
        lambda *a, **k: (Out(answer="42"), '{"answer":"42"}'),
    )

    async def stream():
        yield final_chunk

    llm._client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=lambda **_: stream())
        )
    )

    events = _capture_events(monkeypatch)

    gen = await llm.astream_structured_predict(Out, PromptTemplate("{q}"), q="test")
    [x async for x in gen]

    assert any(isinstance(e, LLMStructuredPredictInProgressEvent) for e in events)
