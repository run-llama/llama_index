from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel
from llama_index.core.prompts import PromptTemplate

from llama_index.llms.google_genai import base as base_mod
from llama_index.llms.google_genai.base import GoogleGenAI
from llama_index.llms.google_genai.utils import extract_token_usage_from_response


class FakeChat:
    def __init__(self, send_message_exc=None, stream_iter=None):
        self._send_message_exc = send_message_exc
        self._stream_iter = stream_iter

    def send_message(self, *_args, **_kwargs):
        if self._send_message_exc:
            raise self._send_message_exc
        return SimpleNamespace()

    def send_message_stream(self, *_args, **_kwargs):
        return self._stream_iter


class FakeAioChat:
    def __init__(self, send_message_exc=None, stream_aiter=None):
        self._send_message_exc = send_message_exc
        self._stream_aiter = stream_aiter

    async def send_message(self, *_args, **_kwargs):
        if self._send_message_exc:
            raise self._send_message_exc
        return SimpleNamespace()

    async def send_message_stream(self, *_args, **_kwargs):
        return self._stream_aiter


class FakeClient:
    def __init__(self, chat: FakeChat, aio_chat: FakeAioChat):
        self.chats = SimpleNamespace(create=lambda **_kwargs: chat)
        self.aio = SimpleNamespace(
            chats=SimpleNamespace(create=lambda **_kwargs: aio_chat)
        )


def _make_llm(file_mode="fileapi"):
    llm = GoogleGenAI.model_construct()
    object.__setattr__(llm, "model", "gemini-3-flash-preview")
    object.__setattr__(llm, "file_mode", file_mode)
    object.__setattr__(llm, "max_retries", 0)
    object.__setattr__(llm, "_generation_config", {})
    return llm


def _make_usage_metadata(prompt=10, completion=20, total=30):
    m = SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=completion,
        total_token_count=total,
        thoughts_token_count=None,
    )
    m.model_dump = lambda: {}
    return m


def _make_structured_llm():
    from llama_index.core.types import PydanticProgramMode

    llm = _make_llm(file_mode="inline")
    object.__setattr__(llm, "pydantic_program_mode", PydanticProgramMode.DEFAULT)
    return llm


def test_chat_bubbles_up_cleanup_error_if_delete_fails(monkeypatch):
    """
    Test that if cleanup fails, the cleanup exception (RuntimeError) is raised.
    Note: In Python, if an exception occurs in 'finally', it supersedes
    any exception that occurred in 'try'.
    """

    async def fake_prepare_chat_params(*_args, **_kwargs):
        return "hello", {}, ["file1"]

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    # 1. Force the model call to fail (ValueError)
    chat = FakeChat(send_message_exc=ValueError("boom"))
    aio_chat = FakeAioChat()
    llm = _make_llm(file_mode="fileapi")
    llm._client = FakeClient(chat=chat, aio_chat=aio_chat)

    # 2. Force delete to fail (RuntimeError)
    def fake_delete_uploaded_files(_names, _client):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(base_mod, "delete_uploaded_files", fake_delete_uploaded_files)

    # 3. We expect the RuntimeError (cleanup failure) to be the one raised/visible
    with pytest.raises(RuntimeError, match="delete failed"):
        llm._chat([])


@pytest.mark.asyncio
async def test_achat_bubbles_up_cleanup_error_if_delete_fails(monkeypatch):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        return "hello", {}, ["file1"]

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    chat = FakeChat()
    aio_chat = FakeAioChat(send_message_exc=ValueError("boom"))
    llm = _make_llm(file_mode="fileapi")
    llm._client = FakeClient(chat=chat, aio_chat=aio_chat)

    async def fake_adelete_uploaded_files(_names, _client):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(base_mod, "adelete_uploaded_files", fake_adelete_uploaded_files)

    with pytest.raises(RuntimeError, match="delete failed"):
        await llm._achat([])


def test_stream_chat_runs_cleanup(monkeypatch):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        return "hello", {}, ["file1"]

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    class Chunk:
        def __init__(self):
            part = SimpleNamespace(text="hi")
            content = SimpleNamespace(parts=[part])
            cand = SimpleNamespace(content=content)
            self.candidates = [cand]

    stream_iter = iter([Chunk()])
    chat = FakeChat(stream_iter=stream_iter)
    aio_chat = FakeAioChat()
    llm = _make_llm(file_mode="fileapi")
    llm._client = FakeClient(chat=chat, aio_chat=aio_chat)

    monkeypatch.setattr(
        base_mod,
        "chat_from_gemini_response",
        lambda *_args, **_kwargs: SimpleNamespace(delta=None),
    )

    deleted = {"called": False}

    def fake_delete_uploaded_files(names, _client):
        assert names == ["file1"]
        deleted["called"] = True

    monkeypatch.setattr(base_mod, "delete_uploaded_files", fake_delete_uploaded_files)

    gen = llm._stream_chat([])
    _ = next(gen)
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
            part = SimpleNamespace(text="hi")
            content = SimpleNamespace(parts=[part])
            cand = SimpleNamespace(content=content)
            self.candidates = [cand]

    async def stream_aiter():
        yield Chunk()

    chat = FakeChat()
    aio_chat = FakeAioChat(stream_aiter=stream_aiter())
    llm = _make_llm(file_mode="fileapi")
    llm._client = FakeClient(chat=chat, aio_chat=aio_chat)

    monkeypatch.setattr(
        base_mod,
        "chat_from_gemini_response",
        lambda *_args, **_kwargs: SimpleNamespace(delta=None),
    )

    deleted = {"called": False}

    async def fake_adelete_uploaded_files(names, _client):
        assert names == ["file1"]
        deleted["called"] = True

    monkeypatch.setattr(base_mod, "adelete_uploaded_files", fake_adelete_uploaded_files)

    agen = await llm._astream_chat([])
    item = await agen.__anext__()
    assert item is not None

    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

    assert deleted["called"] is True


def test_extract_token_usage_returns_empty_when_no_metadata():
    response = SimpleNamespace(usage_metadata=None)
    assert extract_token_usage_from_response(response) == {}


def test_extract_token_usage_returns_correct_keys():
    response = SimpleNamespace(usage_metadata=_make_usage_metadata())
    result = extract_token_usage_from_response(response)
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 20
    assert result["total_tokens"] == 30
    assert "usage_metadata" not in result


def test_structured_predict_sets_span_attributes(monkeypatch):
    class Out(BaseModel):
        answer: str

    llm = _make_structured_llm()

    fake_response = SimpleNamespace(
        parsed=Out(answer="42"),
        usage_metadata=_make_usage_metadata(),
        text='{"answer": "42"}',
    )

    async def fake_chat_message_to_gemini(*args, **kwargs):
        return (SimpleNamespace(), [])

    monkeypatch.setattr(
        base_mod,
        "chat_message_to_gemini",
        fake_chat_message_to_gemini,
    )

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response
    llm._client = fake_client

    span_attrs = {}
    fake_span = SimpleNamespace(set_attribute=lambda k, v: span_attrs.update({k: v}))

    mock_dispatcher = MagicMock()
    mock_dispatcher.get_current_span.return_value = fake_span
    monkeypatch.setattr(base_mod, "dispatcher", mock_dispatcher)

    prompt = PromptTemplate("{question}")
    llm.structured_predict(Out, prompt, question="What is 6x7?")

    assert span_attrs.get("llm.token_usage.prompt_tokens") == 10
    assert span_attrs.get("llm.token_usage.completion_tokens") == 20
    assert span_attrs.get("llm.token_usage.total_tokens") == 30


@pytest.mark.asyncio
async def test_astructured_predict_sets_span_attributes(monkeypatch):
    class Out(BaseModel):
        answer: str

    llm = _make_structured_llm()

    fake_response = SimpleNamespace(
        parsed=Out(answer="42"),
        usage_metadata=_make_usage_metadata(),
        text='{"answer": "42"}',
    )

    async def fake_chat_message_to_gemini(*args, **kwargs):
        return (MagicMock(), ())

    monkeypatch.setattr(base_mod, "chat_message_to_gemini", fake_chat_message_to_gemini)

    fake_client = MagicMock()
    fake_aio = MagicMock()
    fake_aio.models.generate_content = AsyncMock(return_value=fake_response)
    fake_client.aio = fake_aio
    llm._client = fake_client

    span_attrs = {}
    fake_span = SimpleNamespace(set_attribute=lambda k, v: span_attrs.update({k: v}))

    mock_dispatcher = MagicMock()
    mock_dispatcher.get_current_span.return_value = fake_span
    monkeypatch.setattr(base_mod, "dispatcher", mock_dispatcher)

    prompt = PromptTemplate("{question}")
    await llm.astructured_predict(Out, prompt, question="What is 6x7?")

    assert span_attrs.get("llm.token_usage.prompt_tokens") == 10
    assert span_attrs.get("llm.token_usage.completion_tokens") == 20
    assert span_attrs.get("llm.token_usage.total_tokens") == 30


def test_stream_structured_predict_sets_span_from_final_chunk(monkeypatch):
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

    async def fake_chat_message_to_gemini(*args, **kwargs):
        return (SimpleNamespace(), [])

    monkeypatch.setattr(base_mod, "chat_message_to_gemini", fake_chat_message_to_gemini)

    monkeypatch.setattr(
        base_mod,
        "handle_streaming_flexible_model",
        lambda *a, **kw: (Out(answer="42"), '{"answer":"42"}'),
    )

    fake_client = MagicMock()
    fake_client.models.generate_content_stream.return_value = iter([final_chunk])
    llm._client = fake_client

    span_attrs = {}
    fake_span = SimpleNamespace(set_attribute=lambda k, v: span_attrs.update({k: v}))

    mock_dispatcher = MagicMock()
    mock_dispatcher.get_current_span.return_value = fake_span
    monkeypatch.setattr(base_mod, "dispatcher", mock_dispatcher)

    prompt = PromptTemplate("{question}")
    gen = llm.stream_structured_predict(Out, prompt, question="What is 6x7?")
    results = list(gen)

    assert len(results) > 0
    assert span_attrs.get("llm.token_usage.prompt_tokens") == 10
    assert span_attrs.get("llm.token_usage.completion_tokens") == 20


@pytest.mark.asyncio
async def test_astream_structured_predict_sets_span_from_final_chunk(monkeypatch):
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

    # Mock chat_message_to_gemini as an async function
    async def fake_chat_message_to_gemini(*args, **kwargs):
        return (SimpleNamespace(), [])

    monkeypatch.setattr(
        base_mod,
        "chat_message_to_gemini",
        fake_chat_message_to_gemini,
    )

    monkeypatch.setattr(
        base_mod,
        "handle_streaming_flexible_model",
        lambda *a, **kw: (Out(answer="42"), '{"answer":"42"}'),
    )

    async def fake_stream():
        yield final_chunk

    fake_client = MagicMock()
    fake_aio = MagicMock()
    fake_aio.models.generate_content_stream = AsyncMock(return_value=fake_stream())
    fake_client.aio = fake_aio
    llm._client = fake_client

    span_attrs = {}
    fake_span = SimpleNamespace(set_attribute=lambda k, v: span_attrs.update({k: v}))

    mock_dispatcher = MagicMock()
    mock_dispatcher.get_current_span.return_value = fake_span
    monkeypatch.setattr(base_mod, "dispatcher", mock_dispatcher)

    prompt = PromptTemplate("{question}")
    gen = await llm.astream_structured_predict(Out, prompt, question="What is 6x7?")
    results = [item async for item in gen]

    assert len(results) > 0
    assert span_attrs.get("llm.token_usage.prompt_tokens") == 10
    assert span_attrs.get("llm.token_usage.completion_tokens") == 20
