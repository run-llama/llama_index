from types import SimpleNamespace

import pytest

from llama_index.llms.google_genai import base as base_mod
from llama_index.llms.google_genai.base import GoogleGenAI


class FakeChat:
    def __init__(self, send_message_exc=None, stream_iter=None):
        self._send_message_exc = send_message_exc
        self._stream_iter = stream_iter

    def send_message(self, *_args, **_kwargs):
        if self._send_message_exc:
            raise self._send_message_exc
        return SimpleNamespace()  # unused in our error-path tests

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
    object.__setattr__(llm, "model", "gemini-2.0-flash")
    object.__setattr__(llm, "file_mode", file_mode)
    object.__setattr__(llm, "max_retries", 0)
    object.__setattr__(llm, "_generation_config", {})

    return llm


def test_chat_deletes_files_and_logs_if_delete_fails(monkeypatch, caplog):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        next_msg = "hello"
        chat_kwargs = {}
        file_api_names = ["file1"]
        return next_msg, chat_kwargs, file_api_names

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    # Force the model call to fail so we hit finally.
    chat = FakeChat(send_message_exc=ValueError("boom"))
    aio_chat = FakeAioChat()
    llm = _make_llm(file_mode="fileapi")
    llm._client = FakeClient(chat=chat, aio_chat=aio_chat)

    # Force delete to fail so we hit the logger.warning path.
    def fake_delete_uploaded_files(_names, _client):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(base_mod, "delete_uploaded_files", fake_delete_uploaded_files)

    with pytest.raises(ValueError, match="boom"):
        llm._chat([])

    assert "Failed to delete uploaded files from Google GenAI File API" in caplog.text


@pytest.mark.asyncio
async def test_achat_deletes_files_and_logs_if_delete_fails(monkeypatch, caplog):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        next_msg = "hello"
        chat_kwargs = {}
        file_api_names = ["file1"]
        return next_msg, chat_kwargs, file_api_names

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    chat = FakeChat()
    aio_chat = FakeAioChat(send_message_exc=ValueError("boom"))
    llm = _make_llm(file_mode="fileapi")
    llm._client = FakeClient(chat=chat, aio_chat=aio_chat)

    async def fake_adelete_uploaded_files(_names, _client):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(base_mod, "adelete_uploaded_files", fake_adelete_uploaded_files)

    with pytest.raises(ValueError, match="boom"):
        await llm._achat([])

    assert "Failed to delete uploaded files from Google GenAI File API" in caplog.text


def test_stream_chat_runs_cleanup(monkeypatch):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        next_msg = "hello"
        chat_kwargs = {}
        file_api_names = ["file1"]
        return next_msg, chat_kwargs, file_api_names

    monkeypatch.setattr(base_mod, "prepare_chat_params", fake_prepare_chat_params)

    # Make streaming yield one chunk so the generator executes.
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

    # Avoid depending on llama_index response parsing internals
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
    # consume one yield so generator body runs
    _ = next(gen)
    with pytest.raises(StopIteration):
        next(gen)

    assert deleted["called"] is True


@pytest.mark.asyncio
async def test_astream_chat_runs_cleanup(monkeypatch):
    async def fake_prepare_chat_params(*_args, **_kwargs):
        next_msg = "hello"
        chat_kwargs = {}
        file_api_names = ["file1"]
        return next_msg, chat_kwargs, file_api_names

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
