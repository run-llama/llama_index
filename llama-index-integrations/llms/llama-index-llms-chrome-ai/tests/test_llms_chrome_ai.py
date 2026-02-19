"""Unit tests for ChromeAI LLM integration.

All tests mock Playwright so that a real Chrome browser is not required.
The key technique is patching sys.modules so that the local imports inside
each method (``from playwright.sync_api import sync_playwright`` etc.) pick
up our fakes rather than the real Playwright library.
"""

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.core.base.llms.types import (
    ChatMessage,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.llms.chrome_ai import ChromeAI
from llama_index.llms.chrome_ai.base import _extract_prompts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(**kwargs: Any) -> ChromeAI:
    return ChromeAI(temperature=0.5, top_k=3, headless=True, timeout=10.0, **kwargs)


def _user_messages(user: str, system: str = "") -> list:
    msgs = []
    if system:
        msgs.append(ChatMessage(role=MessageRole.SYSTEM, content=system))
    msgs.append(ChatMessage(role=MessageRole.USER, content=user))
    return msgs


def _sync_pw_modules(eval_return: Any = None, eval_side_effect: Any = None) -> tuple:
    """Build a mocked ``playwright.sync_api`` module + exposed page mock.

    Returns ``(mock_module, page)`` where *mock_module* is suitable for
    ``patch.dict(sys.modules, {'playwright.sync_api': mock_module})`` and
    *page* is the MagicMock returned by ``browser.new_page()``.
    """
    page = MagicMock()
    if eval_side_effect is not None:
        page.evaluate.side_effect = eval_side_effect
    elif eval_return is not None:
        page.evaluate.return_value = eval_return

    browser = MagicMock()
    browser.new_page.return_value = page

    pw = MagicMock()
    pw.chromium.launch.return_value = browser

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=pw)
    ctx.__exit__ = MagicMock(return_value=False)

    mod = MagicMock()
    mod.sync_playwright.return_value = ctx
    return mod, page


def _async_pw_modules(eval_return: Any = None, eval_side_effect: Any = None) -> tuple:
    """Build a mocked ``playwright.async_api`` module + exposed page mock."""
    page = AsyncMock()
    if eval_side_effect is not None:
        page.evaluate.side_effect = eval_side_effect
    elif eval_return is not None:
        page.evaluate.return_value = eval_return

    browser = AsyncMock()
    browser.new_page.return_value = page

    pw = AsyncMock()
    pw.chromium.launch.return_value = browser

    ctx = AsyncMock()
    ctx.__aenter__.return_value = pw
    ctx.__aexit__.return_value = False

    mod = MagicMock()
    mod.async_playwright.return_value = ctx
    return mod, page


# ---------------------------------------------------------------------------
# _extract_prompts
# ---------------------------------------------------------------------------


def test_extract_prompts_basic() -> None:
    system, user = _extract_prompts(_user_messages("Hello", system="Be concise."))
    assert system == "Be concise."
    assert user == "Hello"


def test_extract_prompts_no_system() -> None:
    system, user = _extract_prompts(_user_messages("Hi"))
    assert system == ""
    assert user == "Hi"


def test_extract_prompts_multiple_system_joined() -> None:
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="Part 1."),
        ChatMessage(role=MessageRole.SYSTEM, content="Part 2."),
        ChatMessage(role=MessageRole.USER, content="Question?"),
    ]
    system, user = _extract_prompts(messages)
    assert system == "Part 1.\nPart 2."
    assert user == "Question?"


def test_extract_prompts_last_user_message_wins() -> None:
    messages = [
        ChatMessage(role=MessageRole.USER, content="First."),
        ChatMessage(role=MessageRole.USER, content="Second."),
    ]
    _, user = _extract_prompts(messages)
    assert user == "Second."


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_defaults() -> None:
    meta = _make_llm().metadata
    assert isinstance(meta, LLMMetadata)
    assert meta.model_name == "gemini-nano"
    assert meta.is_chat_model is True
    assert meta.context_window == 6144


def test_metadata_custom_context_window() -> None:
    llm = ChromeAI(context_window=2048)
    assert llm.metadata.context_window == 2048


# ---------------------------------------------------------------------------
# _launch_options
# ---------------------------------------------------------------------------


def test_launch_options_uses_chrome_channel_by_default() -> None:
    opts = _make_llm()._launch_options()
    assert opts["channel"] == "chrome"
    assert "executable_path" not in opts


def test_launch_options_custom_executable_path() -> None:
    opts = _make_llm(chrome_executable_path="/usr/bin/google-chrome")._launch_options()
    assert opts["executable_path"] == "/usr/bin/google-chrome"
    assert "channel" not in opts


def test_launch_options_headless_false() -> None:
    assert _make_llm(headless=False)._launch_options()["headless"] is False


def test_launch_options_headless_true() -> None:
    assert _make_llm(headless=True)._launch_options()["headless"] is True


def test_launch_options_additional_args_forwarded() -> None:
    flag = "--enable-features=PromptAPIForGeminiNano"
    opts = _make_llm(additional_launch_args=[flag])._launch_options()
    assert flag in opts["args"]


# ---------------------------------------------------------------------------
# _js_params
# ---------------------------------------------------------------------------


def test_js_params_values_passed_through() -> None:
    params = _make_llm()._js_params("sys", "usr")
    assert params == {
        "systemPrompt": "sys",
        "userPrompt": "usr",
        "temperature": 0.5,
        "topK": 3,
    }


def test_js_params_none_when_not_set() -> None:
    params = ChromeAI()._js_params("", "hi")
    assert params["temperature"] is None
    assert params["topK"] is None


# ---------------------------------------------------------------------------
# chat() — sync, mocked Playwright
# ---------------------------------------------------------------------------


def test_chat_returns_assistant_response() -> None:
    mod, page = _sync_pw_modules(eval_return="The capital is Paris.")
    llm = _make_llm()
    messages = _user_messages("Capital of France?", system="Answer briefly.")

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        response = llm.chat(messages)

    assert response.message.content == "The capital is Paris."
    assert response.message.role == MessageRole.ASSISTANT
    assert response.raw == {"text": "The capital is Paris."}


def test_chat_passes_correct_js_params() -> None:
    mod, page = _sync_pw_modules(eval_return="ok")
    llm = _make_llm()
    messages = _user_messages("Hello!", system="You are helpful.")

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        llm.chat(messages)

    call_args = page.evaluate.call_args
    params = call_args[0][1]  # second positional arg to evaluate()
    assert params["systemPrompt"] == "You are helpful."
    assert params["userPrompt"] == "Hello!"
    assert params["temperature"] == 0.5
    assert params["topK"] == 3


def test_chat_closes_browser_on_success() -> None:
    mod, page = _sync_pw_modules(eval_return="done")
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        llm.chat(_user_messages("Hi"))

    # browser.close() should have been called
    ctx = mod.sync_playwright.return_value
    browser = ctx.__enter__.return_value.chromium.launch.return_value
    browser.close.assert_called_once()


# ---------------------------------------------------------------------------
# complete() — delegates to chat()
# ---------------------------------------------------------------------------


def test_complete_delegates_to_chat() -> None:
    llm = _make_llm()
    fake = MagicMock()
    fake.message.content = "answer"

    with patch.object(llm, "chat", return_value=fake) as mock_chat:
        result = llm.complete("Some prompt")
        mock_chat.assert_called_once()

    assert isinstance(result, CompletionResponse)
    assert result.text == "answer"


# ---------------------------------------------------------------------------
# stream_chat() — sync streaming, mocked Playwright
# ---------------------------------------------------------------------------


def _make_streaming_sync_modules(chunks: list) -> tuple:
    """Return (mod, page) where page.evaluate simulates JS streaming."""
    captured: dict = {}

    def fake_expose_function(name: str, callback: Any) -> None:
        if name == "chromePyChunk":
            captured["cb"] = callback

    def fake_evaluate(js_code: str, params: Any) -> None:
        cb = captured["cb"]
        for chunk in chunks:
            cb(chunk)
        cb(None)  # sentinel

    mod, page = _sync_pw_modules()
    page.expose_function.side_effect = fake_expose_function
    page.evaluate.side_effect = fake_evaluate
    return mod, page


def test_stream_chat_yields_all_deltas() -> None:
    mod, _ = _make_streaming_sync_modules(["Hello", " there", "!"])
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        results = list(llm.stream_chat(_user_messages("Greet.")))

    assert len(results) == 3
    assert results[0].delta == "Hello"
    assert results[1].delta == " there"
    assert results[2].delta == "!"


def test_stream_chat_content_accumulates() -> None:
    mod, _ = _make_streaming_sync_modules(["One", " Two", " Three"])
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        results = list(llm.stream_chat(_user_messages("Count.")))

    assert results[0].message.content == "One"
    assert results[1].message.content == "One Two"
    assert results[2].message.content == "One Two Three"


def test_stream_chat_all_roles_are_assistant() -> None:
    mod, _ = _make_streaming_sync_modules(["A", "B"])
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        results = list(llm.stream_chat(_user_messages("Go.")))

    assert all(r.message.role == MessageRole.ASSISTANT for r in results)


def test_stream_chat_empty_stream_yields_nothing() -> None:
    mod, _ = _make_streaming_sync_modules([])  # only sentinel
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        results = list(llm.stream_chat(_user_messages("Go.")))

    assert results == []


def test_stream_chat_propagates_playwright_error() -> None:
    captured: dict = {}

    def fake_expose(name: str, callback: Any) -> None:
        captured["cb"] = callback

    def fake_evaluate(js_code: str, params: Any) -> None:
        raise RuntimeError("Chrome AI unavailable")

    mod, page = _sync_pw_modules()
    page.expose_function.side_effect = fake_expose
    page.evaluate.side_effect = fake_evaluate

    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        with pytest.raises(RuntimeError, match="Chrome AI unavailable"):
            list(llm.stream_chat(_user_messages("Hi.")))


# ---------------------------------------------------------------------------
# stream_complete() — delegates to stream_chat()
# ---------------------------------------------------------------------------


def test_stream_complete_delegates_to_stream_chat() -> None:
    llm = _make_llm()
    from llama_index.core.base.llms.types import ChatResponse

    def fake_stream(messages, **kwargs):
        yield ChatResponse(
            message=ChatMessage(content="streamed", role=MessageRole.ASSISTANT),
            delta="streamed",
        )

    with patch.object(llm, "stream_chat", side_effect=fake_stream):
        results = list(llm.stream_complete("Prompt."))

    assert len(results) == 1
    assert results[0].text == "streamed"


# ---------------------------------------------------------------------------
# check_availability()
# ---------------------------------------------------------------------------


def test_check_availability_returns_available() -> None:
    mod, page = _sync_pw_modules(eval_return="available")
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        result = llm.check_availability()

    assert result == "available"
    page.evaluate.assert_called_once()


def test_check_availability_returns_downloadable() -> None:
    mod, _ = _sync_pw_modules(eval_return="downloadable")
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.sync_api": mod}):
        result = llm.check_availability()

    assert result == "downloadable"


# ---------------------------------------------------------------------------
# achat() — async, mocked Playwright
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_achat_returns_assistant_response() -> None:
    mod, page = _async_pw_modules(eval_return="Bonjour!")
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.async_api": mod}):
        response = await llm.achat(_user_messages("Say hello in French."))

    assert response.message.content == "Bonjour!"
    assert response.message.role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_achat_passes_correct_params() -> None:
    mod, page = _async_pw_modules(eval_return="ok")
    llm = _make_llm()
    messages = _user_messages("Test?", system="Be brief.")

    with patch.dict(sys.modules, {"playwright.async_api": mod}):
        await llm.achat(messages)

    call_args = page.evaluate.call_args
    params = call_args[0][1]
    assert params["systemPrompt"] == "Be brief."
    assert params["userPrompt"] == "Test?"


@pytest.mark.asyncio
async def test_achat_closes_browser() -> None:
    mod, _ = _async_pw_modules(eval_return="done")
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.async_api": mod}):
        await llm.achat(_user_messages("Hi"))

    ctx = mod.async_playwright.return_value
    browser = ctx.__aenter__.return_value.chromium.launch.return_value
    browser.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# acomplete() — delegates to achat()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acomplete_delegates_to_achat() -> None:
    llm = _make_llm()
    fake = MagicMock()
    fake.message.content = "async answer"

    with patch.object(llm, "achat", AsyncMock(return_value=fake)) as mock_achat:
        result = await llm.acomplete("Async prompt.")
        mock_achat.assert_called_once()

    assert isinstance(result, CompletionResponse)
    assert result.text == "async answer"


# ---------------------------------------------------------------------------
# astream_chat() — async streaming, mocked Playwright
# ---------------------------------------------------------------------------


def _make_streaming_async_modules(chunks: list) -> tuple:
    """Return (mod, page) where page.evaluate simulates async JS streaming."""
    captured: dict = {}

    def fake_expose_function(name: str, callback: Any) -> None:
        if name == "chromePyChunk":
            captured["cb"] = callback

    async def fake_evaluate(js_code: str, params: Any) -> None:
        cb = captured["cb"]
        for chunk in chunks:
            cb(chunk)  # on_chunk uses put_nowait — no await needed
        cb(None)  # sentinel

    mod, page = _async_pw_modules()
    page.expose_function.side_effect = fake_expose_function
    page.evaluate.side_effect = fake_evaluate
    return mod, page


@pytest.mark.asyncio
async def test_astream_chat_yields_all_deltas() -> None:
    mod, _ = _make_streaming_async_modules(["Async", " chunk", "!"])
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.async_api": mod}):
        results = []
        async for chunk in llm.astream_chat(_user_messages("Stream.")):
            results.append(chunk)

    assert len(results) == 3
    assert results[0].delta == "Async"
    assert results[1].delta == " chunk"
    assert results[2].delta == "!"


@pytest.mark.asyncio
async def test_astream_chat_content_accumulates() -> None:
    mod, _ = _make_streaming_async_modules(["A", "B", "C"])
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.async_api": mod}):
        results = []
        async for chunk in llm.astream_chat(_user_messages("Go.")):
            results.append(chunk)

    assert results[0].message.content == "A"
    assert results[1].message.content == "AB"
    assert results[2].message.content == "ABC"


@pytest.mark.asyncio
async def test_astream_chat_empty_stream_yields_nothing() -> None:
    mod, _ = _make_streaming_async_modules([])
    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.async_api": mod}):
        results = []
        async for chunk in llm.astream_chat(_user_messages("Empty.")):
            results.append(chunk)

    assert results == []


@pytest.mark.asyncio
async def test_astream_chat_propagates_error() -> None:
    captured: dict = {}

    def fake_expose(name: str, callback: Any) -> None:
        captured["cb"] = callback

    async def fake_evaluate(js_code: str, params: Any) -> None:
        raise RuntimeError("Async Chrome AI error")

    mod, page = _async_pw_modules()
    page.expose_function.side_effect = fake_expose
    page.evaluate.side_effect = fake_evaluate

    llm = _make_llm()

    with patch.dict(sys.modules, {"playwright.async_api": mod}):
        # The error should propagate out (not just time out) because the
        # run_streaming task puts a None sentinel before re-raising.
        results = []
        with pytest.raises((RuntimeError, Exception)):
            async for chunk in llm.astream_chat(_user_messages("Fail.")):
                results.append(chunk)

    assert results == []  # no chunks before the error
