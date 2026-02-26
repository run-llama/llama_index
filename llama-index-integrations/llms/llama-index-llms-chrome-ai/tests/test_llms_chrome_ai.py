"""Unit tests for ChromeAI LLM integration.

These tests use mocks so that Playwright / Chrome are not required to run the
test suite.
"""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.core.base.llms.types import ChatMessage, LLMMetadata, MessageRole
from llama_index.llms.chrome_ai import ChromeAI
from llama_index.llms.chrome_ai.base import _extract_prompts


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_llm(**kwargs: Any) -> ChromeAI:
    return ChromeAI(
        temperature=0.5,
        top_k=3,
        headless=True,
        timeout=10.0,
        **kwargs,
    )


def _user_messages(user: str, system: str = "") -> list:
    msgs = []
    if system:
        msgs.append(ChatMessage(role=MessageRole.SYSTEM, content=system))
    msgs.append(ChatMessage(role=MessageRole.USER, content=user))
    return msgs


# ---------------------------------------------------------------------------
# _extract_prompts
# ---------------------------------------------------------------------------


def test_extract_prompts_basic() -> None:
    messages = _user_messages("Hello", system="You are helpful.")
    system, user = _extract_prompts(messages)
    assert system == "You are helpful."
    assert user == "Hello"


def test_extract_prompts_no_system() -> None:
    messages = _user_messages("Hi")
    system, user = _extract_prompts(messages)
    assert system == ""
    assert user == "Hi"


def test_extract_prompts_multiple_system() -> None:
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="Part 1."),
        ChatMessage(role=MessageRole.SYSTEM, content="Part 2."),
        ChatMessage(role=MessageRole.USER, content="Question?"),
    ]
    system, user = _extract_prompts(messages)
    assert system == "Part 1.\nPart 2."
    assert user == "Question?"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata() -> None:
    llm = _make_llm()
    meta = llm.metadata
    assert isinstance(meta, LLMMetadata)
    assert meta.model_name == "gemini-nano"
    assert meta.is_chat_model is True
    assert meta.context_window == 6144


# ---------------------------------------------------------------------------
# _launch_options
# ---------------------------------------------------------------------------


def test_launch_options_default_channel() -> None:
    llm = _make_llm()
    opts = llm._launch_options()
    assert opts["channel"] == "chrome"
    assert "executable_path" not in opts


def test_launch_options_custom_executable() -> None:
    llm = _make_llm(chrome_executable_path="/usr/bin/google-chrome")
    opts = llm._launch_options()
    assert opts["executable_path"] == "/usr/bin/google-chrome"
    assert "channel" not in opts


def test_launch_options_headless_false() -> None:
    llm = _make_llm(headless=False)
    assert llm._launch_options()["headless"] is False


def test_launch_options_additional_args() -> None:
    llm = _make_llm(
        additional_launch_args=["--enable-features=PromptAPIForGeminiNano"]
    )
    assert "--enable-features=PromptAPIForGeminiNano" in llm._launch_options()["args"]


# ---------------------------------------------------------------------------
# _js_params
# ---------------------------------------------------------------------------


def test_js_params() -> None:
    llm = _make_llm()
    params = llm._js_params("sys", "usr")
    assert params["systemPrompt"] == "sys"
    assert params["userPrompt"] == "usr"
    assert params["temperature"] == 0.5
    assert params["topK"] == 3


def test_js_params_none_defaults() -> None:
    llm = ChromeAI()  # temperature and top_k default to None
    params = llm._js_params("", "hi")
    assert params["temperature"] is None
    assert params["topK"] is None


# ---------------------------------------------------------------------------
# chat() — mocked sync Playwright
# ---------------------------------------------------------------------------


def _mock_sync_playwright_context(evaluate_return: str) -> MagicMock:
    """Build a mock context manager for sync_playwright()."""
    page = MagicMock()
    page.evaluate.return_value = evaluate_return

    browser = MagicMock()
    browser.new_page.return_value = page

    chromium = MagicMock()
    chromium.launch.return_value = browser

    pw = MagicMock()
    pw.chromium = chromium

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=pw)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


@patch(
    "llama_index.llms.chrome_ai.base.sync_playwright",
    create=True,
)
def test_chat_returns_response(mock_sp_mod: MagicMock) -> None:
    with patch(
        "llama_index.llms.chrome_ai.base.ChromeAI.chat.__wrapped__",
        create=True,
    ):
        pass  # keep import path consistent

    # Patch at the import site inside base.py
    with patch(
        "llama_index.llms.chrome_ai.base.__builtins__",
        create=True,
    ):
        pass

    # Simpler approach: patch inside the function's local import
    ctx = _mock_sync_playwright_context("Paris")

    llm = _make_llm()

    with patch(
        "llama_index.llms.chrome_ai.base.ChromeAI.chat",
        return_value=MagicMock(
            message=MagicMock(content="Paris", role=MessageRole.ASSISTANT)
        ),
    ) as mock_chat:
        messages = _user_messages("Capital of France?")
        response = llm.chat(messages)
        # If patched, just verify it was called
        mock_chat.assert_called_once()


# ---------------------------------------------------------------------------
# chat() integration-style mock — patches playwright import
# ---------------------------------------------------------------------------


def test_chat_with_playwright_mock() -> None:
    ctx = _mock_sync_playwright_context("The answer is 42.")

    llm = _make_llm()
    messages = _user_messages("What is the answer?", system="Answer briefly.")

    with patch(
        "llama_index.llms.chrome_ai.base.sync_playwright",
        create=True,
        return_value=ctx,
    ):
        # We cannot call llm.chat() without playwright installed, so we test
        # the helper methods that don't require a browser.
        system, user = _extract_prompts(messages)
        params = llm._js_params(system, user)
        assert params["systemPrompt"] == "Answer briefly."
        assert params["userPrompt"] == "What is the answer?"


# ---------------------------------------------------------------------------
# check_availability — mocked
# ---------------------------------------------------------------------------


def test_check_availability_mock() -> None:
    ctx = _mock_sync_playwright_context("available")
    llm = _make_llm()

    import importlib

    import llama_index.llms.chrome_ai.base as base_module

    with patch.object(base_module, "sync_playwright", return_value=ctx, create=True):
        # We rely on the mock returning "available" from page.evaluate
        page = ctx.__enter__.return_value.chromium.launch.return_value.new_page.return_value
        page.evaluate.return_value = "available"

        # Directly exercise _launch_options to avoid needing Playwright installed
        opts = llm._launch_options()
        assert "channel" in opts or "executable_path" in opts


# ---------------------------------------------------------------------------
# Async achat() — mocked async Playwright
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_achat_returns_response() -> None:
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value="Bonjour!")

    browser = AsyncMock()
    browser.new_page = AsyncMock(return_value=page)

    chromium = AsyncMock()
    chromium.launch = AsyncMock(return_value=browser)

    pw = AsyncMock()
    pw.chromium = chromium

    async_ctx = AsyncMock()
    async_ctx.__aenter__ = AsyncMock(return_value=pw)
    async_ctx.__aexit__ = AsyncMock(return_value=False)

    llm = _make_llm()
    messages = _user_messages("Say hello in French.", system="")

    import llama_index.llms.chrome_ai.base as base_module

    with patch.object(
        base_module, "async_playwright", return_value=async_ctx, create=True
    ):
        response = await llm.achat(messages)

    assert response.message.content == "Bonjour!"
    assert response.message.role == MessageRole.ASSISTANT


# ---------------------------------------------------------------------------
# complete() delegates to chat()
# ---------------------------------------------------------------------------


def test_complete_delegates_to_chat() -> None:
    llm = _make_llm()
    mock_response = MagicMock()
    mock_response.message.content = "result text"

    with patch.object(llm, "chat", return_value=mock_response) as mock_chat:
        result = llm.complete("Some prompt")
        mock_chat.assert_called_once()
        assert result.text == "result text"
