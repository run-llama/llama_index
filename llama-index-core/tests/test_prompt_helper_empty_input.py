"""
Regression tests for ZeroDivisionError when PromptHelper / ChatPromptHelper
are called with empty sequences.

Root cause: `_get_available_chunk_size` computes
`available_context_size // num_chunks` where `num_chunks = len(sequence)`.
Passing an empty list yields integer division by zero.

Fixed by adding early-return guards in:
  - PromptHelper.truncate
  - ChatPromptHelper.atruncate  (sync .truncate wrapper is also covered)
  - ChatPromptHelper.arepack    (sync .repack wrapper is also covered)
"""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mock_counter():
    m = MagicMock()
    m.get_string_tokens.return_value = 10
    m.estimate_tokens_in_messages.return_value = 10
    m.estimate_tokens_in_tools.return_value = 0
    return m


def _make_ph(mock_counter):
    with patch(
        "llama_index.core.indices.prompt_helper.TokenCounter",
        return_value=mock_counter,
    ):
        from llama_index.core.indices.prompt_helper import PromptHelper

        ph = PromptHelper(context_window=4096, num_output=256)
        ph._token_counter = mock_counter
        return ph


def _make_ch(mock_counter):
    with patch(
        "llama_index.core.indices.prompt_helper.TokenCounter",
        return_value=mock_counter,
    ):
        from llama_index.core.indices.prompt_helper import ChatPromptHelper

        ch = ChatPromptHelper(context_window=4096, num_output=256)
        ch._token_counter = mock_counter
        return ch


@pytest.fixture
def prompt():
    from llama_index.core.prompts import PromptTemplate

    return PromptTemplate("Answer: {query_str}")


# ---------------------------------------------------------------------------
# PromptHelper.truncate
# ---------------------------------------------------------------------------


def test_prompt_helper_truncate_empty_list_returns_empty(prompt):
    """PromptHelper.truncate(prompt, []) must return [] not raise ZeroDivisionError."""
    mc = _mock_counter()
    ph = _make_ph(mc)
    result = ph.truncate(prompt, [])
    assert result == []


def test_prompt_helper_truncate_nonempty_still_works(prompt):
    """PromptHelper.truncate with a non-empty list must still work."""
    mc = _mock_counter()
    ph = _make_ph(mc)
    result = ph.truncate(prompt, ["hello world"])
    assert isinstance(result, list)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# ChatPromptHelper.atruncate / truncate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_prompt_helper_atruncate_empty_list_returns_empty(prompt):
    """ChatPromptHelper.atruncate(prompt, []) must return [] not raise ZeroDivisionError."""
    mc = _mock_counter()
    ch = _make_ch(mc)
    result = await ch.atruncate(prompt, [])
    assert list(result) == []


def test_chat_prompt_helper_truncate_empty_list_returns_empty(prompt):
    """ChatPromptHelper.truncate(prompt, []) (sync wrapper) must return []."""
    mc = _mock_counter()
    ch = _make_ch(mc)
    result = ch.truncate(prompt, [])
    assert list(result) == []


# ---------------------------------------------------------------------------
# ChatPromptHelper.arepack / repack
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_prompt_helper_arepack_empty_list_returns_empty(prompt):
    """ChatPromptHelper.arepack(prompt, []) must return [] not raise ZeroDivisionError."""
    mc = _mock_counter()
    ch = _make_ch(mc)
    result = await ch.arepack(prompt, [])
    assert result == []


def test_chat_prompt_helper_repack_empty_list_returns_empty(prompt):
    """ChatPromptHelper.repack(prompt, []) (sync wrapper) must return []."""
    mc = _mock_counter()
    ch = _make_ch(mc)
    result = ch.repack(prompt, [])
    assert result == []
