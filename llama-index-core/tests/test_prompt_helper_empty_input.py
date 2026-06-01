"""
Regression tests for ZeroDivisionError when PromptHelper / ChatPromptHelper
are called with empty sequences.

Root cause: ``_get_available_chunk_size`` computes
``available_context_size // num_chunks`` where ``num_chunks = len(sequence)``.
Passing an empty list yields integer division by zero.

Fixed by early-return guards in:
- ``PromptHelper.truncate``
- ``ChatPromptHelper.atruncate``  (sync ``.truncate`` wrapper covered transitively)
- ``ChatPromptHelper.arepack``    (sync ``.repack`` wrapper covered transitively)
"""

import asyncio

import pytest

from llama_index.core.indices.prompt_helper import (
    ChatPromptHelper,
    PromptHelper,
)
from llama_index.core.prompts import PromptTemplate


@pytest.fixture
def prompt() -> PromptTemplate:
    return PromptTemplate("Answer: {query_str}")


# ---------------------------------------------------------------------------
# PromptHelper.truncate
# ---------------------------------------------------------------------------


def test_prompt_helper_truncate_empty_returns_empty(prompt: PromptTemplate) -> None:
    """``PromptHelper.truncate(prompt, [])`` must return ``[]``, not raise."""
    ph = PromptHelper(context_window=4096, num_output=256)
    assert ph.truncate(prompt, []) == []


# ---------------------------------------------------------------------------
# ChatPromptHelper.atruncate / .truncate
# ---------------------------------------------------------------------------


def test_chat_prompt_helper_atruncate_empty_returns_empty(
    prompt: PromptTemplate,
) -> None:
    """``ChatPromptHelper.atruncate(prompt, [])`` must return ``[]``, not raise."""
    ch = ChatPromptHelper(context_window=4096, num_output=256)
    assert asyncio.run(ch.atruncate(prompt, [])) == []


def test_chat_prompt_helper_truncate_empty_returns_empty(
    prompt: PromptTemplate,
) -> None:
    """Sync wrapper ``ChatPromptHelper.truncate(prompt, [])`` must return ``[]``."""
    ch = ChatPromptHelper(context_window=4096, num_output=256)
    assert ch.truncate(prompt, []) == []


# ---------------------------------------------------------------------------
# ChatPromptHelper.arepack / .repack
# ---------------------------------------------------------------------------


def test_chat_prompt_helper_arepack_empty_returns_empty(
    prompt: PromptTemplate,
) -> None:
    """``ChatPromptHelper.arepack(prompt, [])`` must return ``[]``, not raise."""
    ch = ChatPromptHelper(context_window=4096, num_output=256)
    assert asyncio.run(ch.arepack(prompt, [])) == []


def test_chat_prompt_helper_repack_empty_returns_empty(
    prompt: PromptTemplate,
) -> None:
    """Sync wrapper ``ChatPromptHelper.repack(prompt, [])`` must return ``[]``."""
    ch = ChatPromptHelper(context_window=4096, num_output=256)
    assert ch.repack(prompt, []) == []
