"""Tests for BaseExtractor retry and error-handling behaviour."""

from typing import Dict, List, Sequence
from unittest.mock import AsyncMock, patch

import pytest
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode, TextNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FailNTimesExtractor(BaseExtractor):
    """Extractor that raises on the first *fail_count* calls, then succeeds."""

    fail_count: int = 0
    _calls: int = 0

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        self._calls += 1
        if self._calls <= self.fail_count:
            raise RuntimeError(f"Simulated failure #{self._calls}")
        return [{"extracted": True} for _ in nodes]


class _AlwaysFailExtractor(BaseExtractor):
    """Extractor that always raises."""

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        raise RuntimeError("Permanent failure")


def _make_nodes(n: int = 2) -> List[TextNode]:
    return [TextNode(text=f"node-{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_raises_on_error():
    """With max_retries=0 and raise_on_error=True, errors propagate."""
    ext = _AlwaysFailExtractor(max_retries=0, raise_on_error=True)
    with pytest.raises(RuntimeError, match="Permanent failure"):
        await ext.aprocess_nodes(_make_nodes())


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failure():
    """Extractor fails once, succeeds on the second attempt."""
    ext = _FailNTimesExtractor(fail_count=1, max_retries=2, retry_backoff=0.01)
    nodes = _make_nodes()
    result = await ext.aprocess_nodes(nodes)
    assert len(result) == 2
    for node in result:
        assert node.metadata.get("extracted") is True


@pytest.mark.asyncio
async def test_skip_returns_empty_metadata():
    """With raise_on_error=False, failed extraction returns empty dicts."""
    ext = _AlwaysFailExtractor(max_retries=0, raise_on_error=False)
    nodes = _make_nodes(3)
    result = await ext.aprocess_nodes(nodes)
    assert len(result) == 3
    for node in result:
        # Metadata should be unchanged (empty dict merge is a no-op)
        assert "extracted" not in node.metadata


@pytest.mark.asyncio
async def test_retries_exhausted_then_raises():
    """After max_retries exhausted with raise_on_error=True, the error propagates."""
    ext = _AlwaysFailExtractor(max_retries=2, raise_on_error=True, retry_backoff=0.01)
    with pytest.raises(RuntimeError, match="Permanent failure"):
        await ext.aprocess_nodes(_make_nodes())


@pytest.mark.asyncio
async def test_retries_exhausted_then_skips():
    """After max_retries exhausted with raise_on_error=False, returns empty dicts."""
    ext = _AlwaysFailExtractor(max_retries=2, raise_on_error=False, retry_backoff=0.01)
    result = await ext.aprocess_nodes(_make_nodes())
    assert len(result) == 2
    for node in result:
        assert "extracted" not in node.metadata


@pytest.mark.asyncio
async def test_backoff_delays_applied():
    """Verify asyncio.sleep is called with exponential backoff delays."""
    ext = _AlwaysFailExtractor(
        max_retries=3,
        raise_on_error=False,
        retry_backoff=2.0,
    )
    with patch(
        "llama_index.core.extractors.interface.asyncio.sleep", new_callable=AsyncMock
    ) as mock_sleep:
        await ext.aprocess_nodes(_make_nodes())

    # 3 retries -> 3 sleep calls: 2*2^0=2.0, 2*2^1=4.0, 2*2^2=8.0
    assert mock_sleep.call_count == 3
    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == [2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_no_retry_single_call():
    """With max_retries=0, aextract is called exactly once."""
    ext = _FailNTimesExtractor(fail_count=0, max_retries=0)
    await ext.aprocess_nodes(_make_nodes())
    assert ext._calls == 1
