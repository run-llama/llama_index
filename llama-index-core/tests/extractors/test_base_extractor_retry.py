import pytest
import asyncio
from typing import ClassVar, List, Sequence

from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import TextNode


class MockFailingExtractor(BaseExtractor):
    """Mock extractor that fails a specified number of times."""

    fail_count: ClassVar[int] = 0
    max_fails: ClassVar[int] = 1

    async def aextract(self, nodes: Sequence[TextNode]) -> List[dict]:
        MockFailingExtractor.fail_count += 1
        if MockFailingExtractor.fail_count <= MockFailingExtractor.max_fails:
            raise Exception("Simulated extraction failure")
        return [{"test": f"value_{i}"} for i in range(len(nodes))]


@pytest.fixture
def reset_fail_count():
    MockFailingExtractor.fail_count = 0
    yield
    MockFailingExtractor.fail_count = 0


class TestBaseExtractorRetry:
    @pytest.mark.asyncio
    async def test_no_retry_raises_immediately(self, reset_fail_count):
        """Test that max_retries=0 raises exception immediately."""
        MockFailingExtractor.fail_count = 0
        MockFailingExtractor.max_fails = 5

        extractor = MockFailingExtractor(
            max_retries=0,
            on_extraction_error="raise",
        )
        nodes = [TextNode(text="test")]

        with pytest.raises(Exception, match="Simulated extraction failure"):
            await extractor._aextract_with_retry(nodes)

        assert MockFailingExtractor.fail_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self, reset_fail_count):
        """Test that retry succeeds after initial failures."""
        MockFailingExtractor.fail_count = 0
        MockFailingExtractor.max_fails = 2

        extractor = MockFailingExtractor(
            max_retries=3,
            retry_backoff=0.01,
            on_extraction_error="raise",
        )
        nodes = [TextNode(text="test")]

        result = await extractor._aextract_with_retry(nodes)

        assert result == [{"test": "value_0"}]
        assert MockFailingExtractor.fail_count == 3

    @pytest.mark.asyncio
    async def test_skip_on_error_returns_empty_metadata(self, reset_fail_count):
        """Test that on_extraction_error='skip' returns empty dicts."""
        MockFailingExtractor.fail_count = 0
        MockFailingExtractor.max_fails = 10

        extractor = MockFailingExtractor(
            max_retries=2,
            retry_backoff=0.01,
            on_extraction_error="skip",
        )
        nodes = [TextNode(text="test1"), TextNode(text="test2")]

        result = await extractor._aextract_with_retry(nodes)

        assert result == [{}, {}]
        assert MockFailingExtractor.fail_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self, reset_fail_count):
        """Test that exception is raised after all retries exhausted."""
        MockFailingExtractor.fail_count = 0
        MockFailingExtractor.max_fails = 10

        extractor = MockFailingExtractor(
            max_retries=2,
            retry_backoff=0.01,
            on_extraction_error="raise",
        )
        nodes = [TextNode(text="test")]

        with pytest.raises(RuntimeError, match="Extraction failed after 3 attempts"):
            await extractor._aextract_with_retry(nodes)

        assert MockFailingExtractor.fail_count == 3

    @pytest.mark.asyncio
    async def test_aprocess_nodes_with_retry(self, reset_fail_count):
        """Test that aprocess_nodes uses retry logic correctly."""
        MockFailingExtractor.fail_count = 0
        MockFailingExtractor.max_fails = 10  # Always fails

        extractor = MockFailingExtractor(
            max_retries=2,
            retry_backoff=0.01,
            on_extraction_error="skip",
        )
        nodes = [TextNode(text="test1"), TextNode(text="test2")]

        result_nodes = await extractor.aprocess_nodes(nodes)

        # Should have retried max_retries+1 times and then skipped
        assert MockFailingExtractor.fail_count == 3
        # Nodes should have empty metadata due to skip
        assert result_nodes[0].metadata == {}
        assert result_nodes[1].metadata == {}

    @pytest.mark.asyncio
    async def test_empty_nodes(self, reset_fail_count):
        """Test handling of empty node list."""
        MockFailingExtractor.fail_count = 0
        MockFailingExtractor.max_fails = 0

        extractor = MockFailingExtractor(
            max_retries=0,
            on_extraction_error="raise",
        )
        nodes = []

        result = await extractor._aextract_with_retry(nodes)

        assert result == []
        assert MockFailingExtractor.fail_count == 1

    @pytest.mark.asyncio
    async def test_backward_compatible_defaults(self):
        """Test that defaults preserve existing behavior."""
        extractor = MockFailingExtractor()

        # Default max_retries=0 means no retry
        assert extractor.max_retries == 0
        assert extractor.retry_backoff == 1.0
        assert extractor.on_extraction_error == "raise"
