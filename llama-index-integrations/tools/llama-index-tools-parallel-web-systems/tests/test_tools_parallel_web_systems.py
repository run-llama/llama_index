"""Tests for Parallel AI tool spec."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llama_index.core.schema import Document
from llama_index.tools.parallel_web_systems import ParallelWebSystemsToolSpec


class TestParallelWebSystemsToolSpec:
    """Test suite for ParallelWebSystemsToolSpec."""

    def test_initialization(self) -> None:
        """Test that the tool spec initializes correctly."""
        api_key = "test-api-key"
        tool = ParallelWebSystemsToolSpec(api_key=api_key)

        assert tool.api_key == api_key
        assert tool.base_url == "https://api.parallel.ai"

    def test_initialization_with_custom_base_url(self) -> None:
        """Test initialization with a custom base URL."""
        api_key = "test-api-key"
        custom_url = "https://custom.api.com"
        tool = ParallelWebSystemsToolSpec(api_key=api_key, base_url=custom_url)

        assert tool.api_key == api_key
        assert tool.base_url == custom_url

    @pytest.mark.asyncio
    async def test_search_requires_objective_or_queries(self) -> None:
        """Test that search raises error when neither objective nor search_queries provided."""
        tool = ParallelWebSystemsToolSpec(api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            await tool.search()

        assert (
            "At least one of 'objective' or 'search_queries' must be provided"
            in str(exc_info.value)
        )

    @patch("llama_index.tools.parallel_web_systems.base.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_search_with_objective(self, mock_async_client: MagicMock) -> None:
        """Test successful search operation with objective."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "search_id": "search_123",
            "results": [
                {
                    "url": "https://example.com/1",
                    "title": "Test Title 1",
                    "publish_date": "2024-01-15",
                    "excerpts": ["Sample excerpt 1", "Sample excerpt 2"],
                },
                {
                    "url": "https://example.com/2",
                    "title": "Test Title 2",
                    "publish_date": "2024-01-16",
                    "excerpts": ["Another excerpt"],
                },
            ],
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client

        tool = ParallelWebSystemsToolSpec(api_key="test-key")
        results = await tool.search(
            objective="What was the GDP of France in 2023?", max_results=5
        )

        # Assertions
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        assert "Sample excerpt 1" in results[0].text
        assert "Sample excerpt 2" in results[0].text
        assert results[0].metadata["url"] == "https://example.com/1"
        assert results[0].metadata["title"] == "Test Title 1"
        assert results[0].metadata["search_id"] == "search_123"

    @patch("llama_index.tools.parallel_web_systems.base.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_search_with_queries(self, mock_async_client: MagicMock) -> None:
        """Test search with search_queries."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"search_id": "search_456", "results": []}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client

        tool = ParallelWebSystemsToolSpec(api_key="test-key")
        await tool.search(
            search_queries=["renewable energy 2024", "solar power"],
            max_results=10,
            mode="agentic",
        )

    @patch("llama_index.tools.parallel_web_systems.base.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_search_api_error(self, mock_async_client: MagicMock, capsys) -> None:
        """Test search handles API errors gracefully."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("API Error"))
        mock_async_client.return_value.__aenter__.return_value = mock_client

        tool = ParallelWebSystemsToolSpec(api_key="test-key")
        results = await tool.search(objective="test query")

        assert results == []

        captured = capsys.readouterr()
        assert "Error calling Parallel AI Search API" in captured.out

    @patch("llama_index.tools.parallel_web_systems.base.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_extract_success(self, mock_async_client: MagicMock) -> None:
        """Test successful extract operation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "extract_id": "extract_123",
            "results": [
                {
                    "url": "https://en.wikipedia.org/wiki/AI",
                    "title": "Artificial intelligence - Wikipedia",
                    "publish_date": "2024-01-15",
                    "excerpts": ["AI excerpt 1", "AI excerpt 2"],
                    "full_content": None,
                },
            ],
            "errors": [],
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client

        tool = ParallelWebSystemsToolSpec(api_key="test-key")
        results = await tool.extract(
            urls=["https://en.wikipedia.org/wiki/AI"],
            objective="What are the main applications of AI?",
        )

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert "AI excerpt 1" in results[0].text
        assert results[0].metadata["url"] == "https://en.wikipedia.org/wiki/AI"
        assert results[0].metadata["extract_id"] == "extract_123"

    @patch("llama_index.tools.parallel_web_systems.base.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_extract_with_full_content(
        self, mock_async_client: MagicMock
    ) -> None:
        """Test extract with full content enabled."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "extract_id": "extract_456",
            "results": [
                {
                    "url": "https://example.com/article",
                    "title": "Test Article",
                    "full_content": "# Full Article Content\n\nThis is the full content...",
                    "excerpts": [],
                },
            ],
            "errors": [],
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client

        tool = ParallelWebSystemsToolSpec(api_key="test-key")
        results = await tool.extract(
            urls=["https://example.com/article"],
            full_content=True,
            excerpts=False,
        )

        assert len(results) == 1
        assert "Full Article Content" in results[0].text

    @patch("llama_index.tools.parallel_web_systems.base.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_extract_handles_errors(self, mock_async_client: MagicMock) -> None:
        """Test extract handles URL errors gracefully"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "extract_id": "extract_789",
            "results": [],
            "errors": [
                {
                    "url": "https://invalid-url.com/",
                    "error_type": "fetch_failed",
                    "content": "Failed to fetch URL",
                },
            ],
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value = mock_client

        tool = ParallelWebSystemsToolSpec(api_key="test-key")
        results = await tool.extract(urls=["https://invalid-url.com/"])

        assert len(results) == 1
        assert "Error extracting content" in results[0].text
        assert results[0].metadata["error_type"] == "fetch_failed"

    @patch("llama_index.tools.parallel_web_systems.base.httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_extract_api_error(
        self, mock_async_client: MagicMock, capsys
    ) -> None:
        """Test extract handles API errors gracefully."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Network Error"))
        mock_async_client.return_value.__aenter__.return_value = mock_client

        tool = ParallelWebSystemsToolSpec(api_key="test-key")
        results = await tool.extract(urls=["https://example.com"])

        assert results == []
        captured = capsys.readouterr()
        assert "Error calling Parallel AI Extract API" in captured.out

    def test_spec_functions_list(self) -> None:
        """Test that spec_functions contains expected methods."""
        expected_functions = ["search", "extract"]
        assert ParallelWebSystemsToolSpec.spec_functions == expected_functions
