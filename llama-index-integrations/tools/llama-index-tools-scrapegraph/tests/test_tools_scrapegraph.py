"""Unit tests for ScrapeGraphAI tool specification."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field

from llama_index.tools.scrapegraph import ScrapegraphToolSpec


class TestSchema(BaseModel):
    """Test schema for scraping operations."""

    title: str = Field(description="Title of the content")
    description: str = Field(description="Description of the content")


class TestProductSchema(BaseModel):
    """Test schema for product information."""

    name: str = Field(description="Product name")
    price: str = Field(description="Product price", default="N/A")


@pytest.fixture()
def tool_spec_with_api_key():
    """Create a ScrapegraphToolSpec instance with explicit API key for testing."""
    with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        spec = ScrapegraphToolSpec(api_key="test-api-key")
        spec.client = mock_client
        yield spec, mock_client


@pytest.fixture()
def tool_spec_from_env():
    """Create a ScrapegraphToolSpec instance from environment for testing."""
    with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.from_env.return_value = mock_client
        spec = ScrapegraphToolSpec()
        spec.client = mock_client
        yield spec, mock_client


class TestScrapegraphToolSpecInitialization:
    """Test initialization of ScrapegraphToolSpec."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
            ScrapegraphToolSpec(api_key="test-key")
            mock_client_class.assert_called_once_with(api_key="test-key")

    def test_init_from_env(self):
        """Test initialization from environment."""
        with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
            ScrapegraphToolSpec()
            mock_client_class.from_env.assert_called_once()


class TestSmartScraper:
    """Test SmartScraper functionality."""

    def test_smartscraper_with_schema(self, tool_spec_with_api_key):
        """Test SmartScraper with Pydantic schema."""
        tool_spec, mock_client = tool_spec_with_api_key

        # Test data
        prompt = "Extract product information"
        url = "https://example.com"
        schema = TestSchema
        expected_response = {"title": "Test Product", "description": "Test Description"}

        # Configure mock
        mock_client.smartscraper.return_value = expected_response

        # Execute test
        response = tool_spec.scrapegraph_smartscraper(
            prompt=prompt, url=url, schema=schema
        )

        # Verify
        mock_client.smartscraper.assert_called_once_with(
            website_url=url, user_prompt=prompt, output_schema=schema
        )
        assert response == expected_response

    def test_smartscraper_without_schema(self, tool_spec_from_env):
        """Test SmartScraper without schema."""
        tool_spec, mock_client = tool_spec_from_env

        prompt = "Extract general information"
        url = "https://example.com"
        expected_response = {"content": "Extracted content"}

        mock_client.smartscraper.return_value = expected_response

        response = tool_spec.scrapegraph_smartscraper(prompt=prompt, url=url)

        mock_client.smartscraper.assert_called_once_with(
            website_url=url, user_prompt=prompt, output_schema=None
        )
        assert response == expected_response

    def test_smartscraper_with_kwargs(self, tool_spec_with_api_key):
        """Test SmartScraper with additional kwargs."""
        tool_spec, mock_client = tool_spec_with_api_key

        prompt = "Extract data"
        url = "https://example.com"
        expected_response = {"data": "test"}

        mock_client.smartscraper.return_value = expected_response

        response = tool_spec.scrapegraph_smartscraper(
            prompt=prompt, url=url, timeout=30, custom_param="value"
        )

        mock_client.smartscraper.assert_called_once_with(
            website_url=url,
            user_prompt=prompt,
            output_schema=None,
            timeout=30,
            custom_param="value",
        )
        assert response == expected_response

    def test_smartscraper_exception_handling(self, tool_spec_with_api_key):
        """Test SmartScraper exception handling."""
        tool_spec, mock_client = tool_spec_with_api_key

        mock_client.smartscraper.side_effect = Exception("API Error")

        response = tool_spec.scrapegraph_smartscraper(
            prompt="test", url="https://example.com"
        )

        assert "error" in response
        assert "SmartScraper failed: API Error" in response["error"]


class TestMarkdownify:
    """Test Markdownify functionality."""

    def test_markdownify_success(self, tool_spec_from_env):
        """Test successful markdownify operation."""
        tool_spec, mock_client = tool_spec_from_env

        url = "https://example.com"
        expected_response = "# Test Page\n\nThis is test content."

        mock_client.markdownify.return_value = expected_response

        response = tool_spec.scrapegraph_markdownify(url=url)

        mock_client.markdownify.assert_called_once_with(website_url=url)
        assert response == expected_response

    def test_markdownify_with_kwargs(self, tool_spec_with_api_key):
        """Test markdownify with additional parameters."""
        tool_spec, mock_client = tool_spec_with_api_key

        url = "https://example.com"
        expected_response = "# Test Content"

        mock_client.markdownify.return_value = expected_response

        response = tool_spec.scrapegraph_markdownify(
            url=url, timeout=60, format="clean"
        )

        mock_client.markdownify.assert_called_once_with(
            website_url=url, timeout=60, format="clean"
        )
        assert response == expected_response

    def test_markdownify_exception_handling(self, tool_spec_from_env):
        """Test markdownify exception handling."""
        tool_spec, mock_client = tool_spec_from_env

        mock_client.markdownify.side_effect = Exception("Network Error")

        response = tool_spec.scrapegraph_markdownify(url="https://example.com")

        assert "Markdownify failed: Network Error" in response


class TestSearch:
    """Test Search functionality."""

    def test_search_basic(self, tool_spec_with_api_key):
        """Test basic search functionality."""
        tool_spec, mock_client = tool_spec_with_api_key

        query = "test search query"
        expected_response = "Search results content"

        mock_client.search.return_value = expected_response

        response = tool_spec.scrapegraph_search(query=query)

        mock_client.search.assert_called_once_with(query=query)
        assert response == expected_response

    def test_search_with_max_results(self, tool_spec_from_env):
        """Test search with max_results parameter."""
        tool_spec, mock_client = tool_spec_from_env

        query = "Python programming"
        max_results = 5
        expected_response = "Limited search results"

        mock_client.search.return_value = expected_response

        response = tool_spec.scrapegraph_search(query=query, max_results=max_results)

        mock_client.search.assert_called_once_with(query=query, max_results=max_results)
        assert response == expected_response

    def test_search_with_kwargs(self, tool_spec_with_api_key):
        """Test search with additional kwargs."""
        tool_spec, mock_client = tool_spec_with_api_key

        query = "AI tools"
        expected_response = "AI search results"

        mock_client.search.return_value = expected_response

        response = tool_spec.scrapegraph_search(query=query, language="en", region="US")

        mock_client.search.assert_called_once_with(
            query=query, language="en", region="US"
        )
        assert response == expected_response

    def test_search_exception_handling(self, tool_spec_from_env):
        """Test search exception handling."""
        tool_spec, mock_client = tool_spec_from_env

        mock_client.search.side_effect = Exception("Search API Error")

        response = tool_spec.scrapegraph_search(query="test query")

        assert "Search failed: Search API Error" in response


class TestBasicScrape:
    """Test basic scrape functionality."""

    def test_scrape_basic(self, tool_spec_with_api_key):
        """Test basic scrape functionality."""
        tool_spec, mock_client = tool_spec_with_api_key

        url = "https://example.com"
        expected_response = {
            "html": "<html><body>Test content</body></html>",
            "request_id": "test-123",
        }

        mock_client.scrape.return_value = expected_response

        response = tool_spec.scrapegraph_scrape(url=url)

        mock_client.scrape.assert_called_once_with(
            website_url=url, render_heavy_js=False
        )
        assert response == expected_response

    def test_scrape_with_js_rendering(self, tool_spec_from_env):
        """Test scrape with JavaScript rendering."""
        tool_spec, mock_client = tool_spec_from_env

        url = "https://example.com"
        expected_response = {"html": "<html>Dynamic content</html>"}

        mock_client.scrape.return_value = expected_response

        response = tool_spec.scrapegraph_scrape(url=url, render_heavy_js=True)

        mock_client.scrape.assert_called_once_with(
            website_url=url, render_heavy_js=True
        )
        assert response == expected_response

    def test_scrape_with_headers(self, tool_spec_with_api_key):
        """Test scrape with custom headers."""
        tool_spec, mock_client = tool_spec_with_api_key

        url = "https://example.com"
        headers = {"User-Agent": "Test Agent", "Accept": "text/html"}
        expected_response = {"html": "<html>Test</html>", "status": "success"}

        mock_client.scrape.return_value = expected_response

        response = tool_spec.scrapegraph_scrape(url=url, headers=headers)

        mock_client.scrape.assert_called_once_with(
            website_url=url, render_heavy_js=False, headers=headers
        )
        assert response == expected_response

    def test_scrape_with_all_options(self, tool_spec_from_env):
        """Test scrape with all options."""
        tool_spec, mock_client = tool_spec_from_env

        url = "https://example.com"
        headers = {"User-Agent": "Full Test"}
        expected_response = {"html": "Full test content"}

        mock_client.scrape.return_value = expected_response

        response = tool_spec.scrapegraph_scrape(
            url=url,
            render_heavy_js=True,
            headers=headers,
            timeout=30,
            custom_option="value",
        )

        mock_client.scrape.assert_called_once_with(
            website_url=url,
            render_heavy_js=True,
            headers=headers,
            timeout=30,
            custom_option="value",
        )
        assert response == expected_response

    def test_scrape_exception_handling(self, tool_spec_with_api_key):
        """Test scrape exception handling."""
        tool_spec, mock_client = tool_spec_with_api_key

        mock_client.scrape.side_effect = Exception("Scrape Error")

        response = tool_spec.scrapegraph_scrape(url="https://example.com")

        assert "error" in response
        assert "Scrape failed: Scrape Error" in response["error"]


class TestAgenticScraper:
    """Test Agentic Scraper functionality."""

    def test_agentic_scraper_with_schema(self, tool_spec_from_env):
        """Test agentic scraper with schema."""
        tool_spec, mock_client = tool_spec_from_env

        prompt = "Navigate and extract product info"
        url = "https://example.com"
        schema = TestProductSchema
        expected_response = {"name": "Test Product", "price": "$99"}

        mock_client.agentic_scraper.return_value = expected_response

        response = tool_spec.scrapegraph_agentic_scraper(
            prompt=prompt, url=url, schema=schema
        )

        mock_client.agentic_scraper.assert_called_once_with(
            website_url=url, user_prompt=prompt, output_schema=schema
        )
        assert response == expected_response

    def test_agentic_scraper_without_schema(self, tool_spec_with_api_key):
        """Test agentic scraper without schema."""
        tool_spec, mock_client = tool_spec_with_api_key

        prompt = "Navigate and find contact info"
        url = "https://example.com"
        expected_response = {"contact": "info@example.com", "phone": "123-456-7890"}

        mock_client.agentic_scraper.return_value = expected_response

        response = tool_spec.scrapegraph_agentic_scraper(prompt=prompt, url=url)

        mock_client.agentic_scraper.assert_called_once_with(
            website_url=url, user_prompt=prompt, output_schema=None
        )
        assert response == expected_response

    def test_agentic_scraper_with_kwargs(self, tool_spec_from_env):
        """Test agentic scraper with additional parameters."""
        tool_spec, mock_client = tool_spec_from_env

        prompt = "Complex navigation task"
        url = "https://example.com"
        expected_response = {"navigation_result": "success"}

        mock_client.agentic_scraper.return_value = expected_response

        response = tool_spec.scrapegraph_agentic_scraper(
            prompt=prompt, url=url, max_depth=3, follow_links=True
        )

        mock_client.agentic_scraper.assert_called_once_with(
            website_url=url,
            user_prompt=prompt,
            output_schema=None,
            max_depth=3,
            follow_links=True,
        )
        assert response == expected_response

    def test_agentic_scraper_exception_handling(self, tool_spec_with_api_key):
        """Test agentic scraper exception handling."""
        tool_spec, mock_client = tool_spec_with_api_key

        mock_client.agentic_scraper.side_effect = Exception("Navigation Error")

        response = tool_spec.scrapegraph_agentic_scraper(
            prompt="test", url="https://example.com"
        )

        assert "error" in response
        assert "Agentic scraper failed: Navigation Error" in response["error"]


class TestToolIntegration:
    """Test tool integration features."""

    def test_spec_functions_list(self):
        """Test that all expected functions are in spec_functions."""
        expected_functions = [
            "scrapegraph_smartscraper",
            "scrapegraph_markdownify",
            "scrapegraph_search",
            "scrapegraph_scrape",
            "scrapegraph_agentic_scraper",
        ]

        assert ScrapegraphToolSpec.spec_functions == expected_functions

    def test_to_tool_list(self, tool_spec_with_api_key):
        """Test conversion to LlamaIndex tool list."""
        tool_spec, _ = tool_spec_with_api_key

        tools = tool_spec.to_tool_list()

        # Should create one tool for each spec function
        assert len(tools) == len(ScrapegraphToolSpec.spec_functions)

        # Check tool names match spec functions
        tool_names = [tool.metadata.name for tool in tools]
        for func_name in ScrapegraphToolSpec.spec_functions:
            assert func_name in tool_names
