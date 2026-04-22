"""
Integration tests for ScrapeGraph tool specification.

These tests verify that the tool integrates properly with LlamaIndex
and can be used in real-world scenarios.
"""

import os
from unittest.mock import Mock, patch
from typing import List

import pytest
from pydantic import BaseModel, Field

from llama_index.tools.scrapegraph import ScrapegraphToolSpec


class IntegrationTestSchema(BaseModel):
    """Test schema for integration testing."""

    title: str = Field(description="Page title")
    content: str = Field(description="Main content")
    links: List[str] = Field(description="Important links", default_factory=list)


class TestLlamaIndexIntegration:
    """Test integration with LlamaIndex core components."""

    @pytest.fixture
    def mock_tool_spec(self):
        """Create a mocked tool spec for integration testing."""
        with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.from_env.return_value = mock_client

            tool_spec = ScrapegraphToolSpec()
            tool_spec.client = mock_client

            return tool_spec, mock_client

    def test_tool_conversion_to_llamaindex_tools(self, mock_tool_spec):
        """Test that tools are properly converted to LlamaIndex format."""
        tool_spec, mock_client = mock_tool_spec

        tools = tool_spec.to_tool_list()

        # Verify all spec functions are converted
        assert len(tools) == len(ScrapegraphToolSpec.spec_functions)

        # Verify each tool has proper metadata
        for tool in tools:
            assert hasattr(tool, "metadata")
            assert hasattr(tool.metadata, "name")
            assert hasattr(tool.metadata, "description")
            assert tool.metadata.name in ScrapegraphToolSpec.spec_functions

        # Verify tools can be called
        for tool in tools:
            assert hasattr(tool, "call")
            assert callable(tool.call)

    def test_tool_metadata_and_descriptions(self, mock_tool_spec):
        """Test that tools have proper metadata and descriptions."""
        tool_spec, _ = mock_tool_spec

        tools = tool_spec.to_tool_list()

        expected_descriptions = {
            "scrapegraph_smartscraper": "Perform intelligent web scraping",
            "scrapegraph_markdownify": "Convert webpage content to markdown",
            "scrapegraph_search": "Perform a search query",
            "scrapegraph_scrape": "Perform basic HTML scraping",
            "scrapegraph_agentic_scraper": "Perform agentic web scraping",
        }

        for tool in tools:
            tool_name = tool.metadata.name
            assert tool_name in expected_descriptions
            # Check that description contains expected keywords
            description_lower = tool.metadata.description.lower()
            expected_keywords = expected_descriptions[tool_name].lower()
            assert any(
                keyword in description_lower for keyword in expected_keywords.split()
            )

    @patch.dict(os.environ, {"SGAI_API_KEY": "test-key"})
    def test_environment_variable_initialization(self):
        """Test initialization using environment variables."""
        with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.from_env.return_value = mock_client

            # This should not raise an exception
            tool_spec = ScrapegraphToolSpec()

            # Verify that from_env was called
            mock_client_class.from_env.assert_called_once()

    def test_error_handling_in_tool_execution(self, mock_tool_spec):
        """Test that tools handle errors gracefully when integrated."""
        tool_spec, mock_client = mock_tool_spec

        # Mock all client methods to raise exceptions
        mock_client.smartscraper.side_effect = Exception("API Error")
        mock_client.markdownify.side_effect = Exception("Network Error")
        mock_client.search.side_effect = Exception("Search Error")
        mock_client.scrape.side_effect = Exception("Scrape Error")
        mock_client.agentic_scraper.side_effect = Exception("Navigation Error")

        # Test each method handles errors gracefully
        response1 = tool_spec.scrapegraph_smartscraper("test", "https://example.com")
        assert "error" in response1
        assert "SmartScraper failed" in response1["error"]

        response2 = tool_spec.scrapegraph_markdownify("https://example.com")
        assert "Markdownify failed" in response2

        response3 = tool_spec.scrapegraph_search("test query")
        assert "Search failed" in response3

        response4 = tool_spec.scrapegraph_scrape("https://example.com")
        assert "error" in response4
        assert "Scrape failed" in response4["error"]

        response5 = tool_spec.scrapegraph_agentic_scraper("test", "https://example.com")
        assert "error" in response5
        assert "Agentic scraper failed" in response5["error"]


class TestSchemaValidation:
    """Test Pydantic schema validation and integration."""

    @pytest.fixture
    def mock_tool_spec(self):
        """Create a mocked tool spec for schema testing."""
        with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.from_env.return_value = mock_client

            tool_spec = ScrapegraphToolSpec()
            tool_spec.client = mock_client

            return tool_spec, mock_client

    def test_schema_integration_smartscraper(self, mock_tool_spec):
        """Test schema integration with SmartScraper."""
        tool_spec, mock_client = mock_tool_spec

        # Mock response that matches schema
        mock_response = {
            "title": "Test Page",
            "content": "Test content",
            "links": ["https://example.com/link1", "https://example.com/link2"],
        }
        mock_client.smartscraper.return_value = mock_response

        result = tool_spec.scrapegraph_smartscraper(
            prompt="Extract page info",
            url="https://example.com",
            schema=IntegrationTestSchema,
        )

        # Verify the schema was passed correctly
        mock_client.smartscraper.assert_called_once_with(
            website_url="https://example.com",
            user_prompt="Extract page info",
            output_schema=IntegrationTestSchema,
        )

        assert result == mock_response

    def test_schema_integration_agentic_scraper(self, mock_tool_spec):
        """Test schema integration with Agentic Scraper."""
        tool_spec, mock_client = mock_tool_spec

        mock_response = {
            "title": "Navigation Result",
            "content": "Found content through navigation",
            "links": ["https://example.com/found"],
        }
        mock_client.agentic_scraper.return_value = mock_response

        result = tool_spec.scrapegraph_agentic_scraper(
            prompt="Navigate and extract",
            url="https://example.com",
            schema=IntegrationTestSchema,
        )

        mock_client.agentic_scraper.assert_called_once_with(
            website_url="https://example.com",
            user_prompt="Navigate and extract",
            output_schema=IntegrationTestSchema,
        )

        assert result == mock_response

    def test_multiple_schema_types(self, mock_tool_spec):
        """Test that different schema types are handled correctly."""
        tool_spec, mock_client = mock_tool_spec

        # Test with list of schemas
        schema_list = [IntegrationTestSchema]
        mock_client.smartscraper.return_value = {"result": "list schema test"}

        tool_spec.scrapegraph_smartscraper(
            prompt="test", url="https://example.com", schema=schema_list
        )

        mock_client.smartscraper.assert_called_with(
            website_url="https://example.com",
            user_prompt="test",
            output_schema=schema_list,
        )

        # Test with dict schema
        mock_client.smartscraper.reset_mock()
        dict_schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        mock_client.smartscraper.return_value = {"result": "dict schema test"}

        tool_spec.scrapegraph_smartscraper(
            prompt="test", url="https://example.com", schema=dict_schema
        )

        mock_client.smartscraper.assert_called_with(
            website_url="https://example.com",
            user_prompt="test",
            output_schema=dict_schema,
        )


class TestParameterValidation:
    """Test parameter validation and handling."""

    @pytest.fixture
    def mock_tool_spec(self):
        """Create a mocked tool spec for parameter testing."""
        with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.from_env.return_value = mock_client

            tool_spec = ScrapegraphToolSpec()
            tool_spec.client = mock_client

            return tool_spec, mock_client

    def test_url_parameter_validation(self, mock_tool_spec):
        """Test that URL parameters are handled correctly."""
        tool_spec, mock_client = mock_tool_spec

        # Test various URL formats
        test_urls = [
            "https://example.com",
            "http://example.com",
            "https://example.com/path",
            "https://example.com/path?param=value",
            "https://subdomain.example.com",
        ]

        mock_client.scrape.return_value = {"html": "test"}

        for url in test_urls:
            mock_client.scrape.reset_mock()
            tool_spec.scrapegraph_scrape(url=url)
            mock_client.scrape.assert_called_once_with(
                website_url=url, render_heavy_js=False
            )

    def test_headers_parameter_handling(self, mock_tool_spec):
        """Test custom headers parameter handling."""
        tool_spec, mock_client = mock_tool_spec

        headers = {
            "User-Agent": "Test Agent",
            "Accept": "text/html",
            "Authorization": "Bearer token",
            "Custom-Header": "custom-value",
        }

        mock_client.scrape.return_value = {"html": "test"}

        tool_spec.scrapegraph_scrape(url="https://example.com", headers=headers)

        mock_client.scrape.assert_called_once_with(
            website_url="https://example.com", render_heavy_js=False, headers=headers
        )

    def test_boolean_parameter_handling(self, mock_tool_spec):
        """Test boolean parameter handling."""
        tool_spec, mock_client = mock_tool_spec

        mock_client.scrape.return_value = {"html": "test"}

        # Test with render_heavy_js=True
        tool_spec.scrapegraph_scrape(url="https://example.com", render_heavy_js=True)

        mock_client.scrape.assert_called_with(
            website_url="https://example.com", render_heavy_js=True
        )

        # Test with render_heavy_js=False
        mock_client.scrape.reset_mock()
        tool_spec.scrapegraph_scrape(url="https://example.com", render_heavy_js=False)

        mock_client.scrape.assert_called_with(
            website_url="https://example.com", render_heavy_js=False
        )

    def test_kwargs_parameter_passing(self, mock_tool_spec):
        """Test that kwargs are passed through correctly."""
        tool_spec, mock_client = mock_tool_spec

        mock_client.smartscraper.return_value = {"result": "test"}

        # Test kwargs with SmartScraper
        tool_spec.scrapegraph_smartscraper(
            prompt="test",
            url="https://example.com",
            timeout=30,
            retries=3,
            custom_param="value",
        )

        mock_client.smartscraper.assert_called_once_with(
            website_url="https://example.com",
            user_prompt="test",
            output_schema=None,
            timeout=30,
            retries=3,
            custom_param="value",
        )


class TestRealWorldScenarios:
    """Test scenarios that simulate real-world usage."""

    @pytest.fixture
    def mock_tool_spec(self):
        """Create a mocked tool spec for real-world testing."""
        with patch("llama_index.tools.scrapegraph.base.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.from_env.return_value = mock_client

            tool_spec = ScrapegraphToolSpec()
            tool_spec.client = mock_client

            return tool_spec, mock_client

    def test_e_commerce_product_extraction(self, mock_tool_spec):
        """Test extracting product information from e-commerce sites."""
        tool_spec, mock_client = mock_tool_spec

        # Mock e-commerce product data
        mock_response = {
            "products": [
                {"name": "Laptop", "price": "$999", "rating": "4.5/5"},
                {"name": "Mouse", "price": "$29", "rating": "4.2/5"},
            ]
        }
        mock_client.smartscraper.return_value = mock_response

        result = tool_spec.scrapegraph_smartscraper(
            prompt="Extract product names, prices, and ratings from this e-commerce page",
            url="https://shop.example.com/laptops",
        )

        assert result == mock_response
        mock_client.smartscraper.assert_called_once()

    def test_news_article_summarization(self, mock_tool_spec):
        """Test extracting and summarizing news articles."""
        tool_spec, mock_client = mock_tool_spec

        # Mock news article markdown
        mock_markdown = """# Breaking News: AI Advances

        ## Summary
        Artificial Intelligence has made significant breakthroughs...

        ## Key Points
        - New neural network architecture
        - 30% improvement in efficiency
        - Applications in healthcare
        """
        mock_client.markdownify.return_value = mock_markdown

        result = tool_spec.scrapegraph_markdownify(
            url="https://news.example.com/ai-breakthrough"
        )

        assert result == mock_markdown
        assert "# Breaking News" in result
        assert "Key Points" in result

    def test_complex_site_navigation(self, mock_tool_spec):
        """Test complex site navigation with agentic scraper."""
        tool_spec, mock_client = mock_tool_spec

        # Mock complex navigation result
        mock_response = {
            "contact_info": {
                "email": "contact@company.com",
                "phone": "+1-555-0123",
                "address": "123 Tech Street, Silicon Valley",
            },
            "navigation_path": ["Home", "About", "Contact", "Support"],
        }
        mock_client.agentic_scraper.return_value = mock_response

        result = tool_spec.scrapegraph_agentic_scraper(
            prompt="Navigate through the website to find comprehensive contact information",
            url="https://company.example.com",
        )

        assert result == mock_response
        assert "contact_info" in result
        assert "navigation_path" in result

    def test_multi_step_workflow(self, mock_tool_spec):
        """Test a multi-step workflow combining different tools."""
        tool_spec, mock_client = mock_tool_spec

        # Step 1: Search for relevant pages
        mock_client.search.return_value = "Found relevant pages about Python tutorials"

        search_result = tool_spec.scrapegraph_search(
            query="Python programming tutorials beginner", max_results=5
        )

        # Step 2: Scrape the found page
        mock_client.scrape.return_value = {
            "html": "<html><head><title>Python Tutorial</title></head><body>Learn Python...</body></html>",
            "request_id": "req-123",
        }

        scrape_result = tool_spec.scrapegraph_scrape(
            url="https://python-tutorial.example.com"
        )

        # Step 3: Convert to markdown for analysis
        mock_client.markdownify.return_value = (
            "# Python Tutorial\n\nLearn Python programming..."
        )

        markdown_result = tool_spec.scrapegraph_markdownify(
            url="https://python-tutorial.example.com"
        )

        # Verify all steps executed correctly
        assert "Python tutorials" in search_result
        assert "html" in scrape_result
        assert "# Python Tutorial" in markdown_result

        # Verify all client methods were called
        mock_client.search.assert_called_once()
        mock_client.scrape.assert_called_once()
        mock_client.markdownify.assert_called_once()
