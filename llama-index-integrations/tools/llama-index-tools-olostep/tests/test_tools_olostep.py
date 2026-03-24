"""Tests for Olostep Tool."""

import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("olostep")


class TestOlostepToolSpec:
    """Test OlostepToolSpec methods."""

    @pytest.fixture
    def tool_spec(self):
        """Fixture to create OlostepToolSpec instance."""
        with patch("olostep.Olostep"):
            from llama_index.tools.olostep import OlostepToolSpec

            return OlostepToolSpec(api_key="test-key")

    def test_scrape_url_success(self, tool_spec):
        """Test successful scrape_url call."""
        # Mock the client response
        mock_response = MagicMock()
        mock_response.markdown = "# Test Content\nThis is test markdown content."
        tool_spec.client.scrapes.create = MagicMock(return_value=mock_response)

        # Call the method
        result = tool_spec.scrape_url(
            url="https://example.com",
            formats="markdown",
            wait_before_scraping=1000,
        )

        # Assertions
        assert len(result) == 1
        assert "Test Content" in result[0].text
        assert result[0].extra_info["url"] == "https://example.com"
        assert result[0].extra_info["format"] == "markdown"

    def test_scrape_url_html_format(self, tool_spec):
        """Test scrape_url with HTML format."""
        mock_response = MagicMock()
        mock_response.html = "<html><body>Test HTML</body></html>"
        tool_spec.client.scrapes.create = MagicMock(return_value=mock_response)

        result = tool_spec.scrape_url(
            url="https://example.com",
            formats="html",
        )

        assert len(result) == 1
        assert "html" in result[0].text
        assert "Test HTML" in result[0].text

    def test_scrape_url_error_handling(self, tool_spec):
        """Test scrape_url error handling."""
        from olostep import Olostep_BaseError

        tool_spec.client.scrapes.create = MagicMock(
            side_effect=Olostep_BaseError("Test error")
        )

        result = tool_spec.scrape_url(url="https://example.com")

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "error" in result[0].extra_info

    def test_crawl_website_success(self, tool_spec):
        """Test successful crawl_website call."""
        mock_page1 = MagicMock()
        mock_page1.url = "https://example.com/page1"
        mock_page1.markdown = "# Page 1 Content"

        mock_page2 = MagicMock()
        mock_page2.url = "https://example.com/page2"
        mock_page2.markdown = "# Page 2 Content"

        mock_crawl = MagicMock()
        mock_crawl.pages = MagicMock(return_value=[mock_page1, mock_page2])

        tool_spec.client.crawls.create = MagicMock(return_value=mock_crawl)

        result = tool_spec.crawl_website(
            url="https://example.com",
            max_pages=50,
        )

        assert len(result) == 2
        assert result[0].extra_info["url"] == "https://example.com/page1"
        assert result[1].extra_info["url"] == "https://example.com/page2"
        assert "Page 1 Content" in result[0].text

    def test_crawl_website_with_filters(self, tool_spec):
        """Test crawl_website with include/exclude filters."""
        mock_page = MagicMock()
        mock_page.url = "https://example.com/docs/api"
        mock_page.markdown = "# API Docs"

        mock_crawl = MagicMock()
        mock_crawl.pages = MagicMock(return_value=[mock_page])

        tool_spec.client.crawls.create = MagicMock(return_value=mock_crawl)

        result = tool_spec.crawl_website(
            url="https://example.com",
            max_pages=20,
            include_urls="/docs/**",
            exclude_urls="/admin/**",
            search_query="API",
        )

        # Verify the create method was called with correct args
        tool_spec.client.crawls.create.assert_called_once()
        call_kwargs = tool_spec.client.crawls.create.call_args[1]
        assert call_kwargs["include_urls"] == "/docs/**"
        assert call_kwargs["exclude_urls"] == "/admin/**"
        assert call_kwargs["search_query"] == "API"

    def test_crawl_website_error_handling(self, tool_spec):
        """Test crawl_website error handling."""
        from olostep import Olostep_BaseError

        tool_spec.client.crawls.create = MagicMock(
            side_effect=Olostep_BaseError("Crawl failed")
        )

        result = tool_spec.crawl_website(url="https://example.com")

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "Crawl failed" in result[0].text

    def test_map_website_success(self, tool_spec):
        """Test successful map_website call."""
        mock_url1 = MagicMock()
        mock_url1.url = "https://example.com/page1"

        mock_url2 = MagicMock()
        mock_url2.url = "https://example.com/page2"

        mock_map = MagicMock()
        mock_map.urls = MagicMock(return_value=[mock_url1, mock_url2])

        tool_spec.client.maps.create = MagicMock(return_value=mock_map)

        result = tool_spec.map_website(url="https://example.com")

        assert len(result) == 1
        assert "https://example.com/page1" in result[0].text
        assert "https://example.com/page2" in result[0].text
        assert result[0].extra_info["total_urls"] == 2

    def test_map_website_with_top_n(self, tool_spec):
        """Test map_website with top_n limit."""
        mock_urls = [MagicMock(url=f"https://example.com/page{i}") for i in range(10)]

        mock_map = MagicMock()
        mock_map.urls = MagicMock(return_value=mock_urls)

        tool_spec.client.maps.create = MagicMock(return_value=mock_map)

        result = tool_spec.map_website(url="https://example.com", top_n=5)

        assert len(result) == 1
        urls = result[0].text.split("\n")
        assert len(urls) == 5

    def test_map_website_error_handling(self, tool_spec):
        """Test map_website error handling."""
        from olostep import Olostep_BaseError

        tool_spec.client.maps.create = MagicMock(
            side_effect=Olostep_BaseError("Map failed")
        )

        result = tool_spec.map_website(url="https://example.com")

        assert len(result) == 1
        assert "Error" in result[0].text

    def test_search_web_success(self, tool_spec):
        """Test successful search_web call."""
        mock_link1 = MagicMock()
        mock_link1.title = "Result 1"
        mock_link1.description = "Description 1"
        mock_link1.url = "https://result1.com"

        mock_link2 = MagicMock()
        mock_link2.title = "Result 2"
        mock_link2.description = "Description 2"
        mock_link2.url = "https://result2.com"

        mock_result_obj = MagicMock()
        mock_result_obj.links = [mock_link1, mock_link2]

        mock_response = MagicMock()
        mock_response.result = mock_result_obj

        tool_spec.client.searches.create = MagicMock(return_value=mock_response)

        result = tool_spec.search_web(query="test query")

        assert len(result) == 2
        assert "Result 1" in result[0].text
        assert "Description 1" in result[0].text
        assert result[0].extra_info["url"] == "https://result1.com"

    def test_search_web_no_results(self, tool_spec):
        """Test search_web with no results."""
        mock_response = MagicMock()
        mock_response.result = None

        tool_spec.client.searches.create = MagicMock(return_value=mock_response)

        result = tool_spec.search_web(query="obscure query")

        assert len(result) == 0

    def test_search_web_error_handling(self, tool_spec):
        """Test search_web error handling."""
        from olostep import Olostep_BaseError

        tool_spec.client.searches.create = MagicMock(
            side_effect=Olostep_BaseError("Search failed")
        )

        result = tool_spec.search_web(query="test")

        assert len(result) == 1
        assert "Error" in result[0].text

    def test_answer_question_success(self, tool_spec):
        """Test successful answer_question call."""
        mock_source1 = MagicMock()
        mock_source1.url = "https://source1.com"

        mock_source2 = MagicMock()
        mock_source2.url = "https://source2.com"

        mock_answer = MagicMock()
        mock_answer.answer = "The answer is 42"
        mock_answer.sources = [mock_source1, mock_source2]

        tool_spec.client.answers.create = MagicMock(return_value=mock_answer)

        result = tool_spec.answer_question(task="What is the answer?")

        assert len(result) == 1
        assert "The answer is 42" in result[0].text
        assert result[0].extra_info["sources"] == [
            "https://source1.com",
            "https://source2.com",
        ]

    def test_answer_question_with_json_schema(self, tool_spec):
        """Test answer_question with JSON schema."""
        mock_answer = MagicMock()
        mock_answer.answer = '{"ceo": "John Doe", "founded": 2010}'
        mock_answer.sources = []

        tool_spec.client.answers.create = MagicMock(return_value=mock_answer)

        result = tool_spec.answer_question(
            task="Get company info",
            json_schema='{"ceo": "", "founded": ""}',
        )

        tool_spec.client.answers.create.assert_called_once()
        call_kwargs = tool_spec.client.answers.create.call_args[1]
        assert call_kwargs["json_schema"] == '{"ceo": "", "founded": ""}'

    def test_answer_question_error_handling(self, tool_spec):
        """Test answer_question error handling."""
        from olostep import Olostep_BaseError

        tool_spec.client.answers.create = MagicMock(
            side_effect=Olostep_BaseError("Answer generation failed")
        )

        result = tool_spec.answer_question(task="Test question")

        assert len(result) == 1
        assert "Error" in result[0].text

    def test_batch_scrape_success(self, tool_spec):
        """Test successful batch_scrape call."""
        mock_item1 = MagicMock()
        mock_item1.url = "https://example.com/1"
        mock_item1.markdown = "# Page 1"

        mock_item2 = MagicMock()
        mock_item2.url = "https://example.com/2"
        mock_item2.markdown = "# Page 2"

        mock_batch = MagicMock()
        mock_batch.items = MagicMock(return_value=[mock_item1, mock_item2])

        tool_spec.client.batches.create = MagicMock(return_value=mock_batch)

        result = tool_spec.batch_scrape(
            urls="https://example.com/1,https://example.com/2",
            formats="markdown",
        )

        assert len(result) == 2
        assert "# Page 1" in result[0].text
        assert result[0].extra_info["url"] == "https://example.com/1"

    def test_batch_scrape_multiple_formats(self, tool_spec):
        """Test batch_scrape with multiple formats."""
        mock_item = MagicMock()
        mock_item.url = "https://example.com"
        mock_item.html = "<html>Test</html>"

        mock_batch = MagicMock()
        mock_batch.items = MagicMock(return_value=[mock_item])

        tool_spec.client.batches.create = MagicMock(return_value=mock_batch)

        result = tool_spec.batch_scrape(
            urls="https://example.com",
            formats="html,markdown",
        )

        assert len(result) == 1
        assert "<html>" in result[0].text

    def test_batch_scrape_with_parser(self, tool_spec):
        """Test batch_scrape with parser_id."""
        mock_item = MagicMock()
        mock_item.url = "https://amazon.com/dp/123"
        mock_item.json = {"title": "Product", "price": "$99"}

        mock_batch = MagicMock()
        mock_batch.items = MagicMock(return_value=[mock_item])

        tool_spec.client.batches.create = MagicMock(return_value=mock_batch)

        result = tool_spec.batch_scrape(
            urls="https://amazon.com/dp/123",
            formats="json",
            parser_id="@olostep/amazon-it-product",
        )

        tool_spec.client.batches.create.assert_called_once()
        call_kwargs = tool_spec.client.batches.create.call_args[1]
        assert call_kwargs["parser_id"] == "@olostep/amazon-it-product"

    def test_batch_scrape_error_handling(self, tool_spec):
        """Test batch_scrape error handling."""
        from olostep import Olostep_BaseError

        tool_spec.client.batches.create = MagicMock(
            side_effect=Olostep_BaseError("Batch failed")
        )

        result = tool_spec.batch_scrape(urls="https://example.com")

        assert len(result) == 1
        assert "Error" in result[0].text

    def test_spec_functions_list(self, tool_spec):
        """Test that all spec_functions are defined."""
        from llama_index.tools.olostep import OlostepToolSpec

        # Check that spec_functions are properly defined
        expected_functions = [
            "scrape_url",
            "crawl_website",
            "map_website",
            "search_web",
            "answer_question",
            "batch_scrape",
        ]

        assert OlostepToolSpec.spec_functions == expected_functions

    def test_to_tool_list(self, tool_spec):
        """Test that to_tool_list() works correctly."""
        tools = tool_spec.to_tool_list()

        # Should have 6 tools
        assert len(tools) == 6

        # All tools should have required attributes
        tool_names = {tool.metadata.name for tool in tools}
        expected_names = {
            "scrape_url",
            "crawl_website",
            "map_website",
            "search_web",
            "answer_question",
            "batch_scrape",
        }
        assert tool_names == expected_names
