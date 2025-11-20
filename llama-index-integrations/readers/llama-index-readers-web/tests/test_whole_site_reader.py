"""Tests for WholeSiteReader."""

from unittest.mock import MagicMock, patch
import pytest
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By

from llama_index.readers.web import WholeSiteReader
from llama_index.core.schema import Document


@pytest.fixture
def mock_driver():
    """Create a mock Selenium WebDriver."""
    driver = MagicMock()

    # Mock find_element for body extraction
    body_element = MagicMock()
    body_element.text = "Sample page content"
    driver.find_element.return_value = body_element

    # Mock execute_script for link extraction
    driver.execute_script.return_value = []

    return driver


@pytest.fixture
def simple_html_page():
    """Simple HTML page content for testing."""
    return """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Test Page</h1>
        <p>This is a test page.</p>
        <a href="https://example.com/page1">Link 1</a>
        <a href="https://example.com/page2">Link 2</a>
    </body>
    </html>
    """


class TestWholeSiteReaderInit:
    """Test WholeSiteReader initialization."""

    @patch("llama_index.readers.web.whole_site.base.WholeSiteReader.setup_driver")
    def test_init_with_defaults(self, mock_setup_driver):
        """Test initialization with default parameters."""
        mock_setup_driver.return_value = MagicMock()

        reader = WholeSiteReader(prefix="https://example.com")

        assert reader.prefix == "https://example.com"
        assert reader.max_depth == 10
        assert reader.uri_as_id is False
        mock_setup_driver.assert_called_once()

    @patch("llama_index.readers.web.whole_site.base.WholeSiteReader.setup_driver")
    def test_init_with_custom_params(self, mock_setup_driver):
        """Test initialization with custom parameters."""
        mock_setup_driver.return_value = MagicMock()

        reader = WholeSiteReader(
            prefix="https://example.com", max_depth=5, uri_as_id=True
        )

        assert reader.prefix == "https://example.com"
        assert reader.max_depth == 5
        assert reader.uri_as_id is True

    def test_init_with_custom_driver(self):
        """Test initialization with custom driver."""
        custom_driver = MagicMock()

        reader = WholeSiteReader(prefix="https://example.com", driver=custom_driver)

        assert reader.driver == custom_driver

    @patch("llama_index.readers.web.whole_site.base.webdriver.Chrome")
    @patch("chromedriver_autoinstaller.install")
    def test_setup_driver(self, mock_install, mock_chrome):
        """Test setup_driver method creates Chrome WebDriver."""
        mock_chrome_instance = MagicMock()
        mock_chrome.return_value = mock_chrome_instance

        reader = WholeSiteReader(prefix="https://example.com", driver=MagicMock())
        driver = reader.setup_driver()

        assert driver == mock_chrome_instance
        mock_install.assert_called_once()
        mock_chrome.assert_called_once()


class TestWholeSiteReaderHelperMethods:
    """Test helper methods."""

    def test_clean_url_removes_fragment(self):
        """Test that clean_url removes URL fragments."""
        reader = WholeSiteReader(prefix="https://example.com", driver=MagicMock())

        url_with_fragment = "https://example.com/page#section"
        cleaned = reader.clean_url(url_with_fragment)

        assert cleaned == "https://example.com/page"

    def test_clean_url_no_fragment(self):
        """Test clean_url with URL without fragment."""
        reader = WholeSiteReader(prefix="https://example.com", driver=MagicMock())

        url_without_fragment = "https://example.com/page"
        cleaned = reader.clean_url(url_without_fragment)

        assert cleaned == "https://example.com/page"

    @patch("llama_index.readers.web.whole_site.base.WholeSiteReader.setup_driver")
    def test_restart_driver(self, mock_setup_driver):
        """Test driver restart functionality."""
        old_driver = MagicMock()
        new_driver = MagicMock()
        mock_setup_driver.return_value = new_driver

        reader = WholeSiteReader(prefix="https://example.com", driver=old_driver)
        reader.restart_driver()

        old_driver.quit.assert_called_once()
        assert reader.driver == new_driver


class TestWholeSiteReaderExtraction:
    """Test content and link extraction methods."""

    def test_extract_content(self):
        """Test content extraction from page."""
        mock_driver = MagicMock()
        mock_wait = MagicMock()

        body_element = MagicMock()
        body_element.text = "  Sample page content  "
        mock_driver.find_element.return_value = body_element

        with patch(
            "llama_index.readers.web.whole_site.base.WebDriverWait",
            return_value=mock_wait,
        ):
            reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)
            content = reader.extract_content()

        assert content == "Sample page content"
        mock_driver.find_element.assert_called_once_with(By.TAG_NAME, "body")

    def test_extract_links(self):
        """Test link extraction from page."""
        mock_driver = MagicMock()
        expected_links = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]
        mock_driver.execute_script.return_value = expected_links

        reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)
        links = reader.extract_links()

        assert links == expected_links
        mock_driver.execute_script.assert_called_once()


class TestWholeSiteReaderLoadData:
    """Test the main load_data method."""

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    def test_load_data_single_page(self, _mock_sleep, mock_driver):
        """Test loading data from a single page."""
        mock_driver.execute_script.return_value = []  # No links

        body_element = MagicMock()
        body_element.text = "Page content"
        mock_driver.find_element.return_value = body_element

        with patch("llama_index.readers.web.whole_site.base.WebDriverWait"):
            reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)
            documents = reader.load_data("https://example.com")

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert documents[0].text == "Page content"
        assert documents[0].metadata["URL"] == "https://example.com"
        mock_driver.quit.assert_called_once()

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    def test_load_data_multiple_pages(self, _mock_sleep, mock_driver):
        """Test loading data from multiple pages with BFS."""
        # First page returns links to page1 and page2
        # Subsequent pages return no links
        call_count = [0]

        def mock_execute_script(_script):
            call_count[0] += 1
            if call_count[0] == 1:
                return ["https://example.com/page1", "https://example.com/page2"]
            return []

        mock_driver.execute_script.side_effect = mock_execute_script

        body_element = MagicMock()
        body_element.text = "Page content"
        mock_driver.find_element.return_value = body_element

        with patch("llama_index.readers.web.whole_site.base.WebDriverWait"):
            reader = WholeSiteReader(
                prefix="https://example.com", driver=mock_driver, max_depth=2
            )
            documents = reader.load_data("https://example.com")

        # Should have 3 documents: base page + 2 linked pages
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    def test_load_data_respects_prefix(self, _mock_sleep, mock_driver):
        """Test that only URLs matching prefix are crawled."""
        # Return links with different prefixes
        mock_driver.execute_script.return_value = [
            "https://example.com/page1",  # Matches prefix
            "https://other.com/page2",  # Doesn't match prefix
            "https://example.com/page3",  # Matches prefix
        ]

        body_element = MagicMock()
        body_element.text = "Page content"
        mock_driver.find_element.return_value = body_element

        with patch("llama_index.readers.web.whole_site.base.WebDriverWait"):
            reader = WholeSiteReader(
                prefix="https://example.com", driver=mock_driver, max_depth=2
            )
            documents = reader.load_data("https://example.com")

        # Should only crawl pages matching prefix
        # Base page + 2 matching pages = 3 documents
        assert len(documents) == 3

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    def test_load_data_respects_max_depth(self, _mock_sleep, mock_driver):
        """Test that crawling respects max_depth parameter."""
        # Always return a new link to test depth limiting
        mock_driver.execute_script.return_value = ["https://example.com/next"]

        body_element = MagicMock()
        body_element.text = "Page content"
        mock_driver.find_element.return_value = body_element

        with patch("llama_index.readers.web.whole_site.base.WebDriverWait"):
            reader = WholeSiteReader(
                prefix="https://example.com",
                driver=mock_driver,
                max_depth=0,  # Only crawl the base page
            )
            documents = reader.load_data("https://example.com")

        # Should only have 1 document (base page)
        assert len(documents) == 1

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    def test_load_data_with_uri_as_id(self, _mock_sleep, mock_driver):
        """Test that uri_as_id sets document ID to URL."""
        mock_driver.execute_script.return_value = []

        body_element = MagicMock()
        body_element.text = "Page content"
        mock_driver.find_element.return_value = body_element

        with patch("llama_index.readers.web.whole_site.base.WebDriverWait"):
            with pytest.warns(UserWarning):
                reader = WholeSiteReader(
                    prefix="https://example.com", driver=mock_driver, uri_as_id=True
                )
                documents = reader.load_data("https://example.com")

        assert documents[0].id_ == "https://example.com"


class TestWholeSiteReaderErrorHandling:
    """Test error handling scenarios."""

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WholeSiteReader.restart_driver")
    def test_webdriver_exception_restarts_driver(
        self, mock_restart, _mock_sleep, mock_driver
    ):
        """Test that WebDriverException triggers driver restart."""
        # First call raises WebDriverException, second succeeds
        mock_driver.get.side_effect = [WebDriverException("Connection lost"), None]
        mock_driver.execute_script.return_value = []

        body_element = MagicMock()
        body_element.text = "Page content"
        mock_driver.find_element.return_value = body_element

        with patch("llama_index.readers.web.whole_site.base.WebDriverWait"):
            reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)
            # Should not raise exception
            reader.load_data("https://example.com")

        mock_restart.assert_called_once()

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    def test_generic_exception_skips_url(self, _mock_sleep, mock_driver):
        """Test that generic exceptions skip the URL and continue."""
        # First URL raises exception, second succeeds
        call_count = [0]

        def mock_get_side_effect(_url):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Some error")

        mock_driver.get.side_effect = mock_get_side_effect
        mock_driver.execute_script.return_value = []

        body_element = MagicMock()
        body_element.text = "Page content"
        mock_driver.find_element.return_value = body_element

        with patch("llama_index.readers.web.whole_site.base.WebDriverWait"):
            reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)
            # Should not raise exception, just skip the failed URL
            documents = reader.load_data("https://example.com")

        # Should have 0 documents since the only URL failed
        assert len(documents) == 0

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    def test_exception_in_link_processing(self, _mock_sleep, mock_driver):
        """Test that exceptions during link processing are caught and skipped."""

        # Create a link object that will raise an exception when startswith() is called
        # This simulates a malformed link that passes through clean_url but fails during processing
        class BadLink(str):
            def startswith(self, prefix):
                raise TypeError("Cannot check startswith on this link")

        # Return links including one that will raise an exception
        mock_driver.execute_script.return_value = [
            "https://example.com/page1",
            BadLink("https://example.com/bad"),  # This will raise exception
            "https://example.com/page2",
        ]

        body_element = MagicMock()
        body_element.text = "Page content"
        mock_driver.find_element.return_value = body_element

        with patch("llama_index.readers.web.whole_site.base.WebDriverWait"):
            reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)
            documents = reader.load_data("https://example.com")

        # Should successfully process despite the exception
        assert len(documents) >= 1


class TestWholeSiteReaderConfigurableDelay:
    """Test configurable rate limiting (delay parameter)."""

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_default_delay(self, _mock_wait, mock_sleep, mock_driver):
        """Test that default delay is 1.0 seconds."""
        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)

        assert reader.delay == 1.0
        reader.load_data("https://example.com")

        # Verify sleep was called with default delay
        mock_sleep.assert_called_with(1.0)

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_custom_delay(self, _mock_wait, mock_sleep, mock_driver):
        """Test that custom delay is respected."""
        mock_driver.execute_script.return_value = []

        custom_delay = 2.5
        reader = WholeSiteReader(
            prefix="https://example.com", driver=mock_driver, delay=custom_delay
        )

        assert reader.delay == custom_delay
        reader.load_data("https://example.com")

        # Verify sleep was called with custom delay
        mock_sleep.assert_called_with(custom_delay)

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_zero_delay(self, _mock_wait, mock_sleep, mock_driver):
        """Test that zero delay works (no rate limiting)."""
        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(
            prefix="https://example.com", driver=mock_driver, delay=0.0
        )

        reader.load_data("https://example.com")

        # Verify sleep was called with zero delay
        mock_sleep.assert_called_with(0.0)


class TestWholeSiteReaderRobotsTxt:
    """Test robots.txt support."""

    @patch("llama_index.readers.web.whole_site.base.RobotFileParser")
    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_robots_txt_allows_url(
        self, _mock_wait, _mock_sleep, mock_robot_parser_class, mock_driver
    ):
        """Test that URLs allowed by robots.txt are crawled."""
        # Mock RobotFileParser to allow all URLs
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = True
        mock_robot_parser_class.return_value = mock_parser

        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(
            prefix="https://example.com", driver=mock_driver, respect_robots_txt=True
        )

        documents = reader.load_data("https://example.com")

        # Should successfully crawl the URL
        assert len(documents) == 1
        mock_parser.can_fetch.assert_called()

    @patch("llama_index.readers.web.whole_site.base.RobotFileParser")
    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_robots_txt_disallows_url(
        self, _mock_wait, _mock_sleep, mock_robot_parser_class, mock_driver
    ):
        """Test that URLs disallowed by robots.txt are skipped."""
        # Mock RobotFileParser to disallow all URLs
        mock_parser = MagicMock()
        mock_parser.can_fetch.return_value = False
        mock_robot_parser_class.return_value = mock_parser

        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(
            prefix="https://example.com", driver=mock_driver, respect_robots_txt=True
        )

        documents = reader.load_data("https://example.com")

        # Should skip the URL and return no documents
        assert len(documents) == 0
        mock_parser.can_fetch.assert_called()

    @patch("llama_index.readers.web.whole_site.base.RobotFileParser")
    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_robots_txt_disabled(
        self, _mock_wait, _mock_sleep, mock_robot_parser_class, mock_driver
    ):
        """Test that robots.txt can be disabled."""
        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(
            prefix="https://example.com", driver=mock_driver, respect_robots_txt=False
        )

        documents = reader.load_data("https://example.com")

        # Should crawl without checking robots.txt
        assert len(documents) == 1
        # RobotFileParser should not be instantiated
        mock_robot_parser_class.assert_not_called()

    @patch("llama_index.readers.web.whole_site.base.RobotFileParser")
    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_robots_txt_fetch_error(
        self, _mock_wait, _mock_sleep, mock_robot_parser_class, mock_driver
    ):
        """Test that errors fetching robots.txt are handled gracefully."""
        # Mock RobotFileParser to raise an exception when reading
        mock_parser = MagicMock()
        mock_parser.read.side_effect = Exception("Failed to fetch robots.txt")
        mock_robot_parser_class.return_value = mock_parser

        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(
            prefix="https://example.com", driver=mock_driver, respect_robots_txt=True
        )

        # Should proceed without robots.txt restrictions
        documents = reader.load_data("https://example.com")
        assert len(documents) == 1


class TestWholeSiteReaderProgressCallback:
    """Test progress tracking callback."""

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_progress_callback_called(self, _mock_wait, _mock_sleep, mock_driver):
        """Test that progress callback is called with correct data."""
        mock_driver.execute_script.return_value = []

        progress_data = []

        def progress_callback(data):
            progress_data.append(data)

        reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)

        reader.load_data("https://example.com", progress_callback=progress_callback)

        # Verify callback was called
        assert len(progress_data) > 0

        # Verify callback data structure
        first_call = progress_data[0]
        assert "current_url" in first_call
        assert "depth" in first_call
        assert "pages_visited" in first_call
        assert "pages_remaining" in first_call
        assert "total_pages_found" in first_call

        # Verify data types
        assert isinstance(first_call["current_url"], str)
        assert isinstance(first_call["depth"], int)
        assert isinstance(first_call["pages_visited"], int)
        assert isinstance(first_call["pages_remaining"], int)
        assert isinstance(first_call["total_pages_found"], int)

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_progress_callback_optional(self, _mock_wait, _mock_sleep, mock_driver):
        """Test that progress callback is optional (None)."""
        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)

        # Should work without progress callback
        documents = reader.load_data("https://example.com")
        assert len(documents) == 1

    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_progress_callback_multiple_pages(
        self, _mock_wait, _mock_sleep, mock_driver
    ):
        """Test progress callback with multiple pages."""

        # Mock multiple pages
        def mock_execute_script(_script):
            return ["https://example.com/page2"]

        mock_driver.execute_script.side_effect = mock_execute_script

        progress_data = []

        def progress_callback(data):
            progress_data.append(data)

        reader = WholeSiteReader(
            prefix="https://example.com", driver=mock_driver, max_depth=1
        )

        reader.load_data("https://example.com", progress_callback=progress_callback)

        # Should be called for each page
        assert len(progress_data) >= 2

        # Verify pages_visited increases
        assert progress_data[0]["pages_visited"] == 0
        assert progress_data[1]["pages_visited"] == 1


class TestWholeSiteReaderLogging:
    """Test enhanced logging."""

    @patch("llama_index.readers.web.whole_site.base.logger")
    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_logging_info_messages(
        self, _mock_wait, _mock_sleep, mock_logger, mock_driver
    ):
        """Test that INFO level logging is used for normal operations."""
        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)

        reader.load_data("https://example.com")

        # Verify INFO logging was called
        assert mock_logger.info.called

        # Check for specific log messages
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Visiting:" in str(call) for call in info_calls)

    @patch("llama_index.readers.web.whole_site.base.logger")
    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_logging_error_on_exception(
        self, _mock_wait, _mock_sleep, mock_logger, mock_driver
    ):
        """Test that ERROR level logging is used for exceptions."""
        # Mock driver to raise exception
        mock_driver.get.side_effect = Exception("Test error")
        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)

        reader.load_data("https://example.com")

        # Verify ERROR logging was called
        assert mock_logger.error.called

    @patch("llama_index.readers.web.whole_site.base.logger")
    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_logging_completion_message(
        self, _mock_wait, _mock_sleep, mock_logger, mock_driver
    ):
        """Test that completion message is logged."""
        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(prefix="https://example.com", driver=mock_driver)

        reader.load_data("https://example.com")

        # Verify completion message was logged
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("Scraping complete" in str(call) for call in info_calls)


class TestWholeSiteReaderRobotsTxtEdgeCases:
    """Test edge cases for robots.txt support."""

    @patch("llama_index.readers.web.whole_site.base.RobotFileParser")
    @patch("llama_index.readers.web.whole_site.base.time.sleep")
    @patch("llama_index.readers.web.whole_site.base.WebDriverWait")
    def test_robots_txt_can_fetch_exception(
        self, _mock_wait, _mock_sleep, mock_robot_parser_class, mock_driver
    ):
        """Test that exceptions in can_fetch are handled gracefully."""
        # Mock RobotFileParser to raise exception in can_fetch
        mock_parser = MagicMock()
        mock_parser.can_fetch.side_effect = Exception("Error checking robots.txt")
        mock_robot_parser_class.return_value = mock_parser

        mock_driver.execute_script.return_value = []

        reader = WholeSiteReader(
            prefix="https://example.com", driver=mock_driver, respect_robots_txt=True
        )

        # Should proceed despite exception
        documents = reader.load_data("https://example.com")
        assert len(documents) == 1
