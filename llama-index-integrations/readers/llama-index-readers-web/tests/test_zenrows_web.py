"""Tests for ZenRowsWebReader."""

from unittest.mock import Mock, patch
import pytest
import requests

from llama_index.readers.web import ZenRowsWebReader
from llama_index.core.schema import Document


@pytest.fixture
def api_key() -> str:
    """Test API key fixture."""
    return "test_api_key_123"


@pytest.fixture
def test_url() -> str:
    """Test URL fixture."""
    return "https://example.com"


@pytest.fixture
def mock_html_response() -> str:
    """Mock HTML response content."""
    return """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Welcome to Test Page</h1>
        <p>This is a test paragraph.</p>
    </body>
    </html>
    """


@pytest.fixture
def mock_json_response() -> dict:
    """Mock JSON response content."""
    return {
        "title": "Test Page",
        "content": "This is test content",
        "links": ["https://example.com/link1", "https://example.com/link2"],
    }


@pytest.fixture
def mock_screenshot_response() -> bytes:
    """Mock screenshot response content."""
    return b"fake_screenshot_data"


class TestZenRowsWebReader:
    """Test cases for ZenRowsWebReader."""

    def test_init_with_api_key(self, api_key):
        """Test initialization with valid API key."""
        reader = ZenRowsWebReader(api_key=api_key)
        assert reader.api_key == api_key
        assert reader.js_render is False
        assert reader.premium_proxy is False

    def test_init_without_api_key(self):
        """Test initialization without API key raises error."""
        with pytest.raises(ValueError, match="ZenRows API key is required"):
            ZenRowsWebReader(api_key="")

    def test_init_with_custom_params(self, api_key):
        """Test initialization with custom parameters."""
        custom_headers = {"User-Agent": "TestAgent"}
        reader = ZenRowsWebReader(
            api_key=api_key,
            js_render=True,
            premium_proxy=True,
            proxy_country="US",
            custom_headers=custom_headers,
            wait=5000,
            response_type="markdown",
        )

        assert reader.js_render is True
        assert reader.premium_proxy is True
        assert reader.proxy_country == "US"
        assert reader.custom_headers == custom_headers
        assert reader.wait == 5000
        assert reader.response_type == "markdown"

    @patch("requests.get")
    def test_load_data_basic(self, mock_get, api_key, test_url, mock_html_response):
        """Test basic load_data functionality."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_response.content = mock_html_response.encode()
        mock_response.headers = {
            "Content-Type": "text/html",
            "X-Request-Cost": "1.0",
            "X-Request-Id": "test_request_123",
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reader = ZenRowsWebReader(api_key=api_key)
        documents = reader.load_data(test_url)

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert documents[0].text == mock_html_response
        assert documents[0].metadata["source_url"] == test_url
        assert documents[0].metadata["request_cost"] == 1.0
        assert documents[0].metadata["request_id"] == "test_request_123"

    @patch("requests.get")
    def test_load_data_multiple_urls(self, mock_get, api_key, mock_html_response):
        """Test load_data with multiple URLs."""
        urls = ["https://example1.com", "https://example2.com"]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_response.content = mock_html_response.encode()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reader = ZenRowsWebReader(api_key=api_key)
        documents = reader.load_data(urls)

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert documents[0].metadata["source_url"] == urls[0]
        assert documents[1].metadata["source_url"] == urls[1]

    @patch("requests.get")
    def test_load_data_with_custom_headers(
        self, mock_get, api_key, test_url, mock_html_response
    ):
        """Test load_data with custom headers."""
        custom_headers = {"User-Agent": "TestAgent", "Authorization": "Bearer token"}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_response.content = mock_html_response.encode()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test instance-level custom headers
        reader = ZenRowsWebReader(api_key=api_key, custom_headers=custom_headers)
        documents = reader.load_data(test_url)

        # Verify request was made with custom headers
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]["headers"] == custom_headers

        # Test per-request custom headers via extra_params
        mock_get.reset_mock()
        reader2 = ZenRowsWebReader(api_key=api_key)
        per_request_headers = {"User-Agent": "PerRequestAgent"}
        documents = reader2.load_data(
            test_url, extra_params={"custom_headers": per_request_headers}
        )

        call_args = mock_get.call_args
        assert call_args[1]["headers"] == per_request_headers

    @patch("requests.get")
    def test_load_data_with_js_render(
        self, mock_get, api_key, test_url, mock_html_response
    ):
        """Test load_data with JavaScript rendering enabled."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_response.content = mock_html_response.encode()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reader = ZenRowsWebReader(
            api_key=api_key, js_render=True, wait=3000, wait_for=".content"
        )
        documents = reader.load_data(test_url)

        # Verify the request parameters include JS rendering options
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["js_render"] is True
        assert params["wait"] == 3000
        assert params["wait_for"] == ".content"

    @patch("requests.get")
    def test_load_data_with_premium_proxy(
        self, mock_get, api_key, test_url, mock_html_response
    ):
        """Test load_data with premium proxy and geo-location."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_response.content = mock_html_response.encode()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reader = ZenRowsWebReader(
            api_key=api_key, premium_proxy=True, proxy_country="GB"
        )
        documents = reader.load_data(test_url)

        # Verify the request parameters include proxy options
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["premium_proxy"] is True
        assert params["proxy_country"] == "GB"

    @patch("requests.get")
    def test_load_data_error_handling(self, mock_get, api_key, test_url):
        """Test error handling in load_data."""
        # Mock a failed request
        mock_get.side_effect = requests.exceptions.RequestException("Connection failed")

        reader = ZenRowsWebReader(api_key=api_key)
        documents = reader.load_data(test_url)

        # Should return error document instead of raising exception
        assert len(documents) == 1
        assert "Error scraping" in documents[0].text
        assert documents[0].metadata["status"] == "failed"
        assert documents[0].metadata["source_url"] == test_url

    @patch("requests.get")
    def test_load_data_with_extra_params(
        self, mock_get, api_key, test_url, mock_html_response
    ):
        """Test load_data with extra parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_response.content = mock_html_response.encode()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reader = ZenRowsWebReader(api_key=api_key)
        extra_params = {
            "css_extractor": '{"title": "h1", "content": "p"}',
            "autoparse": True,
            "block_resources": "images,fonts",
        }
        documents = reader.load_data(test_url, extra_params=extra_params)

        # Verify extra parameters were included in the request
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["css_extractor"] == '{"title": "h1", "content": "p"}'
        assert params["autoparse"] is True
        assert params["block_resources"] == "images,fonts"

    def test_css_extractor_validation(self, api_key):
        """Test CSS extractor validation."""
        # Valid JSON should work
        reader = ZenRowsWebReader(
            api_key=api_key, css_extractor='{"title": "h1", "content": "p"}'
        )
        assert reader.css_extractor == '{"title": "h1", "content": "p"}'

        # Invalid JSON should raise error
        with pytest.raises(ValueError, match="css_extractor must be valid JSON"):
            ZenRowsWebReader(api_key=api_key, css_extractor="invalid json")

    def test_proxy_country_validation(self, api_key):
        """Test proxy country validation."""
        # Valid two-letter country code should work
        reader = ZenRowsWebReader(api_key=api_key, proxy_country="US")
        assert reader.proxy_country == "US"

        # Invalid country code should raise error
        with pytest.raises(
            ValueError, match="proxy_country must be a two-letter country code"
        ):
            ZenRowsWebReader(api_key=api_key, proxy_country="USA")

    def test_class_name(self, api_key):
        """Test class name method."""
        reader = ZenRowsWebReader(api_key=api_key)
        assert reader.class_name() == "ZenRowsWebReader"

    @patch("requests.get")
    def test_metadata_extraction(self, mock_get, api_key, test_url, mock_html_response):
        """Test metadata extraction from response headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_response.content = mock_html_response.encode()
        mock_response.headers = {
            "Content-Type": "text/html",
            "X-Request-Cost": "2.5",
            "X-Request-Id": "req_123456",
            "Zr-Final-Url": "https://example.com/final",
            "Concurrency-Remaining": "10",
            "Concurrency-Limit": "100",
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        reader = ZenRowsWebReader(api_key=api_key, js_render=True)
        documents = reader.load_data(test_url)

        metadata = documents[0].metadata
        assert metadata["request_cost"] == 2.5
        assert metadata["request_id"] == "req_123456"
        assert metadata["final_url"] == "https://example.com/final"
        assert metadata["concurrency_remaining"] == 10
        assert metadata["concurrency_limit"] == 100
        assert metadata["status_code"] == 200
        assert metadata["content_type"] == "text/html"
        assert metadata["zenrows_config"]["js_render"] is True

    @patch("requests.get")
    def test_auto_js_render_enablement(
        self, mock_get, api_key, test_url, mock_html_response
    ):
        """Test automatic JS render enablement for certain parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_response.content = mock_html_response.encode()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test with screenshot parameter (should auto-enable js_render)
        reader = ZenRowsWebReader(api_key=api_key, screenshot="true")
        documents = reader.load_data(test_url)

        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert params["js_render"] is True  # Should be auto-enabled
        assert params["screenshot"] == "true"
