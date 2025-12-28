"""Tests for MassiveWebReader."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_index.readers.massive.base import MassiveWebReader
from llama_index.readers.massive.country_config import COUNTRY_CONFIG


# --- Fixtures ---


@pytest.fixture
def mock_sync_playwright():
    """Fixture for sync playwright mock setup."""
    with patch("llama_index.readers.massive.base.sync_playwright") as mock_pw:
        mock_page = MagicMock()
        mock_page.content.return_value = "<html><body>Test</body></html>"

        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context

        mock_pw_instance = MagicMock()
        mock_pw_instance.chromium.launch.return_value = mock_browser

        mock_pw.return_value.__enter__.return_value = mock_pw_instance

        yield {"page": mock_page, "browser": mock_browser, "context": mock_context}


@pytest.fixture
def mock_async_playwright():
    """Fixture for async playwright mock setup."""
    with patch("llama_index.readers.massive.base.async_playwright") as mock_pw:
        mock_page = AsyncMock()
        mock_page.content.return_value = "<html><body>Test</body></html>"

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = AsyncMock()
        mock_browser.new_context.return_value = mock_context

        mock_pw_instance = AsyncMock()
        mock_pw_instance.chromium.launch.return_value = mock_browser

        mock_pw.return_value.__aenter__.return_value = mock_pw_instance

        yield {"page": mock_page, "browser": mock_browser, "context": mock_context}


# --- Init Tests ---


class TestMassiveWebReaderInit:
    """Test MassiveWebReader initialization."""

    def test_init_with_credentials(self):
        """Test initialization with credentials."""
        reader = MassiveWebReader(
            username="test_user",
            password="test_pass",
            country="US",
        )
        assert reader.username == "test_user"
        assert reader.password == "test_pass"
        assert reader.country == "US"
        assert reader._proxy is not None
        assert "test_user-country-US" in reader._proxy["username"]

    def test_init_requires_credentials(self):
        """Test that credentials are required."""
        with pytest.raises(Exception):
            MassiveWebReader()

    def test_init_with_all_geotargeting(self):
        """Test initialization with all geotargeting parameters."""
        reader = MassiveWebReader(
            username="test_user",
            password="test_pass",
            country="US",
            city="New York",
            zipcode="10001",
            asn="12345",
            device_type="mobile",
            session="session-123",
            ttl=30,
        )
        proxy_username = reader._proxy["username"]
        assert "-country-US" in proxy_username
        assert "-city-New York" in proxy_username
        assert "-zipcode-10001" in proxy_username
        assert "-asn-12345" in proxy_username
        assert "-type-mobile" in proxy_username
        assert "-session-session-123" in proxy_username
        assert "-sessionttl-30" in proxy_username

    def test_init_invalid_device_type(self):
        """Test that invalid device_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device_type"):
            MassiveWebReader(
                username="test_user",
                password="test_pass",
                device_type="invalid",
            )

    def test_init_valid_device_types(self):
        """Test that valid device_types are accepted."""
        for device_type in ["mobile", "common", "tv"]:
            reader = MassiveWebReader(
                username="test_user",
                password="test_pass",
                device_type=device_type,
            )
            assert reader.device_type == device_type

    def test_class_name(self):
        """Test class_name method."""
        assert MassiveWebReader.class_name() == "MassiveWebReader"

    def test_is_remote(self):
        """Test is_remote attribute."""
        reader = MassiveWebReader(username="test_user", password="test_pass")
        assert reader.is_remote is True


# --- Country Config Tests ---


class TestCountryConfig:
    """Test country configuration."""

    def test_country_config_has_default(self):
        """Test that DEFAULT fallback exists."""
        assert "DEFAULT" in COUNTRY_CONFIG
        assert COUNTRY_CONFIG["DEFAULT"]["locale"] == "en-US"
        assert COUNTRY_CONFIG["DEFAULT"]["timezone"] == "UTC"

    def test_country_config_us(self):
        """Test US configuration."""
        assert "US" in COUNTRY_CONFIG
        assert COUNTRY_CONFIG["US"]["locale"] == "en-US"
        assert COUNTRY_CONFIG["US"]["timezone"] == "America/New_York"

    def test_country_config_count(self):
        """Test that sufficient countries are included."""
        assert len(COUNTRY_CONFIG) > 250

    def test_country_config_structure(self):
        """Test that all entries have required keys."""
        for code, config in COUNTRY_CONFIG.items():
            assert "locale" in config, f"Missing locale for {code}"
            assert "timezone" in config, f"Missing timezone for {code}"


# --- Sync Load Data Tests ---


class TestMassiveWebReaderLoadData:
    """Test MassiveWebReader data loading."""

    def test_load_data_success(self, mock_sync_playwright):
        """Test successful data loading."""
        mock_sync_playwright[
            "page"
        ].content.return_value = "<html><body><p>Test content</p></body></html>"

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = reader.load_data(["https://example.com"])

        assert len(docs) == 1
        assert "Test content" in docs[0].text
        assert docs[0].metadata["url"] == "https://example.com"
        mock_sync_playwright["page"].goto.assert_called_once()
        mock_sync_playwright["browser"].close.assert_called_once()

    def test_load_data_raw_html(self, mock_sync_playwright):
        """Test raw HTML mode returns unprocessed content."""
        html_content = "<html><body><script>alert(1)</script><p>Test</p></body></html>"
        mock_sync_playwright["page"].content.return_value = html_content

        reader = MassiveWebReader(
            username="test_user",
            password="test_pass",
            raw_html=True,
        )
        docs = reader.load_data(["https://example.com"])

        assert len(docs) == 1
        assert "<script>" in docs[0].text

    def test_load_data_strips_scripts(self, mock_sync_playwright):
        """Test that scripts are stripped in default mode."""
        html_content = "<html><body><script>alert(1)</script><p>Test</p></body></html>"
        mock_sync_playwright["page"].content.return_value = html_content

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = reader.load_data(["https://example.com"])

        assert len(docs) == 1
        assert "<script>" not in docs[0].text
        assert "alert" not in docs[0].text
        assert "Test" in docs[0].text

    def test_load_data_multiple_urls(self, mock_sync_playwright):
        """Test loading multiple URLs."""
        mock_sync_playwright["page"].content.side_effect = [
            "<html><body>Page 1</body></html>",
            "<html><body>Page 2</body></html>",
        ]

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = reader.load_data(["https://example1.com", "https://example2.com"])

        assert len(docs) == 2
        assert "Page 1" in docs[0].text
        assert "Page 2" in docs[1].text

    def test_load_data_with_additional_wait(self, mock_sync_playwright):
        """Test that additional_wait_ms triggers wait_for_timeout."""
        reader = MassiveWebReader(
            username="test_user",
            password="test_pass",
            additional_wait_ms=2000,
        )
        reader.load_data(["https://example.com"])

        mock_sync_playwright["page"].wait_for_timeout.assert_called_once_with(2000)

    def test_load_data_with_country_config(self, mock_sync_playwright):
        """Test that country config is applied."""
        reader = MassiveWebReader(
            username="test_user",
            password="test_pass",
            country="US",
        )
        reader.load_data(["https://example.com"])

        # Verify browser context was created with locale/timezone
        mock_sync_playwright["browser"].new_context.assert_called_once()


# --- Async Load Data Tests ---


class TestMassiveWebReaderAsync:
    """Test async data loading."""

    @pytest.mark.asyncio
    async def test_aload_data_success(self, mock_async_playwright):
        """Test async data loading."""
        mock_async_playwright[
            "page"
        ].content.return_value = "<html><body>Async Test</body></html>"

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = await reader.aload_data(["https://example.com"])

        assert len(docs) == 1
        assert "Async Test" in docs[0].text

    @pytest.mark.asyncio
    async def test_aload_data_with_additional_wait(self, mock_async_playwright):
        """Test async additional_wait_ms triggers wait_for_timeout."""
        reader = MassiveWebReader(
            username="test_user",
            password="test_pass",
            additional_wait_ms=2000,
        )
        await reader.aload_data(["https://example.com"])

        mock_async_playwright["page"].wait_for_timeout.assert_called_once_with(2000)

    @pytest.mark.asyncio
    async def test_aload_data_handles_timeout(self, mock_async_playwright):
        """Test async timeout handling."""
        from playwright.async_api import TimeoutError as AsyncPlaywrightTimeout

        mock_async_playwright["page"].goto.side_effect = AsyncPlaywrightTimeout(
            "Timeout"
        )

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = await reader.aload_data(["https://slow-site.com"])

        assert docs == []
        mock_async_playwright["browser"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aload_data_handles_playwright_error(self, mock_async_playwright):
        """Test async PlaywrightError handling."""
        from playwright.async_api import Error as AsyncPlaywrightError

        mock_async_playwright["page"].goto.side_effect = AsyncPlaywrightError(
            "Connection failed"
        )

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = await reader.aload_data(["https://error-site.com"])

        assert docs == []
        mock_async_playwright["browser"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aload_data_handles_unexpected_error(self, mock_async_playwright):
        """Test async unexpected error handling."""
        mock_async_playwright["page"].goto.side_effect = Exception("Unexpected error")

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = await reader.aload_data(["https://error-site.com"])

        assert docs == []
        mock_async_playwright["browser"].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_aload_data_browser_close_error(self, mock_async_playwright):
        """Test that async browser close errors are handled gracefully."""
        mock_async_playwright["browser"].close.side_effect = Exception(
            "Browser close failed"
        )

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = await reader.aload_data(["https://example.com"])

        assert len(docs) == 1
        mock_async_playwright["browser"].close.assert_called_once()


# --- Error Handling Tests ---


class TestMassiveWebReaderErrorHandling:
    """Test error handling."""

    def test_handles_timeout(self, mock_sync_playwright):
        """Test that timeout errors are handled gracefully."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        mock_sync_playwright["page"].goto.side_effect = PlaywrightTimeout("Timeout")

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = reader.load_data(["https://slow-site.com"])

        assert docs == []
        mock_sync_playwright["browser"].close.assert_called_once()

    def test_handles_playwright_error(self, mock_sync_playwright):
        """Test PlaywrightError (non-timeout) handling."""
        from playwright.sync_api import Error as PlaywrightError

        mock_sync_playwright["page"].goto.side_effect = PlaywrightError(
            "Connection refused"
        )

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = reader.load_data(["https://error-site.com"])

        assert docs == []
        mock_sync_playwright["browser"].close.assert_called_once()

    def test_handles_unexpected_error(self, mock_sync_playwright):
        """Test that unexpected errors are handled gracefully."""
        mock_sync_playwright["page"].goto.side_effect = Exception("Unexpected error")

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = reader.load_data(["https://error-site.com"])

        assert docs == []
        mock_sync_playwright["browser"].close.assert_called_once()

    def test_browser_close_error_handled(self, mock_sync_playwright):
        """Test that browser close errors are handled gracefully."""
        mock_sync_playwright["browser"].close.side_effect = Exception(
            "Browser close failed"
        )

        reader = MassiveWebReader(username="test_user", password="test_pass")
        docs = reader.load_data(["https://example.com"])

        assert len(docs) == 1
        mock_sync_playwright["browser"].close.assert_called_once()
