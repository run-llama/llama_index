"""Tests for MediaWikiReader (mwclient-backed version)."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from llama_index.readers.mediawiki import MediaWikiReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reader(**overrides):
    """Create a MediaWikiReader with sensible defaults (mwclient.Site is mocked)."""
    kwargs = {"host": "example.com"}
    kwargs.update(overrides)
    return MediaWikiReader(**kwargs)


def _mock_site():
    """Create a mock mwclient.Site."""
    site = MagicMock()
    site.site = {
        "base": "https://example.com/wiki/Main_Page",
        "articlepath": "/wiki/$1",
    }
    return site


# ---------------------------------------------------------------------------
# Construction & Config
# ---------------------------------------------------------------------------


class TestMediaWikiReaderInit:
    """Construction and config validation."""

    def test_class(self):
        """MediaWikiReader must inherit from the LlamaIndex base reader."""
        names_of_base_classes = [b.__name__ for b in MediaWikiReader.__mro__]
        assert "BasePydanticReader" in names_of_base_classes
        assert "BaseReader" in names_of_base_classes

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_is_remote(self, _mock_site_cls):
        reader = _make_reader()
        assert reader.is_remote is True

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_missing_host_raises(self, _mock_site_cls):
        with pytest.raises(ValidationError, match="host"):
            MediaWikiReader(host="")

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_zero_page_limit_raises(self, _mock_site_cls):
        with pytest.raises(ValidationError, match="page_limit"):
            _make_reader(page_limit=0)

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_negative_page_limit_raises(self, _mock_site_cls):
        with pytest.raises(ValidationError, match="page_limit"):
            _make_reader(page_limit=-1)

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_defaults(self, _mock_site_cls):
        reader = _make_reader()
        assert reader.host == "example.com"
        assert reader.path == "/w/"
        assert reader.scheme == "https"
        assert reader.page_limit == 500
        assert reader.namespaces is None
        assert reader.filter_redirects is True

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_logger_injection(self, _mock_site_cls):
        custom = logging.getLogger("custom.mediawiki")
        reader = _make_reader(logger=custom)
        assert reader.logger is custom


# ---------------------------------------------------------------------------
# Site property (lazy creation)
# ---------------------------------------------------------------------------


class TestSiteProperty:
    """The .site property lazily creates an mwclient.Site."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_creates_site_lazily(self, mock_site_cls):
        reader = _make_reader()
        mock_site_cls.assert_not_called()

        _ = reader.site

        mock_site_cls.assert_called_once_with("example.com", path="/w/", scheme="https")

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_returns_same_instance(self, mock_site_cls):
        reader = _make_reader()
        site1 = reader.site
        site2 = reader.site
        assert site1 is site2
        assert mock_site_cls.call_count == 1


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:
    """Authentication via clientlogin."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_login_calls_clientlogin(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        reader.login("testuser", "testpass")

        mock_site.clientlogin.assert_called_once_with(
            username="testuser", password="testpass"
        )

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_login_failure_raises(self, mock_site_cls):
        import mwclient.errors

        mock_site = _mock_site()
        mock_site.clientlogin.side_effect = mwclient.errors.LoginError(
            mock_site, "FAIL", "Bad credentials"
        )
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        with pytest.raises(mwclient.errors.LoginError):
            reader.login("bad", "creds")

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_login_with_bot_password_format(self, mock_site_cls):
        """Bot password uses User@BotName as username; passes through to mwclient."""
        mock_site = _mock_site()
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        reader.login("User@BotName", "bot-password-token")

        mock_site.clientlogin.assert_called_once_with(
            username="User@BotName", password="bot-password-token"
        )


# ---------------------------------------------------------------------------
# Content namespace discovery
# ---------------------------------------------------------------------------


class TestGetContentNamespaceIds:
    """_get_content_namespace_ids filters namespaces for content flag."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_filters_content_namespaces(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.get.return_value = {
            "query": {
                "namespaces": {
                    "0": {"id": 0, "*": "", "content": ""},
                    "1": {"id": 1, "*": "Talk"},
                    "4": {"id": 4, "*": "Project", "content": ""},
                }
            }
        }
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        ids = reader._get_content_namespace_ids()
        assert ids == [0, 4]

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_defaults_to_zero_on_empty(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.get.return_value = {"query": {"namespaces": {}}}
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        assert reader._get_content_namespace_ids() == [0]

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_raises_on_api_error(self, mock_site_cls):
        import mwclient.errors

        mock_site = _mock_site()
        mock_site.get.side_effect = mwclient.errors.APIError("error", "info", {})
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        with pytest.raises(mwclient.errors.APIError):
            reader._get_content_namespace_ids()


# ---------------------------------------------------------------------------
# All-pages generator
# ---------------------------------------------------------------------------


class TestGetAllPages:
    """Page listing via mwclient's allpages."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_iterates_pages(self, mock_site_cls):
        mock_site = _mock_site()
        mock_page = MagicMock()
        mock_page.name = "Page 1"
        mock_page.revision = True
        mock_page.touched = (2024, 1, 1, 12, 0, 0, 0, 0, 0)
        mock_site.allpages.return_value = [mock_page]
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0])
        pages = list(reader._get_all_pages_generator())

        assert len(pages) == 1
        assert pages[0]["title"] == "Page 1"
        assert pages[0]["url"] == "https://example.com/wiki/Page_1"
        assert pages[0]["last_modified"].year == 2024

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_multiple_namespaces(self, mock_site_cls):
        mock_site = _mock_site()
        page_ns0 = MagicMock()
        page_ns0.name = "Main Page"
        page_ns0.revision = True
        page_ns0.last_rev_time = (2024, 1, 1, 0, 0, 0, 0, 0, 0)

        page_ns4 = MagicMock()
        page_ns4.name = "Project:About"
        page_ns4.revision = True
        page_ns4.last_rev_time = (2024, 2, 1, 0, 0, 0, 0, 0, 0)

        mock_site.allpages.side_effect = [[page_ns0], [page_ns4]]
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0, 4])
        pages = list(reader._get_all_pages_generator())

        assert len(pages) == 2
        assert pages[0]["title"] == "Main Page"
        assert pages[1]["title"] == "Project:About"
        assert mock_site.allpages.call_count == 2

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_auto_discovers_content_namespaces(self, mock_site_cls):
        """When namespaces is None, calls siteinfo to find content namespaces."""
        mock_site = _mock_site()
        mock_site.get.return_value = {
            "query": {
                "namespaces": {
                    "0": {"id": 0, "*": "", "content": True},
                }
            }
        }
        page = MagicMock()
        page.name = "Test"
        page.revision = True
        page.last_rev_time = (2024, 6, 1, 0, 0, 0, 0, 0, 0)
        mock_site.allpages.return_value = [page]
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=None)
        pages = list(reader._get_all_pages_generator())

        mock_site.get.assert_called_once()
        assert len(pages) == 1

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_filter_redirects_default(self, mock_site_cls):
        """filterredir='nonredirects' is passed by default."""
        mock_site = _mock_site()
        mock_site.allpages.return_value = []
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0])
        list(reader._get_all_pages_generator())

        call_kwargs = mock_site.allpages.call_args.kwargs
        assert call_kwargs.get("filterredir") == "nonredirects"

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_filter_redirects_disabled(self, mock_site_cls):
        """filterredir='all' is passed when filter_redirects=False."""
        mock_site = _mock_site()
        mock_site.allpages.return_value = []
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0], filter_redirects=False)
        list(reader._get_all_pages_generator())

        call_kwargs = mock_site.allpages.call_args.kwargs
        assert call_kwargs.get("filterredir") == "all"

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_raises_when_siteinfo_base_missing(self, mock_site_cls):
        """RuntimeError is raised when siteinfo has no base URL."""
        mock_site = _mock_site()
        mock_site.site = {}  # no "base" key
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0])
        with pytest.raises(RuntimeError, match="site URL base"):
            list(reader._get_all_pages_generator())


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


class TestBuildPageUrl:
    """_build_page_url builds canonical URLs from title and site base."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_special_characters_in_title(self, _mock_site_cls):
        """Titles with & and spaces are normalized (spaces to underscores)."""
        reader = _make_reader()
        url_base = ("https://example.com", "/wiki/$1")
        result = reader._build_page_url("Page & FAQ", url_base)
        assert result == "https://example.com/wiki/Page_&_FAQ"


# ---------------------------------------------------------------------------
# Page content retrieval
# ---------------------------------------------------------------------------


class TestGetPageContents:
    """Content retrieval via Site.parse()."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_success(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.get.return_value = {"parse": {"text": {"*": "<p>Test page content.</p>"}}}
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        result = reader._get_page_contents("Test Page")

        assert result is not None
        assert "Test page content" in result
        assert "<p>" in result  # raw HTML
        mock_site.get.assert_called_once_with(
            "parse",
            page="Test Page",
            prop="text",
            disablelimitreport=True,
            disableeditsection=True,
            disabletoc=True,
        )

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_empty_parse_result(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.get.return_value = {}
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        assert reader._get_page_contents("Missing") is None

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_api_error_returns_none(self, mock_site_cls):
        import mwclient.errors

        mock_site = _mock_site()
        mock_site.get.side_effect = mwclient.errors.APIError("error", "info", {})
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        assert reader._get_page_contents("Broken") is None


# ---------------------------------------------------------------------------
# HTML-to-text conversion
# ---------------------------------------------------------------------------


class TestHtmlToCleanText:
    """HTML-to-text conversion (no mocking needed)."""

    def test_basic_html(self):
        result = MediaWikiReader._html_to_clean_text("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result

    def test_preserves_structure(self):
        html = (
            "<h1>Title</h1>"
            "<p>Paragraph with <em>emphasis</em> and <strong>strong</strong>.</p>"
            "<ul><li>Item 1</li><li>Item 2</li></ul>"
        )
        result = MediaWikiReader._html_to_clean_text(html)
        assert "Title" in result
        assert "Item 1" in result
        assert "Item 2" in result

    @patch("llama_index.readers.mediawiki.base.html2text.HTML2Text")
    def test_fallback_when_html2text_raises(self, mock_html2text_cls):
        """When html2text raises, fallback strips tags with regex and normalizes space."""
        mock_html2text_cls.return_value.handle.side_effect = RuntimeError(
            "html2text fail"
        )
        result = MediaWikiReader._html_to_clean_text("<p>Hello</p> <b>world</b>")
        assert result == "Hello world"

    def test_deeply_nested_malformed_html(self):
        """Deeply nested or malformed HTML is handled by html2text or tag-strip fallback."""
        html = "<div><div><p>Nested <span>text</span></p></div>"
        result = MediaWikiReader._html_to_clean_text(html)
        assert "Nested" in result
        assert "text" in result


# ---------------------------------------------------------------------------
# lazy_load_data
# ---------------------------------------------------------------------------


class TestLazyLoadData:
    """End-to-end integration via lazy_load_data."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_yields_documents(self, mock_site_cls):
        mock_site = _mock_site()

        page = MagicMock()
        page.name = "Test Page"
        page.revision = True
        page.touched = (2024, 1, 1, 12, 0, 0, 0, 0, 0)
        mock_site.allpages.return_value = [page]

        mock_site.get.return_value = {"parse": {"text": {"*": "<p>Hello world</p>"}}}

        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0])
        docs = list(reader.lazy_load_data())

        assert len(docs) == 1
        assert "Hello world" in docs[0].text
        assert docs[0].metadata["title"] == "Test Page"
        assert "example.com" in docs[0].metadata["url"]

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_raises_when_siteinfo_base_missing(self, mock_site_cls):
        """When siteinfo has no base URL, lazy_load_data raises RuntimeError."""
        mock_site = _mock_site()
        mock_site.site = {}  # no "base"
        mock_site.allpages.return_value = []
        mock_site_cls.return_value = mock_site

        reader = _make_reader(host="wiki.example.com", namespaces=[0])
        with pytest.raises(RuntimeError, match="site URL base"):
            list(reader.lazy_load_data())

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_skips_page_when_get_page_contents_returns_none(self, mock_site_cls):
        """When _get_page_contents returns None for a page, no document is yielded for it."""
        mock_site = _mock_site()

        page_ok = MagicMock()
        page_ok.name = "PageWithContent"
        page_ok.revision = True
        page_ok.touched = (2024, 1, 1, 12, 0, 0, 0, 0, 0)

        page_skip = MagicMock()
        page_skip.name = "PageWithoutContent"
        page_skip.revision = True
        page_skip.touched = (2024, 1, 2, 12, 0, 0, 0, 0, 0)

        mock_site.allpages.return_value = [page_ok, page_skip]
        mock_site.get.side_effect = [
            {"parse": {"text": {"*": "<p>Only this page has content</p>"}}},
            {},  # second page: no content
        ]
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0])
        docs = list(reader.lazy_load_data())

        assert len(docs) == 1
        assert docs[0].metadata["title"] == "PageWithContent"
        assert "Only this page has content" in docs[0].text
