"""Tests for MediaWikiReader (mwclient-backed version)."""

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch, PropertyMock

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

        mock_site_cls.assert_called_once_with(
            "example.com", path="/w/", scheme="https"
        )

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

class TestFetchContentNamespaceIds:
    """_fetch_content_namespace_ids filters namespaces for content=True."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_filters_content_namespaces(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.get.return_value = {
            "query": {
                "namespaces": {
                    "0": {"id": 0, "*": "", "content": True},
                    "1": {"id": 1, "*": "Talk", "content": False},
                    "4": {"id": 4, "*": "Project", "content": True},
                }
            }
        }
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        ids = reader._fetch_content_namespace_ids()
        assert ids == [0, 4]

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_defaults_to_zero_on_empty(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.get.return_value = {"query": {"namespaces": {}}}
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        assert reader._fetch_content_namespace_ids() == [0]

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_defaults_to_zero_on_api_error(self, mock_site_cls):
        import mwclient.errors

        mock_site = _mock_site()
        mock_site.get.side_effect = mwclient.errors.APIError(
            "error", "info", {}
        )
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        assert reader._fetch_content_namespace_ids() == [0]


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
        mock_page.last_rev_time = (2024, 1, 1, 12, 0, 0, 0, 0, 0)
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
    def test_content_ns_cache(self, mock_site_cls):
        """Content namespace IDs are cached after first call."""
        mock_site = _mock_site()
        mock_site.get.return_value = {
            "query": {
                "namespaces": {
                    "0": {"id": 0, "*": "", "content": True},
                }
            }
        }
        page = MagicMock()
        page.name = "A"
        page.revision = True
        page.last_rev_time = (2024, 1, 1, 0, 0, 0, 0, 0, 0)
        mock_site.allpages.return_value = [page]
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=None)
        list(reader._get_all_pages_generator())
        list(reader._get_all_pages_generator())

        # siteinfo should be called only once
        assert mock_site.get.call_count == 1


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
        mock_site.parse.return_value = {
            "text": {"*": "<p>Test page content.</p>"}
        }
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        result = reader._get_page_contents("Test Page")

        assert result is not None
        assert "Test page content" in result
        assert "<p>" in result  # raw HTML
        mock_site.parse.assert_called_once_with(page="Test Page", prop="text")

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_empty_parse_result(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.parse.return_value = {}
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        assert reader._get_page_contents("Missing") is None

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_api_error_returns_none(self, mock_site_cls):
        import mwclient.errors

        mock_site = _mock_site()
        mock_site.parse.side_effect = mwclient.errors.APIError(
            "error", "info", {}
        )
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        assert reader._get_page_contents("Broken") is None


# ---------------------------------------------------------------------------
# HTML-to-text conversion
# ---------------------------------------------------------------------------

class TestHtmlToCleanText:
    """HTML-to-text conversion (no mocking needed)."""

    def test_basic_html(self):
        result = MediaWikiReader._html_to_clean_text(
            "<p>Hello <b>world</b></p>"
        )
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
        mock_html2text_cls.return_value.handle.side_effect = RuntimeError("html2text fail")
        result = MediaWikiReader._html_to_clean_text("<p>Hello</p> <b>world</b>")
        assert result == "Hello world"

    def test_deeply_nested_malformed_html(self):
        """Deeply nested or malformed HTML is handled by html2text or tag-strip fallback."""
        html = "<div><div><p>Nested <span>text</span></p></div>"
        result = MediaWikiReader._html_to_clean_text(html)
        assert "Nested" in result
        assert "text" in result


# ---------------------------------------------------------------------------
# Resource interface
# ---------------------------------------------------------------------------

class TestResourcesInterface:
    """Public resource-based API."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_load_resource(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.parse.return_value = {"text": {"*": "<p>Content</p>"}}
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        timestamp = datetime(2024, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
        docs = reader.load_resource(
            "P", resource_url="https://wiki.com/P", last_modified=timestamp
        )

        assert len(docs) == 1
        assert docs[0].text == "Content"
        assert docs[0].metadata["url"] == "https://wiki.com/P"
        assert docs[0].metadata["last_modified"] == timestamp.isoformat()
        mock_site.parse.assert_called_once()

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_load_resource_missing_page(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.parse.return_value = {}
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        docs = reader.load_resource(
            "Missing",
            resource_url="https://example.com/wiki/Missing",
            last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert docs == []

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_load_resource_last_modified_none(self, mock_site_cls):
        """When last_modified is None, metadata["last_modified"] is None."""
        mock_site = _mock_site()
        mock_site.parse.return_value = {"text": {"*": "<p>Content</p>"}}
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        docs = reader.load_resource(
            "SomePage",
            resource_url="https://example.com/wiki/SomePage",
            last_modified=None,
        )

        assert len(docs) == 1
        assert docs[0].metadata["last_modified"] is None
        assert docs[0].metadata["url"] == "https://example.com/wiki/SomePage"

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_get_resource_info(self, mock_site_cls):
        mock_site = _mock_site()
        mock_site.get.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "pageid": 123,
                        "title": "Page",
                        "canonicalurl": "https://example.com/wiki/Page",
                        "revisions": [
                            {"timestamp": "2024-06-01T00:00:00Z"}
                        ],
                    }
                }
            }
        }
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        info = reader.get_resource_info("Page")

        assert info["url"] == "https://example.com/wiki/Page"
        assert info["last_modified"] is not None
        assert info["last_modified"].year == 2024
        mock_site.get.assert_called_once()

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_get_resource_info_empty_revisions(self, mock_site_cls):
        """When page exists but revisions list is empty, last_modified is None."""
        mock_site = _mock_site()
        mock_site.get.return_value = {
            "query": {
                "pages": {
                    "1": {
                        "pageid": 1,
                        "title": "Page",
                        "canonicalurl": "https://example.com/wiki/Page",
                        "revisions": [],
                    }
                }
            }
        }
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        info = reader.get_resource_info("Page")

        assert info["url"] == "https://example.com/wiki/Page"
        assert info["last_modified"] is None

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_get_resource_info_missing_page(self, mock_site_cls):
        """When the page does not exist, API returns 'missing'; url and last_modified are None."""
        mock_site = _mock_site()
        mock_site.get.return_value = {
            "query": {
                "pages": {
                    "-1": {
                        "ns": 0,
                        "title": "NonExistent",
                        "missing": "",
                    }
                }
            }
        }
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        info = reader.get_resource_info("NonExistent")

        assert info["url"] is None
        assert info["last_modified"] is None

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_get_resource_info_api_error(self, mock_site_cls):
        """When site.get raises APIError, return url and last_modified as None."""
        import mwclient.errors

        mock_site = _mock_site()
        mock_site.get.side_effect = mwclient.errors.APIError(
            "query-error", "code", {}
        )
        mock_site_cls.return_value = mock_site

        reader = _make_reader()
        info = reader.get_resource_info("Any")

        assert info["url"] is None
        assert info["last_modified"] is None


# ---------------------------------------------------------------------------
# lazy_load_data
# ---------------------------------------------------------------------------

class TestLazyLoadData:
    """End-to-end integration via lazy_load_data."""

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_yields_documents(self, mock_site_cls):
        mock_site = _mock_site()

        # allpages returns mwclient Page objects
        page = MagicMock()
        page.name = "Test Page"
        page.revision = True
        page.last_rev_time = (2024, 1, 1, 12, 0, 0, 0, 0, 0)
        mock_site.allpages.return_value = [page]

        # parse returns HTML
        mock_site.parse.return_value = {
            "text": {"*": "<p>Hello world</p>"}
        }

        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0])
        docs = list(reader.lazy_load_data())

        assert len(docs) == 1
        assert "Hello world" in docs[0].text
        assert docs[0].metadata["title"] == "Test Page"
        assert "example.com" in docs[0].metadata["url"]

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_uses_fallback_url_when_siteinfo_has_no_base(self, mock_site_cls):
        """When siteinfo has no base URL, lazy_load_data uses scheme/host/path fallback."""
        mock_site = _mock_site()
        mock_site.site = {}  # no "base" -> url_base is None in generator
        page = MagicMock()
        page.name = "NoURL Page"
        page.revision = True
        page.last_rev_time = (2024, 1, 1, 12, 0, 0, 0, 0, 0)
        mock_site.allpages.return_value = [page]
        mock_site.parse.return_value = {"text": {"*": "<p>Content</p>"}}
        mock_site_cls.return_value = mock_site

        reader = _make_reader(host="wiki.example.com", namespaces=[0])
        docs = list(reader.lazy_load_data())

        assert len(docs) == 1
        assert docs[0].metadata["title"] == "NoURL Page"
        # Fallback URL from reader config
        assert docs[0].metadata["url"] == (
            "https://wiki.example.com/w/index.php?title=NoURL_Page"
        )

    @patch("llama_index.readers.mediawiki.base.mwclient.Site")
    def test_skips_page_when_get_page_contents_returns_none(self, mock_site_cls):
        """When _get_page_contents returns None for a page, no document is yielded for it."""
        mock_site = _mock_site()

        page_ok = MagicMock()
        page_ok.name = "PageWithContent"
        page_ok.revision = True
        page_ok.last_rev_time = (2024, 1, 1, 12, 0, 0, 0, 0, 0)

        page_skip = MagicMock()
        page_skip.name = "PageWithoutContent"
        page_skip.revision = True
        page_skip.last_rev_time = (2024, 1, 2, 12, 0, 0, 0, 0, 0)

        mock_site.allpages.return_value = [page_ok, page_skip]
        mock_site.parse.side_effect = [
            {"text": {"*": "<p>Only this page has content</p>"}},
            {},  # second page: no content -> load_resource returns []
        ]
        mock_site_cls.return_value = mock_site

        reader = _make_reader(namespaces=[0])
        docs = list(reader.lazy_load_data())

        assert len(docs) == 1
        assert docs[0].metadata["title"] == "PageWithContent"
        assert "Only this page has content" in docs[0].text
