"""
MediaWiki reader for LlamaIndex.

Provides a LlamaIndex-compatible reader that fetches and converts pages from
any MediaWiki instance into LlamaIndex Documents, using mwclient for all
MediaWiki API interactions.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import html2text
import mwclient
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

_internal_logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Lightweight record for a single wiki page entry from allpages."""

    title: str
    url: Optional[str]
    last_modified: Optional[datetime]
    pageid: Optional[int]
    namespace: Optional[int]


class MediaWikiReader(BasePydanticReader):
    """
    LlamaIndex reader for MediaWiki instances.

    Fetches pages from a MediaWiki site, converts HTML content to clean text,
    and returns LlamaIndex Documents with metadata (title, URL, last_modified).

    Uses `mwclient` for all API interactions. Supports authentication via
    `login`.

    Example:
        reader = MediaWikiReader(host="en.wikipedia.org")
        docs = reader.load_data()

    With authentication:

        reader = MediaWikiReader(host="my.private.wiki")
        reader.login("username", "password")
        docs = reader.load_data()

    Implements BasePydanticReader (for serialization / LlamaHub compatibility).

    """

    FILTERREDIR_NONREDIRECTS: str = "nonredirects"
    FILTERREDIR_ALL: str = "all"

    is_remote: bool = True
    model_config = {"arbitrary_types_allowed": True}

    # -- Pydantic fields (serialisable config) --------------------------------

    host: str = Field(
        min_length=1,
        description="MediaWiki site hostname (e.g. 'en.wikipedia.org')",
    )
    path: str = Field(
        default="/w/",
        description="MediaWiki script path (default '/w/')",
    )
    scheme: Literal["https", "http"] = Field(
        default="https",
        description="URL scheme: 'https' or 'http'",
    )
    page_limit: int = Field(
        default=500,
        gt=0,
        description=(
            "Max page titles per allpages API call. Pagination continues "
            "until the wiki is fully listed."
        ),
    )
    namespaces: Optional[List[int]] = Field(
        default=None,
        description=(
            "Namespace IDs to list. None = wiki content namespaces from "
            "siteinfo (i.e. $wgContentNamespaces). Set explicitly to override."
        ),
    )
    filter_redirects: bool = Field(
        default=True,
        description=(
            "If True (default), redirect pages are excluded from results. "
            "Set to False to include redirect pages."
        ),
    )
    logger: logging.Logger = Field(
        default_factory=lambda: _internal_logger,
        description="Logger instance (injectable for tests or custom logging)",
        exclude=True,
    )

    # -- Non-serialised internal state ----------------------------------------
    _site: Optional[mwclient.Site] = PrivateAttr(default=None)

    # -- Construction helpers -------------------------------------------------

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.logger.info(
            "Initialized MediaWikiReader for %s://%s%s",
            self.scheme,
            self.host,
            self.path,
        )

    # -- mwclient Site lifecycle ----------------------------------------------

    @property
    def site(self) -> mwclient.Site:
        """Return the mwclient Site, creating one lazily if needed."""
        if self._site is None:
            self._site = mwclient.Site(
                self.host, path=self.path, scheme=self.scheme
            )
        return self._site

    def login(self, username: str, password: str) -> None:
        """
        Authenticate to the wiki using a user or bot credentials.

        Args:
            username: MediaWiki username or bot username.
            password: MediaWiki password or bot password.

        Raises:
            mwclient.errors.LoginError: If login fails.

        """
        self.site.login(username=username, password=password)
        self.logger.info("Logged in as %s", username)

    # -- Internal helpers -----------------------------------------------------

    def _get_content_namespace_ids(self) -> List[int]:
        """
        Return namespace IDs marked as content ($wgContentNamespaces).

        Queries ``action=query&meta=siteinfo&siprop=namespaces`` and filters
        for namespaces that carry the ``content`` flag.

        Raises:
            mwclient.errors.APIError: If the siteinfo query fails.
            RuntimeError: If no content namespaces are found in siteinfo.

        """
        result = self.site.get("query", meta="siteinfo", siprop="namespaces")
        namespaces = result.get("query", {}).get("namespaces", {})
        ids: List[int] = [
            int(ns_data["id"])
            for ns_data in namespaces.values()
            if isinstance(ns_data, dict)
            and "content" in ns_data
            and ns_data.get("id") is not None
        ]

        if not ids:
            raise RuntimeError(
                "No content namespaces found in siteinfo; "
                "the MediaWiki API may be unreachable or misconfigured."
            )
        return sorted(ids)

    def _resolve_namespace_list(self) -> List[int]:
        """Return the effective list of namespace IDs to iterate."""
        if self.namespaces is not None:
            return self.namespaces
        return self._get_content_namespace_ids()

    def _get_url_base(self) -> Tuple[str, str]:
        """
        Return ``(origin, article_path)`` parsed from the site's siteinfo.

        Raises:
            RuntimeError: If the URL base cannot be determined from siteinfo.

        """
        site_info = self.site.site
        base_url = site_info.get("base", "")
        article_path = site_info.get("articlepath", "/wiki/$1")
        if not base_url:
            raise RuntimeError(
                "Could not determine base URL from siteinfo; "
                "the 'base' field is missing or empty."
            )
        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        return origin, article_path

    def _build_page_url(
        self, title: str, url_base: Optional[Tuple[str, str]]
    ) -> Optional[str]:
        """Build canonical page URL from pre-parsed (origin, article_path)."""
        if not url_base:
            return None
        origin, article_path = url_base
        return origin + article_path.replace("$1", title.replace(" ", "_"))

    def _extract_revision_time(
        self, page: Any, title: str
    ) -> Optional[datetime]:
        """Extract last_modified from page revision timestamp (struct_time)."""
        try:
            ts = page.touched
            if ts:
                return datetime(*ts[:6], tzinfo=timezone.utc)
            return None
        except (AttributeError, TypeError) as e:
            self.logger.debug(
                "Revision timestamp extraction failed for page %r: %s",
                title,
                e,
                exc_info=True,
            )
            return None

    def _get_all_pages_generator(self) -> Iterator[Page]:
        """Yield a Page record for every page in the wiki via mwclient's allpages."""
        ns_list = self._resolve_namespace_list()

        # Parse site base URL once; origin + article_path are constant per site
        url_base = self._get_url_base()

        filterredir = (
            self.FILTERREDIR_NONREDIRECTS
            if self.filter_redirects
            else self.FILTERREDIR_ALL
        )

        for ns in ns_list:
            for page in self.site.allpages(
                namespace=ns,
                generator=True,
                api_chunk_size=self.page_limit,
                filterredir=filterredir,
            ):
                title = page.name
                yield Page(
                    title=title,
                    url=self._build_page_url(title, url_base),
                    last_modified=self._extract_revision_time(page, title),
                    pageid=page.pageid,
                    namespace=page.namespace,
                )

    def _get_page_contents(self, page_title: str) -> Optional[str]:
        """
        Fetch parsed HTML content for a page via mwclient's parse API.

        Returns:
            Raw HTML from the API, or ``None`` on failure.

        """
        try:
            result = self.site.post(
                "parse",
                page=page_title,
                prop="text",
                disablelimitreport=True,
                disableeditsection=True,
                disabletoc=True,
            )
        except mwclient.errors.APIError as exc:
            self.logger.warning(
                "Parse API failed for '%s': %s", page_title, exc
            )
            return None

        if not result:
            self.logger.warning("No parse result for page '%s'", page_title)
            return None

        html_content = result.get("parse", {}).get("text", {}).get("*", "")
        if not html_content:
            self.logger.warning(
                "No content in parse result for page '%s'", page_title
            )
            return None

        return html_content

    @staticmethod
    def _html_to_clean_text(html_content: str) -> str:
        """Convert MediaWiki HTML to clean Markdown text."""
        try:
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.body_width = 0
            h.ul_item_mark = "-"
            h.emphasis_mark = "*"
            h.strong_mark = "**"
            return h.handle(html_content).strip()
        except (AttributeError, TypeError, ValueError) as e:
            _internal_logger.debug(
                "html2text failed, using tag-strip fallback: %s",
                e,
                exc_info=True,
            )
            return re.sub(
                r"\s+", " ", re.sub(r"<[^>]+>", "", html_content)
            ).strip()

    def _page_to_document(
        self,
        title: str,
        url: Optional[str],
        last_modified: Optional[datetime],
        pageid: Optional[int] = None,
        namespace: Optional[int] = None,
    ) -> Optional[Document]:
        """
        Fetch and convert a single wiki page into a Document.

        Args:
            title: Page title.
            url: Canonical URL for the page (may be None).
            last_modified: Last-modified timestamp (may be None).
            pageid: Numeric page ID (may be None).
            namespace: Namespace ID for the page (may be None).

        Returns:
            A Document, or None if the page content could not be fetched.
        """
        content = self._get_page_contents(title)
        if not content:
            return None

        text = self._html_to_clean_text(content)
        return Document(
            text=text,
            id_=f"mediawiki:{title}",
            metadata={
                "title": title,
                "url": url,
                "last_modified": (last_modified.isoformat() if last_modified else None),
                "pageid": pageid,
                "namespace": namespace,
            },
        )

    # -- BasePydanticReader / BaseReader interface -----------------------------

    def lazy_load_data(self, *args: Any, **kwargs: Any) -> Iterator[Document]:
        """
        Yield one Document per page in the wiki.

        Iterates all pages via mwclient's allpages, then fetches parsed
        content for each page with one parse API call per page (MediaWiki
        has no batch parse API; see class docstring for scale notes).
        """
        for page_record in self._get_all_pages_generator():
            doc = self._page_to_document(
                title=page_record.title,
                url=page_record.url,
                last_modified=page_record.last_modified,
                pageid=page_record.pageid,
                namespace=page_record.namespace,
            )
            if doc is not None:
                yield doc
