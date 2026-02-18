"""MediaWiki reader for LlamaIndex.

Provides a LlamaIndex-compatible reader that fetches and converts pages from
any MediaWiki instance into LlamaIndex Documents, using mwclient for all
MediaWiki API interactions.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import html2text
import mwclient

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

_internal_logger = logging.getLogger(__name__)


class MediaWikiReader(BasePydanticReader):
    """LlamaIndex reader for MediaWiki instances.

    Fetches pages from a MediaWiki site, converts HTML content to clean text,
    and returns LlamaIndex Documents with metadata (title, URL, last_modified).

    Uses `mwclient` for all API interactions. Supports authentication via
    :meth:`login`.

    **Ingestion scale:** Full ingestion (e.g. :meth:`lazy_load_data`) lists
    pages via the allpages API, then fetches parsed content with one
    ``parse`` API call per page. MediaWiki's parse API does not support
    batch requests, so large wikis require many HTTP requests. This is a
    known limitation of the MediaWiki API.

    Example::

        reader = MediaWikiReader(host="en.wikipedia.org")
        docs = list(reader.lazy_load_data())

    With authentication::

        reader = MediaWikiReader(host="my.private.wiki")
        reader.login("username", "password")
        docs = list(reader.lazy_load_data())

    Implements BasePydanticReader (for serialization / LlamaHub compatibility)
    and provides load_resource and get_resource_info for resource-based use.
    """

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
            "siteinfo API (i.e. $wgContentNamespaces). Set explicitly to "
            "override."
        ),
    )
    logger: logging.Logger = Field(
        default_factory=lambda: _internal_logger,
        description="Logger instance (injectable for tests or custom logging)",
        exclude=True,
    )

    # -- Non-serialised internal state ----------------------------------------
    _site: Optional[mwclient.Site] = PrivateAttr(default=None)
    _content_namespace_ids: Optional[List[int]] = PrivateAttr(default=None)

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
                self.host,
                path=self.path,
                scheme=self.scheme,
            )
        return self._site

    def login(self, username: str, password: str) -> None:
        """Authenticate to the wiki using user credentials.

        Uses ``clientlogin`` (MW 1.27+). For bot passwords, use the bot
        password as the ``password`` argument with the full bot-username
        (e.g. ``User@BotName``).

        Args:
            username: MediaWiki username.
            password: MediaWiki password or bot password.

        Raises:
            mwclient.errors.LoginError: If login fails.
        """
        self.site.clientlogin(username=username, password=password)
        self.logger.info("Logged in as %s", username)

    # -- Internal helpers -----------------------------------------------------

    def _fetch_content_namespace_ids(self) -> List[int]:
        """Return namespace IDs marked as content ($wgContentNamespaces).

        Uses ``action=query&meta=siteinfo&siprop=namespaces`` and filters for
        the ``content`` attribute. Falls back to ``[0]`` on failure.
        """
        try:
            result = self.site.get(
                "query", meta="siteinfo", siprop="namespaces"
            )
        except mwclient.errors.APIError as exc:
            self.logger.warning(
                "Could not fetch siteinfo; defaulting to main namespace (0): %s",
                exc,
            )
            return [0]

        namespaces = result.get("query", {}).get("namespaces", {})
        ids: List[int] = []
        for ns_data in namespaces.values():
            if not isinstance(ns_data, dict):
                continue
            if not ns_data.get("content"):
                continue
            ns_id = ns_data.get("id")
            if ns_id is not None:
                ids.append(int(ns_id))

        if not ids:
            self.logger.warning(
                "No content namespaces in siteinfo; defaulting to main namespace (0)."
            )
            return [0]
        return sorted(ids)

    def _build_page_url(
        self, title: str, url_base: Optional[Tuple[str, str]]
    ) -> Optional[str]:
        """Build canonical page URL from pre-parsed (origin, article_path)."""
        if not url_base:
            return None
        origin, article_path = url_base
        return origin + article_path.replace("$1", title.replace(" ", "_"))

    def _extract_revision_time(self, page: Any, title: str) -> Optional[datetime]:
        """Extract last_modified from page revision timestamp (struct_time)."""
        try:
            ts = page.last_rev_time
            if ts:
                return datetime(*ts[:6], tzinfo=timezone.utc)
            return None
        except Exception as e:
            self.logger.debug(
                "Revision timestamp extraction failed for page %r: %s",
                title,
                e,
                exc_info=True,
            )
            return None

    def _get_all_pages_generator(self) -> Iterator[Dict[str, Any]]:
        """Yield rich dicts for all pages via mwclient's allpages.

        Each yielded dict contains:
            - title (str)
            - url (str or None)
            - last_modified (datetime or None)
        """
        if self.namespaces is None:
            if self._content_namespace_ids is None:
                self._content_namespace_ids = self._fetch_content_namespace_ids()
            ns_list = self._content_namespace_ids
        else:
            ns_list = self.namespaces

        # Parse site base URL once; origin + article_path are constant per site
        url_base: Optional[Tuple[str, str]] = None
        try:
            site_info = self.site.site
            base_url = site_info.get("base", "")
            article_path = site_info.get("articlepath", "/wiki/$1")
            if base_url:
                parsed = urlparse(base_url)
                origin = f"{parsed.scheme}://{parsed.netloc}"
                url_base = (origin, article_path)
        except Exception as e:
            self.logger.debug(
                "URL base from siteinfo failed: %s",
                e,
                exc_info=True,
            )

        for ns in ns_list:
            for page in self.site.allpages(
                namespace=ns,
                generator=True,
                api_chunk_size=self.page_limit,
            ):
                title = page.name
                url = self._build_page_url(title, url_base)
                last_modified = self._extract_revision_time(page, title)
                yield {
                    "title": title,
                    "url": url,
                    "last_modified": last_modified,
                }

    def _get_page_contents(self, page_title: str) -> Optional[str]:
        """Fetch parsed HTML content for a page via mwclient's parse API.

        Returns:
            Raw HTML from the API, or ``None`` on failure.
        """
        try:
            result = self.site.parse(
                page=page_title, prop="text"
            )
        except mwclient.errors.APIError as exc:
            self.logger.warning(
                "Parse API failed for '%s': %s", page_title, exc
            )
            return None

        if not result:
            self.logger.warning("No parse result for page '%s'", page_title)
            return None

        html_content = result.get("text", {}).get("*", "")
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
        except Exception as e:
            _internal_logger.debug(
                "html2text failed, using tag-strip fallback: %s",
                e,
                exc_info=True,
            )
            clean_text = re.sub(r"<[^>]+>", "", html_content)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text

    # -- Resource API (load_resource) -----------------------------------------

    def load_resource(
        self,
        resource_id: str,
        resource_url: str,
        last_modified: Optional[datetime],
    ) -> List[Document]:
        """Load a single page as a list containing one Document.

        Args:
            resource_id: The page title.
            resource_url: Pre-fetched canonical URL for the page.
            last_modified: Pre-fetched last-modified timestamp (or None).

        Returns:
            A one-element list with the page Document, or empty on failure.
        """
        content = self._get_page_contents(resource_id)
        if not content:
            return []

        content = self._html_to_clean_text(content)
        doc = Document(
            text=content,
            id_=f"mediawiki:{resource_id}",
            metadata={
                "title": resource_id,
                "url": resource_url,
                "last_modified": (
                    last_modified.isoformat() if last_modified else None
                ),
            },
        )
        return [doc]

    # -- Resource info (single page) ------------------------------------------

    def get_resource_info(self, resource_id: str) -> Dict[str, Any]:
        """Return metadata for a single page (URL and last_modified).

        Args:
            resource_id: Page title.

        Returns:
            ``{"last_modified": datetime | None, "url": str | None}``.
        """
        try:
            result = self.site.get(
                "query",
                titles=resource_id,
                prop="info|revisions",
                inprop="url",
                rvprop="timestamp",
            )
        except mwclient.errors.APIError as exc:
            self.logger.warning(
                "Query API failed for '%s': %s", resource_id, exc
            )
            return {"last_modified": None, "url": None}

        pages = result.get("query", {}).get("pages", {})
        page_data = next(
            (p for p in pages.values() if p.get("title")), None
        )
        last_modified = None
        url = None
        if page_data and "missing" not in page_data:
            url = page_data.get("canonicalurl")
            revisions = page_data.get("revisions", [])
            if revisions:
                ts_str = revisions[0].get("timestamp")
                if ts_str:
                    try:
                        last_modified = datetime.fromisoformat(
                            ts_str.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass

        return {"last_modified": last_modified, "url": url}

    # -- BasePydanticReader / BaseReader interface -----------------------------

    def lazy_load_data(self, *args: Any, **kwargs: Any) -> Iterator[Document]:
        """Yield one Document per page in the wiki.

        Iterates all pages via mwclient's allpages, then fetches parsed
        content for each page with one parse API call per page (MediaWiki
        has no batch parse API; see class docstring for scale notes).
        """
        for page_record in self._get_all_pages_generator():
            url = page_record.get("url")
            if not url:
                # API-based URL construction failed; use guaranteed fallback
                # so we never drop content due to missing siteinfo.
                safe_title = page_record["title"].replace(" ", "_")
                url = (
                    f"{self.scheme}://{self.host}{self.path}"
                    f"index.php?title={safe_title}"
                )
                self.logger.debug(
                    "Using fallback URL for %s",
                    page_record["title"],
                )
            docs = self.load_resource(
                page_record["title"],
                resource_url=url,
                last_modified=page_record.get("last_modified"),
            )
            yield from docs
