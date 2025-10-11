import os
import logging
from typing import (
    Any,
    AsyncIterable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


logger = logging.getLogger(__name__)


class HyperbrowserWebReader(BaseReader):
    """
    Hyperbrowser Web Reader.

    Scrape or crawl web pages with optional parameters for configuring content extraction.
    Requires the `hyperbrowser` package.
    Get your API Key from https://app.hyperbrowser.ai/

    Args:
        api_key: The Hyperbrowser API key, can be set as an environment variable `HYPERBROWSER_API_KEY` or passed directly

    """

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("HYPERBROWSER_API_KEY")
        if not api_key:
            raise ValueError(
                "`api_key` is required, please set the `HYPERBROWSER_API_KEY` environment variable or pass it directly"
            )

        try:
            from hyperbrowser import Hyperbrowser, AsyncHyperbrowser
        except ImportError:
            raise ImportError(
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            )

        self.hyperbrowser = Hyperbrowser(api_key=api_key)
        self.async_hyperbrowser = AsyncHyperbrowser(api_key=api_key)

    def _prepare_params(self, params: Dict) -> Dict:
        """Prepare session and scrape options parameters."""
        try:
            from hyperbrowser.models.session import CreateSessionParams
            from hyperbrowser.models.scrape import ScrapeOptions
        except ImportError:
            raise ImportError(
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            )

        if "scrape_options" in params:
            if "formats" in params["scrape_options"]:
                formats = params["scrape_options"]["formats"]
                if not all(fmt in ["markdown", "html"] for fmt in formats):
                    raise ValueError("formats can only contain 'markdown' or 'html'")

        if "session_options" in params:
            params["session_options"] = CreateSessionParams(**params["session_options"])
        if "scrape_options" in params:
            params["scrape_options"] = ScrapeOptions(**params["scrape_options"])
        return params

    def _create_document(self, content: str, metadata: dict) -> Document:
        """Create a Document with text and metadata."""
        return Document(text=content, metadata=metadata)

    def _extract_content_metadata(self, data: Union[Any, None]):
        """Extract content and metadata from response data."""
        content = ""
        metadata = {}
        if data:
            content = data.markdown or data.html or ""
            if data.metadata:
                metadata = data.metadata
        return content, metadata

    def lazy_load_data(
        self,
        urls: List[str],
        operation: Literal["scrape", "crawl"] = "scrape",
        params: Optional[Dict] = {},
    ) -> Iterable[Document]:
        """
        Lazy load documents.

        Args:
            urls: List of URLs to scrape or crawl
            operation: Operation to perform. Can be "scrape" or "crawl"
            params: Optional params for scrape or crawl. For more information on the supported params, visit https://docs.hyperbrowser.ai/reference/sdks/python/scrape#start-scrape-job-and-wait or https://docs.hyperbrowser.ai/reference/sdks/python/crawl#start-crawl-job-and-wait

        """
        try:
            from hyperbrowser.models.scrape import StartScrapeJobParams
            from hyperbrowser.models.crawl import StartCrawlJobParams
        except ImportError:
            raise ImportError(
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            )

        if operation == "crawl" and len(urls) > 1:
            raise ValueError("`crawl` operation can only accept a single URL")
        params = self._prepare_params(params)

        if operation == "scrape":
            for url in urls:
                scrape_params = StartScrapeJobParams(url=url, **params)
                try:
                    scrape_resp = self.hyperbrowser.scrape.start_and_wait(scrape_params)
                    content, metadata = self._extract_content_metadata(scrape_resp.data)
                    yield self._create_document(content, metadata)
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    yield self._create_document("", {})
        else:
            crawl_params = StartCrawlJobParams(url=urls[0], **params)
            try:
                crawl_resp = self.hyperbrowser.crawl.start_and_wait(crawl_params)
                for page in crawl_resp.data:
                    content = page.markdown or page.html or ""
                    yield self._create_document(content, page.metadata or {})
            except Exception as e:
                logger.error(f"Error crawling {urls[0]}: {e}")
                yield self._create_document("", {})

    async def alazy_load_data(
        self,
        urls: Sequence[str],
        operation: Literal["scrape", "crawl"] = "scrape",
        params: Optional[Dict] = {},
    ) -> AsyncIterable[Document]:
        """
        Async lazy load documents.

        Args:
            urls: List of URLs to scrape or crawl
            operation: Operation to perform. Can be "scrape" or "crawl"
            params: Optional params for scrape or crawl. For more information on the supported params, visit https://docs.hyperbrowser.ai/reference/sdks/python/scrape#start-scrape-job-and-wait or https://docs.hyperbrowser.ai/reference/sdks/python/crawl#start-crawl-job-and-wait

        """
        try:
            from hyperbrowser.models.scrape import StartScrapeJobParams
            from hyperbrowser.models.crawl import StartCrawlJobParams
        except ImportError:
            raise ImportError(
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            )

        if operation == "crawl" and len(urls) > 1:
            raise ValueError("`crawl` operation can only accept a single URL")
        params = self._prepare_params(params)

        if operation == "scrape":
            for url in urls:
                scrape_params = StartScrapeJobParams(url=url, **params)
                try:
                    scrape_resp = await self.async_hyperbrowser.scrape.start_and_wait(
                        scrape_params
                    )
                    content, metadata = self._extract_content_metadata(scrape_resp.data)
                    yield self._create_document(content, metadata)
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    yield self._create_document("", {})
        else:
            crawl_params = StartCrawlJobParams(url=urls[0], **params)
            try:
                crawl_resp = await self.async_hyperbrowser.crawl.start_and_wait(
                    crawl_params
                )
                for page in crawl_resp.data:
                    content = page.markdown or page.html or ""
                    yield self._create_document(content, page.metadata or {})
            except Exception as e:
                logger.error(f"Error crawling {urls[0]}: {e}")
                yield self._create_document("", {})
