"""Web scraper."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.readers.base import BasePydanticReader
from llama_index.schema import Document

logger = logging.getLogger(__name__)


class SimpleWebPageReader(BasePydanticReader):
    """Simple web page reader.

    Reads pages from the web.

    Args:
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.
        metadata_fn (Optional[Callable[[str], Dict]]): A function that takes in
            a URL and returns a dictionary of metadata.
            Default is None.
    """

    is_remote: bool = True
    html_to_text: bool

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    def __init__(
        self,
        html_to_text: bool = False,
        metadata_fn: Optional[Callable[[str], Dict]] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            import html2text  # noqa
        except ImportError:
            raise ImportError(
                "`html2text` package not found, please run `pip install html2text`"
            )
        self._metadata_fn = metadata_fn
        super().__init__(html_to_text=html_to_text)

    @classmethod
    def class_name(cls) -> str:
        return "SimpleWebPageReader"

    def load_data(self, urls: List[str]) -> List[Document]:
        """Load data from the input directory.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        documents = []
        for url in urls:
            response = requests.get(url, headers=None).text
            if self.html_to_text:
                import html2text

                response = html2text.html2text(response)

            metadata: Optional[Dict] = None
            if self._metadata_fn is not None:
                metadata = self._metadata_fn(url)

            documents.append(Document(text=response, id_=url, metadata=metadata or {}))
            documents.append(Document(id_=url, text=response, metadata=metadata or {}))

        return documents


class TrafilaturaWebReader(BasePydanticReader):
    """Trafilatura web page reader.

    Reads pages from the web.
    Requires the `trafilatura` package.

    """

    is_remote: bool = True
    error_on_missing: bool

    def __init__(self, error_on_missing: bool = False) -> None:
        """Initialize with parameters.

        Args:
            error_on_missing (bool): Throw an error when data cannot be parsed
        """
        try:
            import trafilatura  # noqa
        except ImportError:
            raise ImportError(
                "`trafilatura` package not found, please run `pip install trafilatura`"
            )
        super().__init__(error_on_missing=error_on_missing)

    @classmethod
    def class_name(cls) -> str:
        return "TrafilaturaWebReader"

    def load_data(self, urls: List[str]) -> List[Document]:
        """Load data from the urls.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        import trafilatura

        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        documents = []
        for url in urls:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                if self.error_on_missing:
                    raise ValueError(f"Trafilatura fails to get string from url: {url}")
                continue
            response = trafilatura.extract(downloaded)
            if not response:
                if self.error_on_missing:
                    raise ValueError(f"Trafilatura fails to parse page: {url}")
                continue
            documents.append(Document(id_=url, text=response))

        return documents


def _substack_reader(soup: Any) -> Tuple[str, Dict[str, Any]]:
    """Extract text from Substack blog post."""
    metadata = {
        "Title of this Substack post": soup.select_one("h1.post-title").getText(),
        "Subtitle": soup.select_one("h3.subtitle").getText(),
        "Author": soup.select_one("span.byline-names").getText(),
    }
    text = soup.select_one("div.available-content").getText()
    return text, metadata


DEFAULT_WEBSITE_EXTRACTOR: Dict[str, Callable[[Any], Tuple[str, Dict[str, Any]]]] = {
    "substack.com": _substack_reader,
}


class BeautifulSoupWebReader(BasePydanticReader):
    """BeautifulSoup web page reader.

    Reads pages from the web.
    Requires the `bs4` and `urllib` packages.

    Args:
        website_extractor (Optional[Dict[str, Callable]]): A mapping of website
            hostname (e.g. google.com) to a function that specifies how to
            extract text from the BeautifulSoup obj. See DEFAULT_WEBSITE_EXTRACTOR.
    """

    is_remote: bool = True
    _website_extractor: Dict[str, Callable] = PrivateAttr()

    def __init__(
        self,
        website_extractor: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            from urllib.parse import urlparse  # noqa

            import requests  # noqa
            from bs4 import BeautifulSoup  # noqa
        except ImportError:
            raise ImportError(
                "`bs4`, `requests`, and `urllib` must be installed to scrape websites."
                "Please run `pip install bs4 requests urllib`."
            )

        self._website_extractor = website_extractor or DEFAULT_WEBSITE_EXTRACTOR
        super().__init__()

    @classmethod
    def class_name(cls) -> str:
        return "BeautifulSoupWebReader"

    def load_data(
        self, urls: List[str], custom_hostname: Optional[str] = None
    ) -> List[Document]:
        """Load data from the urls.

        Args:
            urls (List[str]): List of URLs to scrape.
            custom_hostname (Optional[str]): Force a certain hostname in the case
                a website is displayed under custom URLs (e.g. Substack blogs)

        Returns:
            List[Document]: List of documents.

        """
        from urllib.parse import urlparse

        import requests
        from bs4 import BeautifulSoup

        documents = []
        for url in urls:
            try:
                page = requests.get(url)
            except Exception:
                raise ValueError(f"One of the inputs is not a valid url: {url}")

            hostname = custom_hostname or urlparse(url).hostname or ""

            soup = BeautifulSoup(page.content, "html.parser")

            data = ""
            metadata = {"URL": url}
            if hostname in self._website_extractor:
                data, metadata = self._website_extractor[hostname](soup)
                metadata.update(metadata)
            else:
                data = soup.getText()

            documents.append(Document(id_=url, text=data, metadata=metadata))

        return documents


class RssReader(BasePydanticReader):
    """RSS reader.

    Reads content from an RSS feed.

    """

    is_remote: bool = True
    html_to_text: bool

    def __init__(self, html_to_text: bool = False) -> None:
        """Initialize with parameters.

        Args:
            html_to_text (bool): Whether to convert HTML to text.
                Requires `html2text` package.

        """
        try:
            import feedparser  # noqa
        except ImportError:
            raise ImportError(
                "`feedparser` package not found, please run `pip install feedparser`"
            )

        if html_to_text:
            try:
                import html2text  # noqa
            except ImportError:
                raise ImportError(
                    "`html2text` package not found, please run `pip install html2text`"
                )
        super().__init__(html_to_text=html_to_text)

    @classmethod
    def class_name(cls) -> str:
        return "RssReader"

    def load_data(self, urls: List[str]) -> List[Document]:
        """Load data from RSS feeds.

        Args:
            urls (List[str]): List of RSS URLs to load.

        Returns:
            List[Document]: List of documents.

        """
        import feedparser

        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")

        documents = []

        for url in urls:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                doc_id = entry.id or entry.link
                if "content" in entry:
                    data = entry.content[0].value
                else:
                    data = entry.description or entry.summary

                if self.html_to_text:
                    import html2text

                    data = html2text.html2text(data)

                metadata = {"title": entry.title, "link": entry.link}
                documents.append(Document(id_=doc_id, text=data, metadata=metadata))

        return documents


if __name__ == "__main__":
    reader = SimpleWebPageReader()
    logger.info(reader.load_data(["http://www.google.com"]))
