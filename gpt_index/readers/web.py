"""Web scraper."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class SimpleWebPageReader(BaseReader):
    """Simple web page reader.

    Reads pages from the web.

    Args:
        html_to_text (bool): Whether to convert HTML to text.
            Requires `html2text` package.

    """

    def __init__(self, html_to_text: bool = False) -> None:
        """Initialize with parameters."""
        try:
            import html2text  # noqa: F401
        except ImportError:
            raise ValueError(
                "`html2text` package not found, please run `pip install html2text`"
            )
        self._html_to_text = html_to_text

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
            if self._html_to_text:
                import html2text

                response = html2text.html2text(response)

            documents.append(Document(response))

        return documents


class TrafilaturaWebReader(BaseReader):
    """Trafilatura web page reader.

    Reads pages from the web.
    Requires the `trafilatura` package.

    """

    def __init__(self, error_on_missing: bool = False) -> None:
        """Initialize with parameters.

        Args:
            error_on_missing (bool): Throw an error when data cannot be parsed
        """
        self.error_on_missing = error_on_missing
        try:
            import trafilatura  # noqa: F401
        except ImportError:
            raise ValueError(
                "`trafilatura` package not found, please run `pip install trafilatura`"
            )

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
            documents.append(Document(response))

        return documents


def _substack_reader(soup: Any) -> Tuple[str, Dict[str, Any]]:
    """Extract text from Substack blog post."""
    extra_info = {
        "Title of this Substack post": soup.select_one("h1.post-title").getText(),
        "Subtitle": soup.select_one("h3.subtitle").getText(),
        "Author": soup.select_one("span.byline-names").getText(),
    }
    text = soup.select_one("div.available-content").getText()
    return text, extra_info


DEFAULT_WEBSITE_EXTRACTOR: Dict[str, Callable[[Any], Tuple[str, Dict[str, Any]]]] = {
    "substack.com": _substack_reader,
}


class BeautifulSoupWebReader(BaseReader):
    """BeautifulSoup web page reader.

    Reads pages from the web.
    Requires the `bs4` and `urllib` packages.

    Args:
        file_extractor (Optional[Dict[str, Callable]]): A mapping of website
            hostname (e.g. google.com) to a function that specifies how to
            extract text from the BeautifulSoup obj. See DEFAULT_WEBSITE_EXTRACTOR.
    """

    def __init__(
        self,
        website_extractor: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """Initialize with parameters."""
        try:
            from urllib.parse import urlparse  # noqa: F401

            import requests  # noqa: F401
            from bs4 import BeautifulSoup  # noqa: F401
        except ImportError:
            raise ValueError(
                "`bs4`, `requests`, and `urllib` must be installed to scrape websites."
                "Please run `pip install bs4 requests urllib`."
            )

        self.website_extractor = website_extractor or DEFAULT_WEBSITE_EXTRACTOR

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
            extra_info = {"URL": url}
            if hostname in self.website_extractor:
                data, metadata = self.website_extractor[hostname](soup)
                extra_info.update(metadata)
            else:
                data = soup.getText()

            documents.append(Document(data, extra_info=extra_info))

        return documents


class RssReader(BaseReader):
    """RSS reader.

    Reads content from an RSS feed.

    """

    def __init__(self, html_to_text: bool = False) -> None:
        """Initialize with parameters.

        Args:
            html_to_text (bool): Whether to convert HTML to text.
                Requires `html2text` package.

        """
        try:
            import feedparser  # noqa: F401
        except ImportError:
            raise ValueError(
                "`feedparser` package not found, please run `pip install feedparser`"
            )

        if html_to_text:
            try:
                import html2text  # noqa: F401
            except ImportError:
                raise ValueError(
                    "`html2text` package not found, please run `pip install html2text`"
                )
        self._html_to_text = html_to_text

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
                if "content" in entry:
                    data = entry.content[0].value
                else:
                    data = entry.description or entry.summary

                if self._html_to_text:
                    import html2text

                    data = html2text.html2text(data)

                extra_info = {"title": entry.title, "link": entry.link}
                documents.append(Document(data, extra_info=extra_info))

        return documents


if __name__ == "__main__":
    reader = SimpleWebPageReader()
    logging.info(reader.load_data(["http://www.google.com"]))
