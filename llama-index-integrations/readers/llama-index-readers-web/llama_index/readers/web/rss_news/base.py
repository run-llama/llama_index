"""RSS feed reader for news - processes each article with NewsArticleReader."""

import logging
from typing import Any, List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.web.news.base import NewsArticleReader

logger = logging.getLogger(__name__)


class RssNewsReader(BaseReader):
    """
    RSS news reader.

    Reads news content from RSS feeds and parses with NewsArticleReader.

    """

    def __init__(self, **reader_kwargs: Any) -> None:
        """
        Initialize with parameters.

        Args:
            html_to_text (bool): Whether to convert HTML to text.
                Requires `html2text` package.

        """
        try:
            import feedparser  # noqa: F401
        except ImportError:
            raise ImportError(
                "`feedparser` package not found, please run `pip install feedparser`"
            )

        try:
            import listparser  # noqa: F401
        except ImportError:
            raise ImportError(
                "`listparser` package not found, please run `pip install listparser`"
            )

        self.reader_kwargs = reader_kwargs

    def load_data(self, urls: List[str] = None, opml: str = None) -> List[Document]:
        """
        Load data from either RSS feeds or OPML.

        Args:
            urls (List[str]): List of RSS URLs to load.
            opml (str): URL to OPML file or string or byte OPML content.

        Returns:
            List[Document]: List of documents.

        """
        if (urls is None) == (
            opml is None
        ):  # This is True if both are None or neither is None
            raise ValueError(
                "Provide either the urls or the opml argument, but not both."
            )

        import feedparser

        if urls and not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")

        documents = []

        if not urls and opml:
            try:
                import listparser
            except ImportError as e:
                raise ImportError(
                    "Package listparser must be installed if the opml arg is used. "
                    "Please install with 'pip install listparser' or use the "
                    "urls arg instead."
                ) from e
            rss = listparser.parse(opml)
            urls = [feed.url for feed in rss.feeds]

        for url in urls:
            try:
                feed = feedparser.parse(url)
                for i, entry in enumerate(feed.entries):
                    article = NewsArticleReader(**self.reader_kwargs).load_data(
                        urls=[entry.link],
                    )[0]
                    article.metadata["feed"] = url

                    documents.append(
                        Document(text=article.text, metadata=article.metadata)
                    )

            except Exception as e:
                logger.error(f"Error fetching or processing {url}, exception: {e}")
                continue

        return documents


if __name__ == "__main__":
    reader = RssNewsReader()
    logger.info(reader.load_data(urls=["https://www.engadget.com/rss.xml"]))

    # Generate keywords and summary for each article
    reader = RssNewsReader(use_nlp=True)
    logger.info(reader.load_data(urls=["https://www.engadget.com/rss.xml"]))
