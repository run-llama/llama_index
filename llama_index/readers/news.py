"""News article reader using Newspaper."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from llama_index.readers.base import BaseReader
from llama_index.schema import Document

logger = logging.getLogger(__name__)


class NewsArticleReader(BaseReader):
    """Simple news article reader.

    Reads news articles from the web and parses them using the `newspaper` library.

    Args:
        text_mode (bool): Whether to load a text version or HTML version of the content (default=True).
        use_nlp (bool): Whether to use NLP to extract additional summary and keywords (default=False).
        newspaper_kwargs: Additional keyword arguments to pass to newspaper.Article. See
            https://newspaper.readthedocs.io/en/latest/user_guide/quickstart.html#article
    """

    def __init__(self,
                 text_mode: bool = True,
                 use_nlp: bool = True,
                 **newspaper_kwargs: Any
                 ) -> None:
        """Initialize with parameters."""
        try:
            import newspaper
        except ImportError:
            raise ImportError(
                "`newspaper` package not found, please run `pip install newspaper3k`"
            )
        self.load_text = text_mode
        self.use_nlp = use_nlp
        self.newspaper_kwargs = newspaper_kwargs

    def load_data(self, urls: List[str]) -> List[Document]:
        """Load data from the list of news article urls.

        Args:
            urls (List[str]): List of URLs to load news articles.

        Returns:
            List[Document]: List of documents.

        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        documents = []
        for url in urls:
            from newspaper import Article
            try:
                article = Article(url, **self.newspaper_kwargs)
                article.download()
                article.parse()

                if self.use_nlp:
                    article.nlp()

            except Exception as e:
                logger.error(f"Error fetching or processing {url}, exception: {e}")
                continue

            metadata = {
                "title": getattr(article, "title", ""),
                "link": getattr(article, "url", getattr(article, "canonical_link", "")),
                "authors": getattr(article, "authors", []),
                "language": getattr(article, "meta_lang", ""),
                "description": getattr(article, "meta_description", ""),
                "publish_date": getattr(article, "publish_date", ""),
            }

            if self.load_text:
                content = article.text
            else:
                content = article.html

            if self.use_nlp:
                metadata["keywords"] = getattr(article, "keywords", [])
                metadata["summary"] = getattr(article, "summary", "")

            documents.append(Document(text=content, metadata=metadata))

        return documents


class RssNewsReader(BaseReader):
    """RSS news reader.

    Reads news content from RSS feeds and parses with NewsArticleReader.

    """

    def __init__(self, **reader_kwargs: Any) -> None:
        """Initialize with parameters.

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

    def load_data(self,
                  urls: List[str] = None,
                  opml: str = None
                  ) -> List[Document]:
        """Load data from either RSS feeds or OPML.

        Args:
            urls (List[str]): List of RSS URLs to load.
            opml (str): URL to OPML file or string or byte OPML content.

        Returns:
            List[Document]: List of documents.

        """
        if (urls is None) == (opml is None):  # This is True if both are None or neither is None
            raise ValueError('Provide either the urls or the opml argument, but not both.')

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
                    article.metadata['feed'] = url

                    documents.append(Document(text=article.text, metadata=article.metadata))

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
