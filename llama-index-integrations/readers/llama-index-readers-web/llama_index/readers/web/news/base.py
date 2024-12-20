"""News article reader using Newspaper."""
import logging
from importlib.util import find_spec
from typing import Any, Generator, List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class NewsArticleReader(BaseReader):
    """Simple news article reader.

    Reads news articles from the web and parses them using the `newspaper` library.

    Args:
        text_mode (bool): Whether to load a text version or HTML version of the content (default=True).
        use_nlp (bool): Whether to use NLP to extract additional summary and keywords (default=True).
        newspaper_kwargs: Additional keyword arguments to pass to newspaper.Article. See
            https://newspaper.readthedocs.io/en/latest/user_guide/quickstart.html#article
    """

    def __init__(
        self, text_mode: bool = True, use_nlp: bool = True, **newspaper_kwargs: Any
    ) -> None:
        """Initialize with parameters."""
        if find_spec("newspaper") is None:
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
        if not isinstance(urls, list) and not isinstance(urls, Generator):
            raise ValueError("urls must be a list or generator.")
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


if __name__ == "__main__":
    reader = NewsArticleReader()
    article = reader.load_data(["https://www.bbc.com/news/world-us-canada-56797998"])
    print(article)
