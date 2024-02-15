"""Rss reader."""

from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


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
                documents.append(Document(text=data, extra_info=extra_info))

        return documents
