"""Rss reader."""

from typing import List

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class RssReader(BasePydanticReader):
    """RSS reader.

    Reads content from an RSS feed.

    """

    is_remote: bool = True
    html_to_text: bool = False

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

                extra_info = {"title": entry.title, "link": entry.link}
                documents.append(Document(text=data, id_=doc_id, extra_info=extra_info))

        return documents
