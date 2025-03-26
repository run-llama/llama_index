"""Rss reader."""

from typing import List, Any, Union
import logging

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class RssReader(BasePydanticReader):
    """RSS reader.

    Reads content from an RSS feed.

    """

    is_remote: bool = True
    html_to_text: bool = False
    user_agent: Union[str, None] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # https://pythonhosted.org/feedparser/http-useragent.html
        self.user_agent = kwargs.get("user_agent", None)

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

        if self.user_agent:
            feedparser.USER_AGENT = self.user_agent

        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")

        documents = []

        for url in urls:
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                doc_id = getattr(entry, "id", None) or getattr(entry, "link", None)
                data = entry.get("content", [{}])[0].get(
                    "value", entry.get("description", entry.get("summary", ""))
                )

                if self.html_to_text:
                    import html2text

                    data = html2text.html2text(data)

                extra_info = {
                    "title": getattr(entry, "title", None),
                    "link": getattr(entry, "link", None),
                    "date": getattr(entry, "published", None),
                }

                if doc_id:
                    documents.append(
                        Document(text=data, id_=doc_id, extra_info=extra_info)
                    )
                else:
                    documents.append(Document(text=data, extra_info=extra_info))

        return documents


if __name__ == "__main__":
    default_reader = RssReader()
    print(
        default_reader.load_data(urls=["https://rsshub.app/hackernews/newest"])
    )  # 0 blocked by cloudflare
    reader = RssReader(user_agent="MyApp/1.0 +http://example.com/")
    print(reader.load_data(urls=["https://rsshub.app/hackernews/newest"]))
