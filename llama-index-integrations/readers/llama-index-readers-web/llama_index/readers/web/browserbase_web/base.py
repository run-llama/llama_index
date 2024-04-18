import os
import logging
from typing import Iterator, List
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


logger = logging.getLogger(__name__)


class BrowserbaseWebReader(BaseReader):
    """Browserbase Web Reader"""

    def __init__(
        self,
        api_key: str = os.environ["BROWSERBASE_KEY"],
    ) -> None:
        try:
            from browserbase import Browserbase
        except ImportError:
            raise ImportError(
                "`browserbase` package not found, please run `pip install browserbase`"
            )

        self.browserbase = Browserbase(api_key=api_key)

    def lazy_load_data(self, urls: List[str], text_content: bool = False) -> Iterator[Document]:
        """Load pages using Browserbase Web Reader"""
        pages = self.browserbase.load_urls(urls, text_content)

        for i, page in enumerate(pages):
            yield Document(
                text=page,
                metadata={
                    "url": urls[i],
                },
            )


if __name__ == "__main__":
    reader = BrowserbaseWebReader()
    logger.info(reader.load_data(urls=["https://example.com"]))
