import os
import logging
from typing import List
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

    def load_data(self, urls: List[str], text_content: bool = False) -> List[Document]:
        """Load pages using Browserbase Web Reader"""

        pages = self.browserbase.load_urls(urls, text_content)

        documents = []
        for i, page in enumerate(pages):
            documents.append(
                Document(
                    text=page,
                    metadata={
                        "url": urls[i],
                    },
                )
            )

        return documents


if __name__ == "__main__":
    reader = BrowserbaseWebReader()
    logger.info(reader.load_data(urls=["https://example.com"]))
