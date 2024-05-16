import logging
from typing import Optional, Iterator, Sequence
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


logger = logging.getLogger(__name__)


class BrowserbaseWebReader(BaseReader):
    """BrowserbaseWebReader.

    Load pre-rendered web pages using a headless browser hosted on Browserbase.
    Depends on `browserbase` package.
    Get your API key from https://browserbase.com
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        try:
            from browserbase import Browserbase
        except ImportError:
            raise ImportError(
                "`browserbase` package not found, please run `pip install browserbase`"
            )

        self.browserbase = Browserbase(api_key, project_id)

    def lazy_load_data(
        self,
        urls: Sequence[str],
        text_content: bool = False,
        session_id: Optional[str] = None,
        proxy: Optional[bool] = None,
    ) -> Iterator[Document]:
        """Load pages from URLs."""
        pages = self.browserbase.load_urls(urls, text_content, session_id, proxy)

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
