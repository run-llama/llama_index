"""Firecrawl Web Reader."""
from typing import List, Optional, Dict, Callable
from pydantic import Field

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class FireCrawlWebReader(BasePydanticReader):
    """turn a url to llm accessible markdown with `Firecrawl.dev`.

    Args:
    api_key: The Firecrawl API key.
    url: The url to be crawled (or)
    mode: The mode to run the loader in. Default is "crawl".
    Options include "scrape" (single url) and
    "crawl" (all accessible sub pages).
    params: The parameters to pass to the Firecrawl API.
    Examples include crawlerOptions.
    For more details, visit: https://github.com/mendableai/firecrawl-py

    """

    firecrawl: Optional[object] = Field(None)
    api_key: str
    mode: Optional[str]
    params: Optional[dict]

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        mode: Optional[str] = "crawl",
        params: Optional[dict] = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(api_key=api_key, mode=mode, params=params)
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )
        self.firecrawl = FirecrawlApp(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "Firecrawl_reader"

    def load_data(self, url: str) -> List[Document]:
        """Load data from the input directory.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        documents = []

        if self.mode == "scrape":
            firecrawl_docs = self.firecrawl.scrape_url(url, params=self.params)
            documents.append(
                Document(
                    page_content=firecrawl_docs.get("markdown", ""),
                    metadata=firecrawl_docs.get("metadata", {}),
                )
            )
        else:
            firecrawl_docs = self.firecrawl.crawl_url(url, params=self.params)
            for doc in firecrawl_docs:
                documents.append(
                    Document(
                        page_content=doc.get("markdown", ""),
                        metadata=doc.get("metadata", {}),
                    )
                )

        return documents
