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
    api_url: url to be passed to FirecrawlApp for local deployment
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
    api_url: Optional[str]
    mode: Optional[str]
    params: Optional[dict]

    _metadata_fn: Optional[Callable[[str], Dict]] = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        mode: Optional[str] = "crawl",
        params: Optional[dict] = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(api_key=api_key, api_url=api_url, mode=mode, params=params)
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )
        if api_url:
            self.firecrawl = FirecrawlApp(api_key=api_key, api_url=api_url)
        else:
            self.firecrawl = FirecrawlApp(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "Firecrawl_reader"

    def load_data(
        self, url: Optional[str] = None, query: Optional[str] = None
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            url (Optional[str]): URL to scrape or crawl.
            query (Optional[str]): Query to search for.

        Returns:
            List[Document]: List of documents.

        Raises:
            ValueError: If neither or both url and query are provided.
        """
        if url is None and query is None:
            raise ValueError("Either url or query must be provided.")
        if url is not None and query is not None:
            raise ValueError("Only one of url or query must be provided.")

        documents = []

        if self.mode == "scrape":
            firecrawl_docs = self.firecrawl.scrape_url(url, params=self.params)
            documents.append(
                Document(
                    text=firecrawl_docs.get("markdown", ""),
                    metadata=firecrawl_docs.get("metadata", {}),
                )
            )
        elif self.mode == "crawl":
            firecrawl_docs = self.firecrawl.crawl_url(url, params=self.params)
            firecrawl_docs = firecrawl_docs.get("data", [])
            for doc in firecrawl_docs:
                documents.append(
                    Document(
                        text=doc.get("markdown", ""),
                        metadata=doc.get("metadata", {}),
                    )
                )
        elif self.mode == "search":
            firecrawl_docs = self.firecrawl.search(query, params=self.params)
            for doc in firecrawl_docs:
                documents.append(
                    Document(
                        text=doc.get("markdown", ""),
                        metadata=doc.get("metadata", {}),
                    )
                )
        else:
            raise ValueError(
                "Invalid mode. Please choose 'scrape', 'crawl' or 'search'."
            )

        return documents
