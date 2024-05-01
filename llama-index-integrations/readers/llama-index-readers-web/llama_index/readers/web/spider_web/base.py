from enum import Enum
from pydantic import Extra
from typing import Optional, List, Literal

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class Mode(str, Enum):
    SCRAPE = "scrape"
    CRAWL = "crawl"


class SpiderWebReader(BasePydanticReader):
    """
    Scrapes a URL for data and returns llm-ready data with `Spider.cloud`.

    Must have the Python package `spider-client` installed and a Spider API key.
    See https://spider.cloud for more.

    Args:
        api_key (str): The Spider API key, get one at https://spider.cloud
        mode (str): "Scrape" the url (default) or "crawl" the url following all subpages.
        params (Dict[str, Any]): Additional parameters to pass to the Spider API.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        mode: Literal["scrape", "crawl"] = "scrape",
        params: Optional[dict] = {"return_format": "markdown"},
    ) -> None:
        super().__init__(api_key=api_key, mode=mode, params=params)
        try:
            from spider import Spider  # Ensure this import matches the actual package
        except ImportError:
            raise ImportError(
                "`spider-client` package not found, please run `pip install spider-client`"
            )
        self.spider = Spider(api_key=api_key)

        if mode not in ("scrape", "crawl"):
            raise ValueError(
                f"Unrecognized mode '{mode}'. Expected one of 'scrape', 'crawl'."
            )
        # If `params` is `None`, initialize it as an empty dictionary
        if params is None:
            params = {}

        # Add a default value for 'metadata' if it's not already present
        if "metadata" not in params:
            params["metadata"] = True

        # Use the environment variable if the API key isn't provided
        api_key = api_key
        self.spider = Spider(api_key=api_key)
        self.mode = mode
        self.params = params

    class Config:
        use_enum_values = True
        extra = Extra.allow

    def load_data(self, url: str) -> List[Document]:
        if self.mode != "scrape" and self.mode != "crawl":
            raise ValueError(
                "Unknown mode in `mode` parameter, `scrape` or `crawl` is the allowed modes"
            )

        documents = []
        action = (
            self.spider.scrape_url if self.mode == "scrape" else self.spider.crawl_url
        )
        spider_docs = action(url=url, params=self.params)

        if not spider_docs:
            return Document(
                page_content="",
                metadata="",
            )

        if isinstance(spider_docs, list):  # Expected in 'crawl' mode
            for doc in spider_docs:
                text = doc.get("content", "")
                if text is not None:
                    documents.append(
                        Document(
                            text=doc.get("content", ""),
                            metadata=doc.get("metadata", {}),
                        )
                    )

        return documents
