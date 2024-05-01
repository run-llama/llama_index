from typing import Optional, List, Dict, Any
from pydantic import field_validator

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class SpiderWebReader(BasePydanticReader):
    """
    Scrapes a URL for data and returns llm-ready data with `Spider.cloud`.

    Args:
        api_key (str): The Spider API key, get one at https://spider.cloud
        mode (str): "Scrape" the url (default) or "crawl" the url following all subpages.
        params (Dict[str, Any]): Additional parameters to pass to the Spider API.
    """

    api_key: str
    mode: Optional[str] = "scrape"
    params: Optional[Dict[str, Any]] = None

    @field_validator("mode")
    def validate_mode(cls, v):
        if v not in ["scrape", "crawl"]:
            raise ValueError("Mode must be 'scrape' or 'crawl'")
        return v

    def __init__(
        self,
        api_key: str,
        mode: str = "scrape",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            api_key=api_key, mode=mode, params=params if params is not None else {}
        )
        try:
            from spider_async import (
                Spider,
            )  # Assume an asynchronous version of the Spider
        except ImportError:
            raise ImportError(
                "`spider-client` package not found, please run `pip install spider-client`"
            )
        self.spider = Spider(api_key)

    async def load_data(self, url: str) -> List[Document]:
        documents = []
        action = (
            self.spider.scrape_url if self.mode == "scrape" else self.spider.crawl_url
        )
        spider_docs = await action(
            url=url, params=self.params
        )  # Assuming this is an async call

        if isinstance(spider_docs, list):  # Expected in 'crawl' mode
            for doc in spider_docs:
                documents.append(
                    Document(
                        page_content=doc.get("markdown", ""),
                        metadata=doc.get("metadata", {}),
                    )
                )
        else:  # Expected in 'scrape' mode
            documents.append(
                Document(
                    page_content=spider_docs.get("markdown", ""),
                    metadata=spider_docs.get("metadata", {}),
                )
            )

        return documents
