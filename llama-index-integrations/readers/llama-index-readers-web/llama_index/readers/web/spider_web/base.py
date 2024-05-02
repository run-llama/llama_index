from typing import List, Optional, Literal

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document


class SpiderWebReader(BasePydanticReader):
    """
    Scrapes a URL for data and returns llm-ready data with `Spider.cloud`.

    Must have the Python package `spider-client` installed and a Spider API key.
    See https://spider.cloud for more.

    Args:
        api_key (str): The Spider API key, get one at https://spider.cloud
        mode (Mode): "Scrape" the url (default) or "crawl" the url following all subpages.
        params (dict): Additional parameters to pass to the Spider API.
    """

    class Config:
        use_enum_values = True
        extra = "allow"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        mode: Literal["scrape", "crawl"] = "scrape",
        params: Optional[dict] = None,
    ) -> None:
        super().__init__(api_key=api_key, mode=mode, params=params)

        if params is None:
            params = {"return_format": "markdown", "metadata": True}
        try:
            from spider import Spider
        except ImportError:
            raise ImportError(
                "`spider-client` package not found, please run `pip install spider-client`"
            )
        self.spider = Spider(api_key=api_key)
        self.mode = mode
        self.params = params

    def load_data(self, url: str) -> List[Document]:
        if self.mode != "scrape" and self.mode != "crawl":
            raise ValueError(
                "Unknown mode in `mode` parameter, `scrape` or `crawl` is the allowed modes"
            )
        action = (
            self.spider.scrape_url if self.mode == "scrape" else self.spider.crawl_url
        )
        spider_docs = action(url=url, params=self.params)

        if not spider_docs:
            return [Document(page_content="", metadata={})]

        documents = []
        if isinstance(spider_docs, list):
            for doc in spider_docs:
                text = doc.get("content", "")
                if text is not None:
                    documents.append(
                        Document(
                            text=text,
                            metadata=doc.get("metadata", {}),
                        )
                    )

        return documents
