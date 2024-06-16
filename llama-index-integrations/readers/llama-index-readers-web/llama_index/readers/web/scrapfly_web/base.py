"""Scrapfly Web Reader."""
import logging
from typing import List, Optional, Literal

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

logger = logging.getLogger(__file__)


class ScrapflyReader(BasePydanticReader):
    """Turn a url to llm accessible markdown with `Scrapfly.io`.

    Args:
    api_key: The Scrapfly API key.
    scrape_config: The Scrapfly ScrapeConfig object.
    ignore_scrape_failures: Whether to continue on failures.
    urls: List of urls to scrape.
    scrape_format: Scrape result format (markdown or text)
    For further details, visit: https://scrapfly.io/docs/sdk/python
    """

    api_key: str
    ignore_scrape_failures: bool = True
    scrapfly: Optional["ScrapflyClient"] = None  # Declare the scrapfly attribute

    def __init__(self, api_key: str, ignore_scrape_failures: bool = True) -> None:
        """Initialize client."""
        super().__init__(api_key=api_key, ignore_scrape_failures=ignore_scrape_failures)
        try:
            from scrapfly import ScrapflyClient
        except ImportError:
            raise ImportError(
                "`scrapfly` package not found, please run `pip install scrapfly-sdk`"
            )
        self.scrapfly = ScrapflyClient(key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "Scrapfly_reader"

    def load_data(
        self,
        urls: List[str],
        scrape_format: Literal["markdown", "text"] = "markdown",
        scrape_config: Optional[dict] = None,
    ) -> List[Document]:
        """Load data from the urls.

        Args:
            urls: List[str]): List of URLs to scrape.
            scrape_config: Optional[dict]: Dictionary of ScrapFly scrape config object.

        Returns:
            List[Document]: List of documents.

        Raises:
            ValueError: If URLs aren't provided.
        """
        from scrapfly import ScrapeApiResponse, ScrapeConfig

        if urls is None:
            raise ValueError("URLs must be provided.")
        scrape_config = scrape_config if scrape_config is not None else {}

        documents = []
        for url in urls:
            try:
                response: ScrapeApiResponse = self.scrapfly.scrape(
                    ScrapeConfig(url, format=scrape_format, **scrape_config)
                )
                documents.append(
                    Document(
                        text=response.scrape_result["content"], extra_info={"url": url}
                    )
                )
            except Exception as e:
                if self.ignore_scrape_failures:
                    logger.error(f"Error fetching data from {url}, exception: {e}")
                else:
                    raise e  # noqa: TRY201

        return documents
