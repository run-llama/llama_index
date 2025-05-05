from typing import Any

from llama_index.readers.oxylabs.base import OxylabsBaseReader
from oxylabs.sources.response import Response


class OxylabsAmazonBestsellersReader(OxylabsBaseReader):
    """
    Get data from Amazon Best Sellers pages.

    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/amazon/best-sellers
    """

    top_level_header: str = "Bestsellers"

    def __init__(self, username: str, password: str, **data) -> None:
        super().__init__(username=username, password=password, **data)

    @classmethod
    def class_name(cls) -> str:
        return "OxylabsAmazonBestsellersReader"

    def get_response(self, payload: dict[str, Any]) -> Response:
        return self.oxylabs_api.amazon.scrape_bestsellers(**payload)

    async def aget_response(self, payload: dict[str, Any]) -> Response:
        return await self.async_oxylabs_api.amazon.scrape_bestsellers(**payload)
