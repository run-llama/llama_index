from typing import Any

from llama_index.readers.oxylabs.base import OxylabsBaseReader
from oxylabs.sources.response import Response


class OxylabsAmazonPricingReader(OxylabsBaseReader):
    """
    Get data about Amazon product offer listings.

    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/amazon/pricing
    """

    top_level_header: str = "Product pricing data"

    def __init__(self, username: str, password: str, **data) -> None:
        super().__init__(username=username, password=password, **data)

    @classmethod
    def class_name(cls) -> str:
        return "OxylabsAmazonPricingReader"

    def get_response(self, payload: dict[str, Any]) -> Response:
        return self.oxylabs_api.amazon.scrape_pricing(**payload)

    async def aget_response(self, payload: dict[str, Any]) -> Response:
        return await self.async_oxylabs_api.amazon.scrape_pricing(**payload)
