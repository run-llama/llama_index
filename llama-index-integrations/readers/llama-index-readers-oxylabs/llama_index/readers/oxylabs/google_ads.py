from typing import Any

from oxylabs.sources.response import Response

from llama_index.readers.oxylabs.google_base import OxylabsGoogleBaseReader


class OxylabsGoogleAdsReader(OxylabsGoogleBaseReader):
    """
    Get Google Search results data with paid ads.

    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/google/ads
    """

    def __init__(self, username: str, password: str, **data) -> None:
        super().__init__(username=username, password=password, **data)

    @classmethod
    def class_name(cls) -> str:
        return "OxylabsGoogleAdsReader"

    def get_response(self, payload: dict[str, Any]) -> Response:
        return self.oxylabs_api.google.scrape_ads(**payload)

    async def aget_response(self, payload: dict[str, Any]) -> Response:
        return await self.async_oxylabs_api.google.scrape_ads(**payload)
