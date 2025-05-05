from typing import Any

from oxylabs.sources.response import Response

from llama_index.readers.oxylabs.google_base import OxylabsGoogleBaseReader


class OxylabsGoogleSearchReader(OxylabsGoogleBaseReader):
    """
    Get Google Search results data.

    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/google/search/search
    """

    def __init__(self, username: str, password: str, **data) -> None:
        super().__init__(username=username, password=password, **data)

    @classmethod
    def class_name(cls) -> str:
        return "OxylabsGoogleSearchReader"

    def get_response(self, payload: dict[str, Any]) -> Response:
        return self.oxylabs_api.google.scrape_search(**payload)

    async def aget_response(self, payload: dict[str, Any]) -> Response:
        return await self.async_oxylabs_api.google.scrape_search(**payload)
