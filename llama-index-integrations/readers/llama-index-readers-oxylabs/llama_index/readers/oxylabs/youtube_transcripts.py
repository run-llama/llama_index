from typing import Any

from llama_index.readers.oxylabs.base import OxylabsBaseReader
from oxylabs.sources.response import Response


class OxylabsYoutubeTranscriptReader(OxylabsBaseReader):
    """
    Get YouTube video transcripts.

    https://developers.oxylabs.io/scraper-apis/web-scraper-api/targets/youtube/youtube-transcript
    """

    top_level_header: str = "YouTube video transcripts"

    def __init__(self, username: str, password: str, **data) -> None:
        super().__init__(username=username, password=password, **data)

    @classmethod
    def class_name(cls) -> str:
        return "OxylabsYoutubeTranscriptReader"

    def get_response(self, payload: dict[str, Any]) -> Response:
        return self.oxylabs_api.youtube_transcript.scrape_transcript(**payload)

    async def aget_response(self, payload: dict[str, Any]) -> Response:
        return await self.async_oxylabs_api.youtube_transcript.scrape_transcript(
            **payload
        )
