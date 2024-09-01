# YoutubeMetaData.py
# Class to return Youtube Meta data for a video ID
import requests
from pydantic import Field
from typing import Any, List, Dict
from youtube_transcript_api import YouTubeTranscriptApi
from llama_index.core.readers.base import BasePydanticReader


class YouTubeMetaData(BasePydanticReader):
    api_key: str

    def load_data(self, video_ids):
        details = {}

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        video_id_chunks = list(chunks(video_ids, 20))
        for chunk in video_id_chunks:
            videos_string = ",".join(chunk)
            url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={videos_string}&key={self.api_key}"
            response = requests.get(url).json()
            if "items" not in response:
                print("Error in API response:", response)
                continue

            for item in response["items"]:
                video_id = item["id"]
                details[video_id] = {
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "publishDate": item["snippet"]["publishedAt"],
                    "statistics": item["statistics"],
                    "tags": item["snippet"].get("tags", []),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                }

        return details


class YouTubeMetaDataAndTranscript(BasePydanticReader):
    api_key: str = Field(..., description="API key for YouTube data access")
    metadata_loader: YouTubeMetaData = None  # Don't instantiate here
    transcript_loader: Any = YouTubeTranscriptApi  # Assume this is a simple callable

    def initialize_loaders(self):
        if not self.metadata_loader:
            self.metadata_loader = YouTubeMetaData(api_key=self.api_key)

    def load_data(self, video_ids: List[str]) -> Dict[str, Any]:
        self.initialize_loaders()  # Make sure loaders are initialized
        all_details = {}
        for video_id in video_ids:
            metadata = self.metadata_loader.load_data([video_id])
            try:
                transcripts = self.transcript_loader.get_transcript(video_id)
            except Exception as e:
                transcripts = str(e)
            all_details[video_id] = {
                "metadata": metadata.get(video_id, {}),
                "transcript": transcripts,
            }
        return all_details
