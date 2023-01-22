"""Simple Reader that reads transcript of youtube video."""
from typing import Any, List

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class YoutubeTranscriptReader(BaseReader):
    """Youtube Transcript reader."""

    def __init__(self) -> None:
        """Initialize with parameters."""

    def load_data(self, ytlinks: List[str], **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        Args:
            pages (List[str]): List of youtube links \
                for which transcripts are to be read.

        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ValueError(
                "`youtube_transcript_api` package not found, \
                    please run `pip install youtube-transcript-api`"
            )

        results = []
        for link in ytlinks:
            video_id = link.split("?v=")[-1]
            srt = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = ""
            for chunk in srt:
                transcript = transcript + chunk["text"] + "\n"
            results.append(Document(transcript))
        return results
