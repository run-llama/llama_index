"""Simple Reader that reads transcript of youtube video."""
import re
from typing import Any, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from importlib.util import find_spec

from llama_index.readers.youtube_transcript.utils import YOUTUBE_URL_PATTERNS


class YoutubeTranscriptReader(BaseReader):
    """Youtube Transcript reader."""

    def __init__(self) -> None:
        if find_spec("youtube_transcript_api") is None:
            raise ImportError(
                "Missing package: youtube_transcript_api.\n"
                "Please `pip install youtube_transcript_api` to use this Reader"
            )
        super().__init__()

    def load_data(
        self,
        ytlinks: List[str],
        languages: Optional[List[str]] = ["en"],
        **load_kwargs: Any,
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            pages (List[str]): List of youtube links \
                for which transcripts are to be read.

        """
        from youtube_transcript_api import YouTubeTranscriptApi

        results = []
        for link in ytlinks:
            video_id = self._extract_video_id(link)
            if not video_id:
                raise ValueError(
                    f"Supplied url {link} is not a supported youtube URL."
                    "Supported formats include:"
                    "  youtube.com/watch?v=\{video_id\} "
                    "(with or without 'www.')\n"
                    "  youtube.com/embed?v=\{video_id\} "
                    "(with or without 'www.')\n"
                    "  youtu.be/{video_id\} (never includes www subdomain)"
                )
            transcript_chunks = YouTubeTranscriptApi.get_transcript(
                video_id, languages=languages
            )
            chunk_text = [chunk["text"] for chunk in transcript_chunks]
            transcript = "\n".join(chunk_text)
            results.append(Document(text=transcript, extra_info={"video_id": video_id}))
        return results

    @staticmethod
    def _extract_video_id(yt_link) -> Optional[str]:
        for pattern in YOUTUBE_URL_PATTERNS:
            match = re.search(pattern, yt_link)
            if match:
                return match.group(1)

        # return None if no match is found
        return None
