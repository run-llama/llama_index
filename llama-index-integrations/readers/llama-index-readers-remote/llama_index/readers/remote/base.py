"""
Remote file reader.

A loader that fetches an arbitrary remote page or file by URL and parses its contents.

"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader


class RemoteReader(BaseReader):
    """General reader for any remote page or file."""

    def __init__(
        self,
        *args: Any,
        file_extractor: Optional[Dict[str, Union[str, BaseReader]]] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(*args, **kwargs)

        self.file_extractor = file_extractor

    @staticmethod
    def _is_youtube_video(url: str) -> bool:
        # TODO create more global method for detecting all types
        """
        Returns True if the given URL is a video on YouTube, False otherwise.
        """
        # Regular expression pattern to match YouTube video URLs
        youtube_pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?([^\s&]+)"

        # Match the pattern against the URL
        match = re.match(youtube_pattern, url)

        # If there's a match, it's a YouTube video URL
        return match is not None

    def load_data(self, url: str) -> List[Document]:
        """Parse whatever is at the URL."""
        import tempfile
        from urllib.parse import urlparse
        from urllib.request import Request, urlopen

        # check the URL
        parsed_url = urlparse(url)

        # Check if the scheme is http or https
        if parsed_url.scheme not in (
            "http",
            "https",
            "ftp",
            "ws",
            "wss",
            "sftp",
            "ftps",
            "s3",
        ):
            raise ValueError(
                "Invalid URL scheme. Only http, https, ftp, ftps, sftp, ws, wss, and s3 are allowed."
            )

        extra_info = {"Source": url}

        req = Request(url, headers={"User-Agent": "Magic Browser"})
        result = urlopen(req)
        url_type = result.info().get_content_type()
        documents = []
        if url_type == "text/html" or url_type == "text/plain":
            text = "\n\n".join([str(el.decode("utf-8-sig")) for el in result])
            documents = [Document(text=text, extra_info=extra_info)]
        elif self._is_youtube_video(url):
            youtube_reader = YoutubeTranscriptReader()
            # TODO should we have another language, like english / french?
            documents = youtube_reader.load_data([url])
        else:
            suffix = Path(urlparse(url).path).suffix
            with tempfile.TemporaryDirectory() as temp_dir:
                filepath = f"{temp_dir}/temp{suffix}"
                with open(filepath, "wb") as output:
                    output.write(result.read())
                loader = SimpleDirectoryReader(
                    temp_dir,
                    file_metadata=(lambda _: extra_info),
                    file_extractor=self.file_extractor,
                )
                documents = loader.load_data()
        return documents
