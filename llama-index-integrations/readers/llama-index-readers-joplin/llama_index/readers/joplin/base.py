"""
Joplin reader class.

When Joplin is installed and running it will parse all markdown
files into a List of Documents.

"""

import json
import os
import urllib
from datetime import datetime
from typing import Iterator, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import MarkdownReader
from llama_index.core.schema import Document

LINK_NOTE_TEMPLATE = "joplin://x-callback-url/openNote?id={id}"


class JoplinReader(BaseReader):
    """
    Reader that fetches notes from Joplin.

    In order to use this reader, you need to have Joplin running with the
    Web Clipper enabled (look for "Web Clipper" in the app settings).

    To get the access token, you need to go to the Web Clipper options and
    under "Advanced Options" you will find the access token. You may provide
    it as an argument or set the JOPLIN_ACCESS_TOKEN environment variable.

    You can find more information about the Web Clipper service here:
    https://joplinapp.org/clipper/
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        parse_markdown: bool = True,
        port: int = 41184,
        host: str = "localhost",
    ) -> None:
        """
        Initialize a new instance of JoplinReader.

        Args:
            access_token (Optional[str]): The access token for Joplin's Web Clipper service.
                If not provided, the JOPLIN_ACCESS_TOKEN environment variable is used. Default is None.
            parse_markdown (bool): Whether to parse the markdown content of the notes using MarkdownReader. Default is True.
            port (int): The port on which Joplin's Web Clipper service is running. Default is 41184.
            host (str): The host on which Joplin's Web Clipper service is running. Default is "localhost".

        """
        self.parse_markdown = parse_markdown
        if parse_markdown:
            self.parser = MarkdownReader()

        access_token = access_token or self._get_token_from_env()
        base_url = f"http://{host}:{port}"
        self._get_note_url = (
            f"{base_url}/notes?token={access_token}"
            "&fields=id,parent_id,title,body,created_time,updated_time&page={page}"
        )
        self._get_folder_url = (
            f"{base_url}/folders/{{id}}?token={access_token}&fields=title"
        )
        self._get_tag_url = (
            f"{base_url}/notes/{{id}}/tags?token={access_token}&fields=title"
        )

    def _get_token_from_env(self) -> str:
        if "JOPLIN_ACCESS_TOKEN" in os.environ:
            return os.environ["JOPLIN_ACCESS_TOKEN"]
        else:
            raise ValueError(
                "You need to provide an access token to use the Joplin reader. You may"
                " provide it as an argument or set the JOPLIN_ACCESS_TOKEN environment"
                " variable."
            )

    def _get_notes(self) -> Iterator[Document]:
        has_more = True
        page = 1
        while has_more:
            req_note = urllib.request.Request(self._get_note_url.format(page=page))
            with urllib.request.urlopen(req_note) as response:
                json_data = json.loads(response.read().decode())
                for note in json_data["items"]:
                    metadata = {
                        "source": LINK_NOTE_TEMPLATE.format(id=note["id"]),
                        "folder": self._get_folder(note["parent_id"]),
                        "tags": self._get_tags(note["id"]),
                        "title": note["title"],
                        "created_time": self._convert_date(note["created_time"]),
                        "updated_time": self._convert_date(note["updated_time"]),
                    }
                    if self.parse_markdown:
                        yield from self.parser.load_data(
                            None, content=note["body"], extra_info=metadata
                        )
                    else:
                        yield Document(text=note["body"], extra_info=metadata)

                has_more = json_data["has_more"]
                page += 1

    def _get_folder(self, folder_id: str) -> str:
        req_folder = urllib.request.Request(self._get_folder_url.format(id=folder_id))
        with urllib.request.urlopen(req_folder) as response:
            json_data = json.loads(response.read().decode())
            return json_data["title"]

    def _get_tags(self, note_id: str) -> List[str]:
        req_tag = urllib.request.Request(self._get_tag_url.format(id=note_id))
        with urllib.request.urlopen(req_tag) as response:
            json_data = json.loads(response.read().decode())
            return ",".join([tag["title"] for tag in json_data["items"]])

    def _convert_date(self, date: int) -> str:
        return datetime.fromtimestamp(date / 1000).strftime("%Y-%m-%d %H:%M:%S")

    def lazy_load(self) -> Iterator[Document]:
        yield from self._get_notes()

    def load_data(self) -> List[Document]:
        return list(self.lazy_load())
