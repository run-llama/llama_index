import logging
import os
from datetime import datetime
from typing import List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class ZulipReader(BaseReader):
    """Zulip reader."""

    def __init__(
        self,
        zulip_email: str,
        zulip_domain: str,
        earliest_date: Optional[datetime] = None,
        latest_date: Optional[datetime] = None,
    ) -> None:
        import zulip

        """Initialize with parameters."""
        # Read the Zulip token from the environment variable
        zulip_token = os.environ.get("ZULIP_TOKEN")

        if zulip_token is None:
            raise ValueError("ZULIP_TOKEN environment variable not set.")

        # Initialize Zulip client with provided parameters
        self.client = zulip.Client(
            api_key=zulip_token, email=zulip_email, site=zulip_domain
        )

    def _read_stream(self, stream_name: str, reverse_chronological: bool) -> str:
        """Read a stream."""
        params = {
            "narrow": [{"operator": "stream", "operand": stream_name}],
            "anchor": "newest",
            "num_before": 100,
            "num_after": 0,
        }
        response = self.client.get_messages(params)
        messages = response["messages"]
        if reverse_chronological:
            messages.reverse()
        return " ".join([message["content"] for message in messages])

    def load_data(
        self, streams: List[str], reverse_chronological: bool = True
    ) -> List[Document]:
        """Load data from the input streams."""
        # Load data logic here
        data = []
        for stream_name in streams:
            stream_content = self._read_stream(stream_name, reverse_chronological)
            data.append(
                Document(text=stream_content, extra_info={"stream": stream_name})
            )
        return data

    def get_all_streams(self) -> list:
        # Fetch all streams
        response = self.client.get_streams()
        streams_data = response["streams"]
        # Collect the stream IDs
        return [stream["name"] for stream in streams_data]


if __name__ == "__main__":
    reader = ZulipReader(
        zulip_email="ianita-bot@plurigrid.zulipchat.com",
        zulip_domain="plurigrid.zulipchat.com",
    )
    logging.info(reader.load_data(reader.get_all_streams()))
