import logging
from pathlib import Path
from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class FacebookMessengerLoader(BaseReader):
    """
    Facebook Messenger chat data loader.

    Args:
        path (str): Path to Facebook Messenger chat file.
    """

    def __init__(self, path: str):
        """Initialize with path."""
        self.file_path = path

    def load_data(self) -> List[Document]:
        """
        Parse Facebook Messenger file into Documents.
        """
        import json

        path = Path(self.file_path)

        # Load the Facebook Messenger chat data
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logging.debug(f"> Number of messages: {len(data['messages'])}.")

        docs = []
        n = 0

        # Parsing through messages
        for message in data["messages"]:
            # Check for required fields in message
            author = message.get("sender_name", "Unknown")
            timestamp = message.get("timestamp_ms", "")
            text = message.get("content", "")

            extra_info = {
                "source": str(path).split("/")[-1].replace(".json", ""),
                "author": author,
                "timestamp": str(timestamp),
            }

            docs.append(
                Document(
                    text=str(timestamp) + " " + author + ": " + text,
                    extra_info=extra_info,
                )
            )

            n += 1
            logging.debug(f"Added {n} of {len(data['messages'])} messages.")

        logging.debug(f"> Document creation for {path} is complete.")
        return docs
