"""Simple reader for mbox (mailbox) files."""
import os
from pathlib import Path
from typing import Any, List

from gpt_index.readers.base import BaseReader
from gpt_index.readers.file.mbox_parser import MboxParser
from gpt_index.readers.schema.base import Document


class MboxReader(BaseReader):
    """Mbox e-mail reader.

    Reads a set of e-mails saved in the mbox format.
    """

    def __init__(self) -> None:
        """Initialize."""

    def load_data(self, input_dir: str, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory.

        load_kwargs:
            max_count (int): Maximum amount of messages to read.
            message_format (str): Message format overriding default.
        """
        docs: List[Document] = []
        for (dirpath, dirnames, filenames) in os.walk(input_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.endswith(".mbox"):
                    filepath = os.path.join(dirpath, filename)
                    content = MboxParser(**load_kwargs).parse_file(Path(filepath))
                    for msg in content:
                        docs.append(Document(msg))
        return docs
