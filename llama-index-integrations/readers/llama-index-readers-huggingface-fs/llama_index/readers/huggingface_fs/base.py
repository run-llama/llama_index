"""
Hugging Face file reader.

A parser for HF files.

"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import pandas as pd
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


class HuggingFaceFSReader(BaseReader):
    """
    Hugging Face File System reader.

    Uses the new Filesystem API from the Hugging Face Hub client library.
    """

    def __init__(self) -> None:
        from huggingface_hub import HfFileSystem

        self.fs = HfFileSystem()

    def load_dicts(self, path: str) -> List[Dict]:
        """Parse file."""
        test_data = self.fs.read_bytes(path)

        path = Path(path)
        if ".gz" in path.suffixes:
            import gzip

            with TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                with open(tmp / "tmp.jsonl.gz", "wb") as fp:
                    fp.write(test_data)

                f = gzip.open(tmp / "tmp.jsonl.gz", "rb")
                raw = f.read()
                data = raw.decode()
        else:
            data = test_data.decode()

        text_lines = data.split("\n")
        json_dicts = []
        for t in text_lines:
            try:
                json_dict = json.loads(t)
            except json.decoder.JSONDecodeError:
                continue
            json_dicts.append(json_dict)
        return json_dicts

    def load_df(self, path: str) -> pd.DataFrame:
        """Load pandas dataframe."""
        return pd.DataFrame(self.load_dicts(path))

    def load_data(self, path: str) -> List[Document]:
        """Load data."""
        json_dicts = self.load_dicts(path)
        docs = []
        for d in json_dicts:
            docs.append(Document(text=str(d)))
        return docs
