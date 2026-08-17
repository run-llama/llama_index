import gzip
import json
from unittest.mock import MagicMock

from llama_index.core.readers.base import BaseReader
from llama_index.readers.huggingface_fs import HuggingFaceFSReader


def test_class():
    names_of_base_classes = [b.__name__ for b in HuggingFaceFSReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_load_dicts_from_gzipped_file():
    reader = HuggingFaceFSReader()
    reader.fs = MagicMock()
    lines = "\n".join([json.dumps({"a": 1}), json.dumps({"a": 2})]).encode()
    reader.fs.read_bytes.return_value = gzip.compress(lines)

    result = reader.load_dicts("hf://datasets/example/file.jsonl.gz")

    assert result == [{"a": 1}, {"a": 2}]
