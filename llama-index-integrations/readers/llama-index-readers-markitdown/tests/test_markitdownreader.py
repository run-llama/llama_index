from llama_index.core.readers.base import BaseReader
from llama_index.readers.markitdown import MarkItDownReader


def test_class():
    reader = MarkItDownReader()
    assert isinstance(reader, BaseReader)
    assert reader._reader is not None
