import inspect
from llama_index.node_parser.chonkie import Chunker


def test_chunker_init_docstring_is_present():
    doc = inspect.getdoc(Chunker.__init__) or ""
    assert "Initialize with a Chonkie chunker instance" in doc