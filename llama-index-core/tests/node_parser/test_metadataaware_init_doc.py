import inspect
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter


def test_metadataaware_docstring_preserved():
    class MySplitter(MetadataAwareTextSplitter):
        def __init__(self, *args, **kwargs):
            """My custom doc"""
            super().__init__(*args, **kwargs)

    doc = inspect.getdoc(MySplitter.__init__) or ""
    assert "my custom doc" in doc
    assert "Create a new model by parsing and validating" not in doc
