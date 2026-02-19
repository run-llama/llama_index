from typing import List
import pydoc
from llama_index.core.node_parser.interface import MetadataAwareTextSplitter


def test_metadataaware_init_docstring_preserved_in_help():
    class MySplitter(MetadataAwareTextSplitter):
        def __init__(self):
            """my custom doc"""
            super().__init__()

        def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
            return [text]

    rendered = pydoc.render_doc(MySplitter.__init__)
    assert "my custom doc" in rendered
    assert "Create a new model by parsing and validating" not in rendered
