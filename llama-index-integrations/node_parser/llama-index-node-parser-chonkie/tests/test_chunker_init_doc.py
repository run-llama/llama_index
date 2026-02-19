import pydoc
from llama_index.node_parser.chonkie import Chunker


def test_chunker_init_docstring_is_present_in_help():
    rendered = pydoc.render_doc(Chunker.__init__)
    assert "Initialize with a Chonkie chunker instance" in rendered
    assert "Create a new model by parsing and validating" not in rendered
