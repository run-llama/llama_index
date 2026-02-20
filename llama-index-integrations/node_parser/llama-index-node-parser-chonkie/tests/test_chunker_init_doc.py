import pydoc
from llama_index.node_parser.chonkie import Chunker


def test_chunker_init_docstring_is_present_in_help():
    rendered = pydoc.render_doc(Chunker.__init__).lower()

    assert "initialize with a chonkie chunker instance" in rendered
    assert "create a new model by parsing and validating" not in rendered
    