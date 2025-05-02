from llama_index.core.readers.base import BaseReader
from llama_index.readers.upstage import UpstageLayoutAnalysisReader
from llama_index.readers.upstage import UpstageDocumentParseReader


def test_class():
    names_of_base_classes = [b.__name__ for b in UpstageLayoutAnalysisReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes

    names_of_base_classes = [b.__name__ for b in UpstageDocumentParseReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
