from llama_index.core.readers.base import BaseReader
from llama_index.readers.builtsimple import (
    BuiltSimplePubMedReader,
    BuiltSimpleArxivReader,
    BuiltSimpleWikipediaReader,
)


def test_pubmed_class():
    names_of_base_classes = [b.__name__ for b in BuiltSimplePubMedReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_arxiv_class():
    names_of_base_classes = [b.__name__ for b in BuiltSimpleArxivReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes


def test_wikipedia_class():
    names_of_base_classes = [b.__name__ for b in BuiltSimpleWikipediaReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
