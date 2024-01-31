from llama_index.core.readers.base import BaseReader
from llama_index.readers.airtable import AirtableReader


def test_class():
    names_of_base_classes = [b.__name__ for b in AirtableReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
