from llama_index.core.readers.base import BaseReader
from llama_index.readers.astra_db import AstraDBReader


def test_class() -> None:
    names_of_base_classes = [b.__name__ for b in AstraDBReader.__mro__]
    assert BaseReader.__name__ in names_of_base_classes
